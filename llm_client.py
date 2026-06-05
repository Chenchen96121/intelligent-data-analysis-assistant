from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from analyzer import AnalysisSummary
from data_processor import ColumnTypes, QualityReport


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int = 30
    max_sample_rows: int = 5


class OpenAICompatibleLLMClient:
    """Small client for OpenAI-compatible Chat Completions APIs."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def answer_question(
        self,
        question: str,
        df: pd.DataFrame,
        quality_report: QualityReport,
        column_types: ColumnTypes,
        analysis: AnalysisSummary,
    ) -> str:
        endpoint = _build_chat_completions_url(self.config.base_url)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个数据分析助手。请只基于用户提供的数据摘要回答，"
                        "不要编造数据中不存在的结论。回答要简洁、清楚，适合非技术用户理解。"
                    ),
                },
                {
                    "role": "user",
                    "content": _build_analysis_prompt(
                        question=question,
                        df=df,
                        quality_report=quality_report,
                        column_types=column_types,
                        analysis=analysis,
                        max_sample_rows=self.config.max_sample_rows,
                    ),
                },
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = _extract_error_message(response)
            raise RuntimeError(f"API 请求失败：{response.status_code}，{message}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"API 连接失败：{exc}") from exc

        data = response.json()
        return _extract_answer(data)

    def test_connection(self) -> str:
        endpoint = _build_chat_completions_url(self.config.base_url)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个 API 连接测试助手。请用中文简短回复。",
                },
                {
                    "role": "user",
                    "content": "请回复：API 大模型连接成功。",
                },
            ],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = _extract_error_message(response)
            raise RuntimeError(f"API 测试失败：{response.status_code}，{message}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"API 连接失败：{exc}") from exc

        return _extract_answer(response.json())


def _build_chat_completions_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _build_analysis_prompt(
    question: str,
    df: pd.DataFrame,
    quality_report: QualityReport,
    column_types: ColumnTypes,
    analysis: AnalysisSummary,
    max_sample_rows: int,
) -> str:
    sample_rows = df.head(max_sample_rows).to_dict(orient="records")
    descriptive_stats = _frame_to_records(analysis.descriptive_stats, limit=20)
    missing_summary = _frame_to_records(quality_report.missing_summary, limit=20)
    outlier_summary = _frame_to_records(quality_report.outlier_summary, limit=20)
    category_summary = _frame_to_records(analysis.category_summaries, limit=30)

    return (
        f"用户问题：{question}\n\n"
        "数据摘要：\n"
        f"- 数据行数：{len(df)}\n"
        f"- 字段数量：{df.shape[1]}\n"
        f"- 数值字段：{column_types.numeric}\n"
        f"- 类别字段：{column_types.categorical}\n"
        f"- 日期字段：{column_types.datetime}\n"
        f"- 重复行数量：{quality_report.duplicate_rows}\n\n"
        f"描述统计：{descriptive_stats}\n\n"
        f"缺失值统计：{missing_summary}\n\n"
        f"异常值统计：{outlier_summary}\n\n"
        f"类别频数摘要：{category_summary}\n\n"
        f"样例数据（前 {max_sample_rows} 行）：{sample_rows}\n\n"
        "请基于以上信息回答用户问题。"
    )


def _frame_to_records(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = df.head(limit).where(pd.notna(df.head(limit)), None).to_dict(orient="records")
    return records


def _extract_answer(data: dict[str, Any]) -> str:
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"API 返回格式不符合 Chat Completions 规范：{data}") from exc

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("API 返回了空回答。")
    return answer.strip()


def _extract_error_message(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:300]

    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(data)[:300]
