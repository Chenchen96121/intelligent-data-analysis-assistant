from dataclasses import dataclass
import json
import re
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


@dataclass(frozen=True)
class QueryIntent:
    intent_type: str
    target_columns: list[str]
    metrics: list[str]
    dimensions: list[str]
    filters: list[str]
    response_style: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_type": self.intent_type,
            "target_columns": self.target_columns,
            "metrics": self.metrics,
            "dimensions": self.dimensions,
            "filters": self.filters,
            "response_style": self.response_style,
        }


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
                    "content": _build_low_cost_system_prompt(),
                },
                {
                    "role": "user",
                    "content": _build_compact_analysis_prompt(
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


def build_structured_query_intent(
    question: str,
    df: pd.DataFrame,
    quality_report: QualityReport,
    column_types: ColumnTypes,
) -> QueryIntent:
    normalized_question = question.strip().lower()
    target_columns = _find_columns_in_question(question, df.columns.tolist())
    metrics = _extract_metrics(normalized_question)
    intent_type = _infer_intent_type(normalized_question, metrics)
    dimensions = _infer_dimensions(intent_type, target_columns, column_types)
    filters = _extract_simple_filters(question, df.columns.tolist())
    response_style = "detailed" if any(word in normalized_question for word in ["详细", "原因", "为什么", "分析"]) else "concise"

    if not target_columns:
        target_columns = _fallback_target_columns(intent_type, quality_report, column_types)

    return QueryIntent(
        intent_type=intent_type,
        target_columns=target_columns[:5],
        metrics=metrics[:5],
        dimensions=dimensions[:5],
        filters=filters[:5],
        response_style=response_style,
    )


def _build_low_cost_system_prompt() -> str:
    return (
        "你是面向表格数据的数据分析问答助手。你的任务是用低成本方式回答问题："
        "先根据用户问题识别结构化查询意图，再只基于提供的压缩数据上下文回答。"
        "不要编造未提供的数据，不要要求查看完整原始表。"
        "输出必须包含两部分：\n"
        "1. 查询意图：用一行紧凑 JSON 表示，字段为 intent_type、target_columns、metrics、dimensions、filters。\n"
        "2. 回答：用中文给出简洁结论，必要时说明依据的数据摘要。"
    )


def _build_compact_analysis_prompt(
    question: str,
    df: pd.DataFrame,
    quality_report: QualityReport,
    column_types: ColumnTypes,
    analysis: AnalysisSummary,
    max_sample_rows: int,
) -> str:
    intent = build_structured_query_intent(question, df, quality_report, column_types)
    context = _build_compact_data_context(
        df=df,
        quality_report=quality_report,
        column_types=column_types,
        analysis=analysis,
        intent=intent,
        max_sample_rows=max_sample_rows,
    )

    return (
        "QUESTION:\n"
        f"{question}\n\n"
        "LOCAL_INTENT:\n"
        f"{_compact_json(intent.to_dict())}\n\n"
        "DATA_CONTEXT:\n"
        f"{_compact_json(context)}\n\n"
        "RESPONSE_RULES:\n"
        "- 如果 LOCAL_INTENT 已经足够明确，请沿用它；如需修正，只能基于 QUESTION 和 DATA_CONTEXT。\n"
        "- 优先使用 DATA_CONTEXT 中的统计值、缺失值、异常值和字段信息。\n"
        "- 输出格式固定为：查询意图：{...}\\n回答：...\n"
        "- 回答控制在 120 字以内，除非用户明确要求详细分析。"
    )


def _build_compact_data_context(
    df: pd.DataFrame,
    quality_report: QualityReport,
    column_types: ColumnTypes,
    analysis: AnalysisSummary,
    intent: QueryIntent,
    max_sample_rows: int,
) -> dict[str, Any]:
    relevant_columns = _relevant_columns_for_context(intent, column_types)
    return {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "numeric": _limit_list(column_types.numeric, 10),
            "categorical": _limit_list(column_types.categorical, 10),
            "datetime": _limit_list(column_types.datetime, 10),
        },
        "quality": {
            "duplicate_rows": int(quality_report.duplicate_rows),
            "top_missing": _top_records(
                quality_report.missing_summary,
                count_column="缺失值数量",
                target_columns=relevant_columns,
                limit=5,
            ),
            "top_outliers": _top_records(
                quality_report.outlier_summary,
                count_column="异常值数量",
                target_columns=relevant_columns,
                limit=5,
            ),
        },
        "statistics": _filtered_records(
            analysis.descriptive_stats,
            target_columns=relevant_columns,
            column_name="字段",
            limit=6,
        ),
        "categories": _filtered_records(
            analysis.category_summaries,
            target_columns=relevant_columns,
            column_name="字段",
            limit=8,
        ),
        "time": _filtered_records(
            analysis.time_summaries,
            target_columns=relevant_columns,
            column_name="日期字段",
            limit=3,
        ),
        "sample": _sample_relevant_rows(df, relevant_columns, max_sample_rows),
    }


def _frame_to_records(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = df.head(limit).where(pd.notna(df.head(limit)), None).to_dict(orient="records")
    return records


def _find_columns_in_question(question: str, columns: list[str]) -> list[str]:
    normalized = question.lower()
    matches = []
    for column in sorted(columns, key=len, reverse=True):
        if column and column.lower() in normalized:
            matches.append(column)
    return matches


def _extract_metrics(normalized_question: str) -> list[str]:
    metric_keywords = [
        ("mean", ["平均", "均值", "mean", "average"]),
        ("median", ["中位", "median"]),
        ("max", ["最大", "最高", "max"]),
        ("min", ["最小", "最低", "min"]),
        ("std", ["标准差", "波动", "std"]),
        ("missing_count", ["缺失", "missing"]),
        ("outlier_count", ["异常", "outlier"]),
        ("count", ["数量", "频数", "分布", "count"]),
        ("trend", ["趋势", "变化", "走势", "trend"]),
        ("correlation", ["关系", "相关", "影响", "correlation"]),
    ]
    metrics = []
    for metric, keywords in metric_keywords:
        if any(keyword in normalized_question for keyword in keywords):
            metrics.append(metric)
    return metrics or ["summary"]


def _infer_intent_type(normalized_question: str, metrics: list[str]) -> str:
    if "missing_count" in metrics:
        return "missing_value_check"
    if "outlier_count" in metrics:
        return "outlier_check"
    if "trend" in metrics:
        return "trend_analysis"
    if "correlation" in metrics:
        return "relationship_analysis"
    if "count" in metrics and any(word in normalized_question for word in ["分布", "占比", "类别", "频数"]):
        return "category_distribution"
    if any(metric in metrics for metric in ["mean", "median", "max", "min", "std"]):
        return "descriptive_metric"
    if any(word in normalized_question for word in ["图", "可视化", "chart", "plot"]):
        return "chart_recommendation"
    if any(word in normalized_question for word in ["总结", "概览", "summary", "主要问题"]):
        return "dataset_summary"
    return "general_analysis"


def _infer_dimensions(
    intent_type: str,
    target_columns: list[str],
    column_types: ColumnTypes,
) -> list[str]:
    dimensions = []
    if intent_type == "trend_analysis":
        dimensions.extend(column_types.datetime[:1])
    if intent_type in {"category_distribution", "dataset_summary", "general_analysis"}:
        dimensions.extend(column for column in column_types.categorical[:2] if column not in dimensions)
    dimensions.extend(column for column in target_columns if column not in dimensions)
    return dimensions


def _extract_simple_filters(question: str, columns: list[str]) -> list[str]:
    filters = []
    for column in columns:
        pattern = rf"{re.escape(column)}\s*(=|为|是|等于)\s*([^\s，,。；;]+)"
        match = re.search(pattern, question)
        if match:
            filters.append(f"{column}{match.group(1)}{match.group(2)}")
    return filters


def _fallback_target_columns(
    intent_type: str,
    quality_report: QualityReport,
    column_types: ColumnTypes,
) -> list[str]:
    if intent_type == "missing_value_check" and not quality_report.missing_summary.empty:
        return quality_report.missing_summary["字段"].head(3).tolist()
    if intent_type == "outlier_check" and not quality_report.outlier_summary.empty:
        return quality_report.outlier_summary["字段"].head(3).tolist()
    if intent_type == "trend_analysis":
        return column_types.datetime[:1] + column_types.numeric[:2]
    if intent_type == "category_distribution":
        return column_types.categorical[:2]
    return column_types.numeric[:3] or column_types.categorical[:3] or column_types.datetime[:3]


def _relevant_columns_for_context(intent: QueryIntent, column_types: ColumnTypes) -> list[str]:
    columns = []
    for column in [*intent.target_columns, *intent.dimensions]:
        if column not in columns:
            columns.append(column)
    if not columns:
        columns.extend(column_types.numeric[:3])
        columns.extend(column_types.categorical[:2])
        columns.extend(column_types.datetime[:1])
    return columns[:6]


def _limit_list(values: list[str], limit: int) -> list[str]:
    return values[:limit]


def _top_records(
    df: pd.DataFrame,
    count_column: str,
    target_columns: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    filtered = df
    if target_columns and "字段" in df.columns:
        matched = df[df["字段"].isin(target_columns)]
        if not matched.empty:
            filtered = matched
    if count_column in filtered.columns:
        filtered = filtered[filtered[count_column] > 0]
    return _frame_to_records(filtered.head(limit), limit)


def _filtered_records(
    df: pd.DataFrame,
    target_columns: list[str],
    column_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    filtered = df
    if target_columns and column_name in df.columns:
        matched = df[df[column_name].isin(target_columns)]
        if not matched.empty:
            filtered = matched
    return _frame_to_records(filtered.head(limit), limit)


def _sample_relevant_rows(
    df: pd.DataFrame,
    relevant_columns: list[str],
    max_sample_rows: int,
) -> list[dict[str, Any]]:
    columns = [column for column in relevant_columns if column in df.columns]
    if not columns:
        return []
    sample_size = min(max_sample_rows, 3)
    return df[columns].head(sample_size).where(pd.notna(df[columns].head(sample_size)), None).to_dict(orient="records")


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


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
