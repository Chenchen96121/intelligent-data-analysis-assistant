from io import BytesIO

import pandas as pd

from analyzer import build_analysis_summary
from data_processor import clean_data, detect_column_types, read_csv_file, read_data_file, read_excel_file
from llm_client import LLMConfig, OpenAICompatibleLLMClient
from qa_engine import RuleBasedQAEngine
from report_generator import generate_excel_report


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-03"],
            "地区": ["华东", "华南", None, None],
            "销售额": [100, 120, None, 10000],
            "订单数": [10, 12, 11, 10],
        }
    )


def test_clean_data_detects_missing_duplicates_and_outliers() -> None:
    df = make_sample_df()
    cleaned, quality = clean_data(df)

    assert len(cleaned) == 4
    assert quality.missing_summary["缺失值数量"].sum() == 3
    assert "销售额" in quality.outlier_summary["字段"].tolist()


def test_analysis_and_report_generation() -> None:
    cleaned, quality = clean_data(make_sample_df())
    column_types = detect_column_types(cleaned)
    analysis = build_analysis_summary(cleaned, quality, column_types)
    report = generate_excel_report(cleaned, quality, analysis)

    assert "销售额" in analysis.descriptive_stats["字段"].tolist()
    assert len(report) > 1000


def test_rule_based_qa_answers_known_questions() -> None:
    cleaned, quality = clean_data(make_sample_df())
    column_types = detect_column_types(cleaned)
    analysis = build_analysis_summary(cleaned, quality, column_types)
    qa = RuleBasedQAEngine(cleaned, quality, column_types, analysis)

    assert "缺失值最多" in qa.answer("缺失值最多的列是什么？")
    assert "异常值最多" in qa.answer("哪个字段异常值最多？")
    assert "均值" in qa.answer("销售额平均值是多少？")


def test_read_excel_file_deduplicates_column_names() -> None:
    df = pd.DataFrame([[1, 2]], columns=["字段", "字段"])
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    result = read_excel_file(buffer)

    assert result.columns.tolist() == ["字段", "字段.1"]


def test_read_data_file_supports_csv(tmp_path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("地区,销售额\n华东,100\n华南,120\n", encoding="utf-8-sig")

    result = read_data_file(str(csv_path))

    assert result.shape == (2, 2)
    assert result["销售额"].tolist() == [100, 120]


def test_read_csv_file_supports_gbk(tmp_path) -> None:
    csv_path = tmp_path / "sample_gbk.csv"
    csv_path.write_bytes("地区,销售额\n华东,100\n".encode("gbk"))

    result = read_csv_file(str(csv_path))

    assert result.iloc[0]["地区"] == "华东"


def test_openai_compatible_llm_client_builds_chat_request(monkeypatch) -> None:
    cleaned, quality = clean_data(make_sample_df())
    column_types = detect_column_types(cleaned)
    analysis = build_analysis_summary(cleaned, quality, column_types)
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "这是一段 API 大模型回答。"}}]}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("llm_client.requests.post", fake_post)
    client = OpenAICompatibleLLMClient(
        LLMConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test-model",
            timeout=10,
        )
    )

    answer = client.answer_question("请总结数据", cleaned, quality, column_types, analysis)

    assert answer == "这是一段 API 大模型回答。"
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "test-model"


def test_openai_compatible_llm_client_tests_connection(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "API 大模型连接成功。"}}]}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("llm_client.requests.post", fake_post)
    client = OpenAICompatibleLLMClient(
        LLMConfig(api_key="test-key", base_url="https://example.com/v1", model="test-model")
    )

    assert client.test_connection() == "API 大模型连接成功。"
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["json"]["messages"][1]["content"] == "请回复：API 大模型连接成功。"
