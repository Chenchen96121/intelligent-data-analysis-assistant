import streamlit as st

from analyzer import build_analysis_summary, create_plotly_charts
from data_processor import clean_data, detect_column_types, read_data_file
from llm_client import LLMConfig, OpenAICompatibleLLMClient
from qa_engine import RuleBasedQAEngine
from report_generator import generate_excel_report


st.set_page_config(
    page_title="智能数据分析助手",
    page_icon="📊",
    layout="wide",
)


def render_empty_state() -> None:
    st.title("智能数据分析助手")
    st.caption("可以先在左侧接入 API 大模型，测试成功后再上传 Excel / CSV 文件开始分析。")
    st.info("请选择一个 .xlsx、.xls 或 .csv 文件开始分析。")


def render_llm_setup() -> LLMConfig | None:
    with st.sidebar.expander("API 大模型接入", expanded=True):
        st.caption("可先测试 AI 模型连接，再上传数据文件。API Key 只在当前页面会话中使用。")
        api_key = st.text_input("API Key", type="password", placeholder="粘贴你的 API Key")
        base_url = st.text_input(
            "接口地址 Base URL",
            value="https://api.openai.com/v1",
            help="OpenAI 官方接口保持默认；其他兼容服务请填写服务商提供的 Base URL。",
        )
        model = st.text_input(
            "模型名称",
            value="gpt-4o-mini",
            help="填写服务商支持的模型名。",
        )
        max_sample_rows = st.slider("发送给模型的样例数据行数", 3, 20, 5)

        config_ready = bool(api_key.strip() and base_url.strip() and model.strip())
        if st.button("测试 API 连接", disabled=not config_ready):
            try:
                client = OpenAICompatibleLLMClient(
                    LLMConfig(
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        max_sample_rows=max_sample_rows,
                    )
                )
                with st.spinner("正在测试 API 连接..."):
                    answer = client.test_connection()
                st.success(answer)
                st.session_state["llm_connected"] = True
            except Exception as exc:
                st.session_state["llm_connected"] = False
                st.error(str(exc))

        if not config_ready:
            st.caption("填写 API Key、Base URL 和模型名称后即可测试连接。")
            return None

        return LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_sample_rows=max_sample_rows,
        )


def render_api_tutorial() -> None:
    with st.sidebar.expander("API 大模型接入教程", expanded=False):
        st.markdown(
            """
            **1. 准备 API Key**

            - OpenAI 官方接口：进入 OpenAI 平台创建 API Key。
            - 其他大模型服务：找到服务商提供的 API Key、Base URL 和模型名称。
            - 不要把 API Key 写进代码、README 或上传到 GitHub。

            **2. 先在左侧填写并测试**

            - 在左侧 `API 大模型接入` 中填写 API Key、Base URL 和模型名称。
            - 点击 `测试 API 连接`。
            - 测试成功后，再上传 Excel / CSV 文件进行数据问答。

            **3. 常见示例**

            - OpenAI Base URL：`https://api.openai.com/v1`
            - 第三方兼容接口：通常类似 `https://xxx.com/v1`
            - 模型名称必须和服务商文档一致。

            **4. 测试问题**

            - `这份数据主要有什么问题？`
            - `缺失值最多的字段是什么？`
            - `异常值最多的字段是什么？`
            - `请用三句话总结这份数据。`
            """
        )


def main() -> None:
    st.title("智能数据分析助手")
    st.caption("Excel / CSV 自动清洗 · 统计分析 · 可视化报告 · 自然语言问答")

    uploaded_file = st.sidebar.file_uploader(
        "上传 Excel / CSV 文件",
        type=["xlsx", "xls", "csv"],
        help="Excel 默认读取第一个工作表；CSV 支持常见 UTF-8 / GBK 中文编码。",
    )
    llm_config = render_llm_setup()
    render_api_tutorial()

    if uploaded_file is None:
        render_empty_state()
        return

    try:
        raw_df = read_data_file(uploaded_file)
    except Exception as exc:
        st.error(f"文件读取失败：{exc}")
        return

    if raw_df.empty:
        st.warning("上传的文件没有可分析的数据。")
        return

    cleaned_df, quality_report = clean_data(raw_df)
    column_types = detect_column_types(cleaned_df)
    analysis = build_analysis_summary(cleaned_df, quality_report, column_types)
    qa_engine = RuleBasedQAEngine(cleaned_df, quality_report, column_types, analysis)

    overview_cols = st.columns(4)
    overview_cols[0].metric("原始行数", f"{len(raw_df):,}")
    overview_cols[1].metric("清洗后行数", f"{len(cleaned_df):,}")
    overview_cols[2].metric("字段数量", f"{cleaned_df.shape[1]:,}")
    overview_cols[3].metric("重复行", f"{quality_report.duplicate_rows:,}")

    tab_preview, tab_quality, tab_analysis, tab_charts, tab_qa, tab_report = st.tabs(
        ["数据预览", "数据质量", "统计分析", "可视化图表", "自然语言问答", "报告导出"]
    )

    with tab_preview:
        st.subheader("清洗后数据")
        st.dataframe(cleaned_df, use_container_width=True)

        st.subheader("字段识别")
        type_cols = st.columns(3)
        type_cols[0].write("数值列")
        type_cols[0].dataframe(column_types.numeric, use_container_width=True)
        type_cols[1].write("类别列")
        type_cols[1].dataframe(column_types.categorical, use_container_width=True)
        type_cols[2].write("日期列")
        type_cols[2].dataframe(column_types.datetime, use_container_width=True)

    with tab_quality:
        st.subheader("缺失值统计")
        st.dataframe(quality_report.missing_summary, use_container_width=True)
        st.subheader("异常值统计")
        st.dataframe(quality_report.outlier_summary, use_container_width=True)

    with tab_analysis:
        st.subheader("描述统计")
        st.dataframe(analysis.descriptive_stats, use_container_width=True)

        if not analysis.category_summaries.empty:
            st.subheader("类别字段频数 Top 10")
            st.dataframe(analysis.category_summaries, use_container_width=True)

        if not analysis.time_summaries.empty:
            st.subheader("日期字段趋势摘要")
            st.dataframe(analysis.time_summaries, use_container_width=True)

    with tab_charts:
        st.subheader("自动推荐图表")
        charts = create_plotly_charts(cleaned_df, column_types)
        if not charts:
            st.info("当前数据字段不足，暂未生成推荐图表。")
        for title, chart in charts:
            st.plotly_chart(chart, use_container_width=True)

    with tab_qa:
        st.subheader("自然语言数据问答")
        qa_mode = st.radio(
            "选择问答模式",
            ["规则模式（无需 API）", "API 大模型模式（测试）"],
            horizontal=True,
        )
        question = st.text_input(
            "输入你的问题",
            placeholder="例如：缺失值最多的列是什么？销售额平均值是多少？生成数据总结",
        )

        if qa_mode == "规则模式（无需 API）":
            if question:
                st.write(qa_engine.answer(question))
            st.caption("当前为规则式 Mock 问答，不需要 API Key，适合本地演示和基础测试。")
        else:
            if llm_config is None:
                st.warning("请先在左侧「API 大模型接入」中填写配置并测试连接。")
            elif st.session_state.get("llm_connected"):
                st.success("已检测到 API 连接测试成功，可开始数据问答。")
            else:
                st.info("你已经填写了 API 配置，建议先点击左侧「测试 API 连接」。")

            if question:
                if llm_config is not None:
                    with st.spinner("正在调用 API 大模型分析数据..."):
                        try:
                            llm_client = OpenAICompatibleLLMClient(llm_config)
                            st.write(
                                llm_client.answer_question(
                                    question,
                                    cleaned_df,
                                    quality_report,
                                    column_types,
                                    analysis,
                                )
                            )
                        except Exception as exc:
                            st.error(str(exc))

    with tab_report:
        st.subheader("下载 Excel 分析包")
        report_bytes = generate_excel_report(cleaned_df, quality_report, analysis)
        st.download_button(
            "下载智能数据分析报告.xlsx",
            data=report_bytes,
            file_name="智能数据分析报告.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
