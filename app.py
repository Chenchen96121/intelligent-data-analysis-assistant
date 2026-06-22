import html

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


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f3f6f8;
            --panel-bg: #ffffff;
            --panel-border: #d9e2ec;
            --text-main: #172033;
            --text-muted: #64748b;
            --teal: #0f766e;
            --blue: #2563eb;
            --amber: #b45309;
            --rose: #be123c;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--text-main);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--panel-border);
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
            padding: 0.85rem;
            border: 1px dashed #94a3b8;
            border-radius: 8px;
            background: #f8fafc;
        }

        .app-header {
            padding: 1.15rem 1.25rem;
            border: 1px solid var(--panel-border);
            border-left: 5px solid var(--teal);
            border-radius: 8px;
            background: var(--panel-bg);
            margin-bottom: 1rem;
        }

        .app-kicker {
            margin: 0 0 0.35rem 0;
            color: var(--teal);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .app-title {
            margin: 0;
            color: var(--text-main);
            font-size: 1.85rem;
            line-height: 1.22;
            font-weight: 760;
            letter-spacing: 0;
        }

        .app-subtitle {
            margin: 0.55rem 0 0 0;
            color: var(--text-muted);
            font-size: 0.98rem;
            line-height: 1.65;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.9rem 0 1rem 0;
        }

        .workflow-step {
            padding: 0.72rem 0.78rem;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: #ffffff;
            min-height: 82px;
        }

        .workflow-step strong {
            display: block;
            color: var(--text-main);
            font-size: 0.88rem;
            margin-bottom: 0.25rem;
        }

        .workflow-step span {
            color: var(--text-muted);
            font-size: 0.78rem;
            line-height: 1.45;
        }

        .insight-strip {
            display: grid;
            grid-template-columns: 1.2fr 1fr 1fr;
            gap: 0.75rem;
            margin: 0.9rem 0 1rem 0;
        }

        .insight-card {
            padding: 0.9rem 1rem;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--panel-bg);
            min-height: 104px;
        }

        .insight-card h3 {
            margin: 0 0 0.45rem 0;
            color: var(--text-main);
            font-size: 0.98rem;
        }

        .insight-card p {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.86rem;
            line-height: 1.6;
        }

        .qa-answer {
            padding: 0.9rem 1rem;
            border: 1px solid #bfdbfe;
            border-left: 4px solid var(--blue);
            border-radius: 8px;
            background: #eff6ff;
            color: #1e3a8a;
            line-height: 1.7;
        }

        .report-panel {
            padding: 1rem;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--panel-bg);
        }

        .report-panel p {
            color: var(--text-muted);
            line-height: 1.65;
            margin-top: 0;
        }

        .metric-card {
            min-height: 118px;
            padding: 1rem;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--panel-bg);
        }

        .metric-label {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.82rem;
            font-weight: 650;
        }

        .metric-value {
            margin: 0.45rem 0 0.2rem 0;
            color: var(--text-main);
            font-size: 1.72rem;
            line-height: 1.15;
            font-weight: 780;
        }

        .metric-note {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.78rem;
        }

        .accent-teal { border-top: 4px solid var(--teal); }
        .accent-blue { border-top: 4px solid var(--blue); }
        .accent-amber { border-top: 4px solid var(--amber); }
        .accent-rose { border-top: 4px solid var(--rose); }

        .empty-panel {
            padding: 1.05rem 1.15rem;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--panel-bg);
            margin-top: 1rem;
        }

        .empty-panel h3 {
            margin: 0 0 0.65rem 0;
            font-size: 1rem;
            color: var(--text-main);
        }

        .empty-panel p,
        .empty-panel li {
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.65;
        }

        .field-card {
            padding: 0.9rem;
            min-height: 136px;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--panel-bg);
        }

        .field-card h4 {
            margin: 0 0 0.55rem 0;
            color: var(--text-main);
            font-size: 0.95rem;
        }

        .field-list {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.88rem;
            line-height: 1.7;
            word-break: break-word;
        }

        .sidebar-status {
            padding: 0.75rem 0.85rem;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background: #f8fafc;
            color: #334155;
            font-size: 0.86rem;
            line-height: 1.55;
            margin: 0.3rem 0 0.75rem 0;
        }

        div[data-testid="stTabs"] button {
            border-radius: 6px;
            color: #334155;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--teal);
            font-weight: 700;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px;
            border: 1px solid #0f766e;
            background: #0f766e;
            color: white;
            font-weight: 700;
        }

        .stButton > button:disabled {
            border-color: #cbd5e1;
            background: #e2e8f0;
            color: #64748b;
        }

        @media (max-width: 900px) {
            .workflow-grid,
            .insight-strip {
                grid-template-columns: 1fr;
            }

            .app-title {
                font-size: 1.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <p class="app-kicker">DATA ANALYSIS ASSISTANT</p>
            <h1 class="app-title">智能数据分析助手</h1>
            <p class="app-subtitle">
                面向 Excel / CSV 的自动清洗、数据质量检测、统计分析、可视化报告与自然语言问答工具。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-panel">
            <h3>开始分析前</h3>
            <p>你可以先在左侧上传 Excel / CSV 文件；如果要测试真实大模型问答，也可以先完成 API 接入测试。</p>
            <ul>
                <li>支持 .xlsx、.xls、.csv 文件</li>
                <li>自动识别缺失值、重复行、异常值和字段类型</li>
                <li>生成图表建议、统计结果和可下载 Excel 分析包</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_steps() -> None:
    st.markdown(
        """
        <div class="workflow-grid">
            <div class="workflow-step"><strong>上传数据</strong><span>支持 Excel / CSV，多种中文编码。</span></div>
            <div class="workflow-step"><strong>自动清洗</strong><span>识别缺失值、重复行和异常值。</span></div>
            <div class="workflow-step"><strong>统计分析</strong><span>生成描述统计和类别频数。</span></div>
            <div class="workflow-step"><strong>智能问答</strong><span>规则模式或 API 大模型模式。</span></div>
            <div class="workflow-step"><strong>导出报告</strong><span>下载多 Sheet Excel 分析包。</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_llm_setup() -> LLMConfig | None:
    with st.sidebar.expander("API 大模型接入", expanded=True):
        st.caption("可先测试 AI 模型连接，再上传数据文件。API Key 只在当前页面会话中使用。")
        st.caption("API 问答使用精简 Prompt：先提取结构化查询意图，再发送压缩数据摘要，以降低调用成本。")
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


def render_sidebar_file_status(uploaded_file) -> None:
    if uploaded_file is None:
        message = "文件状态：等待上传。支持 Excel / CSV，建议首行为字段名。"
    else:
        file_name = html.escape(uploaded_file.name)
        message = f"文件状态：已选择 {file_name}，正在准备分析。"
    st.sidebar.markdown(f'<div class="sidebar-status">{message}</div>', unsafe_allow_html=True)


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


def render_metric_card(label: str, value: str, note: str, accent: str) -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_note = html.escape(note)
    st.markdown(
        f"""
        <div class="metric-card accent-{accent}">
            <p class="metric-label">{safe_label}</p>
            <p class="metric-value">{safe_value}</p>
            <p class="metric-note">{safe_note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_field_card(title: str, values: list[str], accent: str) -> None:
    field_text = "、".join(html.escape(value) for value in values) if values else "暂无"
    safe_title = html.escape(title)
    st.markdown(
        f"""
        <div class="field-card accent-{accent}">
            <h4>{safe_title}</h4>
            <p class="field-list">{field_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_strip(cleaned_df, quality_report, column_types) -> None:
    missing_fields = int((quality_report.missing_summary["缺失值数量"] > 0).sum())
    outlier_fields = (
        int((quality_report.outlier_summary["异常值数量"] > 0).sum())
        if not quality_report.outlier_summary.empty
        else 0
    )
    field_mix = (
        f"数值字段 {len(column_types.numeric)} 个，"
        f"类别字段 {len(column_types.categorical)} 个，"
        f"日期字段 {len(column_types.datetime)} 个。"
    )
    quality_text = (
        f"检测到 {missing_fields} 个字段存在缺失值，"
        f"{outlier_fields} 个数值字段存在异常值，"
        f"重复行 {quality_report.duplicate_rows} 行。"
    )
    report_text = f"当前清洗后数据为 {len(cleaned_df):,} 行，可直接进入问答或导出报告。"

    st.markdown(
        f"""
        <div class="insight-strip">
            <div class="insight-card">
                <h3>数据结构</h3>
                <p>{html.escape(field_mix)}</p>
            </div>
            <div class="insight-card">
                <h3>质量提示</h3>
                <p>{html.escape(quality_text)}</p>
            </div>
            <div class="insight-card">
                <h3>下一步</h3>
                <p>{html.escape(report_text)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_box(answer: str) -> None:
    st.markdown(
        f'<div class="qa-answer">{html.escape(answer)}</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_css()
    render_app_header()
    render_workflow_steps()

    uploaded_file = st.sidebar.file_uploader(
        "上传 Excel / CSV 文件",
        type=["xlsx", "xls", "csv"],
        help="Excel 默认读取第一个工作表；CSV 支持常见 UTF-8 / GBK 中文编码。",
    )
    render_sidebar_file_status(uploaded_file)
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
    with overview_cols[0]:
        render_metric_card("原始行数", f"{len(raw_df):,}", "上传文件中的总记录数", "teal")
    with overview_cols[1]:
        render_metric_card("清洗后行数", f"{len(cleaned_df):,}", "去重与填充后的有效记录", "blue")
    with overview_cols[2]:
        render_metric_card("字段数量", f"{cleaned_df.shape[1]:,}", "当前可分析的数据字段", "amber")
    with overview_cols[3]:
        render_metric_card("重复行", f"{quality_report.duplicate_rows:,}", "自动检测到的重复记录", "rose")

    render_insight_strip(cleaned_df, quality_report, column_types)

    tab_preview, tab_quality, tab_analysis, tab_charts, tab_qa, tab_report = st.tabs(
        ["数据预览", "数据质量", "统计分析", "可视化图表", "自然语言问答", "报告导出"]
    )

    with tab_preview:
        st.subheader("清洗后数据")
        st.dataframe(cleaned_df, hide_index=True)

        st.subheader("字段识别")
        type_cols = st.columns(3)
        with type_cols[0]:
            render_field_card("数值列", column_types.numeric, "teal")
        with type_cols[1]:
            render_field_card("类别列", column_types.categorical, "blue")
        with type_cols[2]:
            render_field_card("日期列", column_types.datetime, "amber")

    with tab_quality:
        quality_cols = st.columns(2)
        with quality_cols[0]:
            with st.container(border=True):
                st.subheader("缺失值统计")
                st.dataframe(quality_report.missing_summary, hide_index=True)
        with quality_cols[1]:
            with st.container(border=True):
                st.subheader("异常值统计")
                st.dataframe(quality_report.outlier_summary, hide_index=True)

    with tab_analysis:
        with st.container(border=True):
            st.subheader("描述统计")
            st.dataframe(analysis.descriptive_stats, hide_index=True)

        if not analysis.category_summaries.empty:
            with st.container(border=True):
                st.subheader("类别字段频数 Top 10")
                st.dataframe(analysis.category_summaries, hide_index=True)

        if not analysis.time_summaries.empty:
            with st.container(border=True):
                st.subheader("日期字段趋势摘要")
                st.dataframe(analysis.time_summaries, hide_index=True)

    with tab_charts:
        st.subheader("自动推荐图表")
        charts = create_plotly_charts(cleaned_df, column_types)
        if not charts:
            st.info("当前数据字段不足，暂未生成推荐图表。")
        for title, chart in charts:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.plotly_chart(chart, width="stretch")

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
                render_answer_box(qa_engine.answer(question))
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
                            render_answer_box(
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
        st.markdown(
            """
            <div class="report-panel">
                <p>报告会把清洗后数据、数据质量、描述统计、缺失值、异常值和图表推荐数据整理到一个 Excel 文件中，适合保存、展示或继续二次分析。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        report_bytes = generate_excel_report(cleaned_df, quality_report, analysis)
        st.download_button(
            "下载智能数据分析报告.xlsx",
            data=report_bytes,
            file_name="智能数据分析报告.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
