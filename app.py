import streamlit as st

from analyzer import build_analysis_summary, create_plotly_charts
from data_processor import clean_data, detect_column_types, read_excel_file
from qa_engine import RuleBasedQAEngine
from report_generator import generate_excel_report


st.set_page_config(
    page_title="智能数据分析助手",
    page_icon="📊",
    layout="wide",
)


def render_empty_state() -> None:
    st.title("智能数据分析助手")
    st.caption("上传 Excel 文件后，自动完成数据清洗、统计分析、图表生成与问答。")
    st.info("请选择一个 .xlsx 或 .xls 文件开始分析。")


def main() -> None:
    st.title("智能数据分析助手")
    st.caption("Excel 自动清洗 · 统计分析 · 可视化报告 · 自然语言问答")

    uploaded_file = st.sidebar.file_uploader(
        "上传 Excel 文件",
        type=["xlsx", "xls"],
        help="默认读取第一个工作表，并使用首行作为字段名。",
    )

    if uploaded_file is None:
        render_empty_state()
        return

    try:
        raw_df = read_excel_file(uploaded_file)
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
        question = st.text_input(
            "输入你的问题",
            placeholder="例如：缺失值最多的列是什么？销售额平均值是多少？生成数据总结",
        )
        if question:
            st.write(qa_engine.answer(question))

        st.caption("当前版本使用规则式 Mock 问答，后续可替换为真实 LLM API。")

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
