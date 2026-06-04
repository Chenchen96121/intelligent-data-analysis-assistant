from io import BytesIO

import pandas as pd

from analyzer import AnalysisSummary
from data_processor import QualityReport


def generate_excel_report(
    cleaned_df: pd.DataFrame,
    quality_report: QualityReport,
    analysis: AnalysisSummary,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        cleaned_df.to_excel(writer, sheet_name="清洗后数据", index=False)
        _write_quality_sheet(writer, cleaned_df, quality_report)
        analysis.descriptive_stats.to_excel(writer, sheet_name="描述统计", index=False)
        quality_report.missing_summary.to_excel(writer, sheet_name="缺失值统计", index=False)
        quality_report.outlier_summary.to_excel(writer, sheet_name="异常值统计", index=False)
        analysis.chart_data.to_excel(writer, sheet_name="图表数据", index=False)

        if not analysis.category_summaries.empty:
            analysis.category_summaries.to_excel(writer, sheet_name="类别频数", index=False)
        if not analysis.time_summaries.empty:
            analysis.time_summaries.to_excel(writer, sheet_name="时间趋势摘要", index=False)

        _format_workbook(writer)

    return output.getvalue()


def _write_quality_sheet(
    writer: pd.ExcelWriter,
    cleaned_df: pd.DataFrame,
    quality_report: QualityReport,
) -> None:
    summary = pd.DataFrame(
        [
            {"指标": "清洗后行数", "值": len(cleaned_df)},
            {"指标": "字段数量", "值": cleaned_df.shape[1]},
            {"指标": "重复行数量", "值": quality_report.duplicate_rows},
            {"指标": "存在缺失值字段数", "值": int((quality_report.missing_summary["缺失值数量"] > 0).sum())},
            {
                "指标": "存在异常值字段数",
                "值": int((quality_report.outlier_summary["异常值数量"] > 0).sum())
                if not quality_report.outlier_summary.empty
                else 0,
            },
        ]
    )
    summary.to_excel(writer, sheet_name="数据质量报告", index=False)


def _format_workbook(writer: pd.ExcelWriter) -> None:
    workbook = writer.book
    body_format = workbook.add_format({"border": 1})

    for worksheet in writer.sheets.values():
        worksheet.freeze_panes(1, 0)
        worksheet.set_column(0, 20, 18, body_format)
        worksheet.set_row(
            0,
            None,
            workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "align": "center"}),
        )
