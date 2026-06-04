from dataclasses import dataclass

import pandas as pd
import plotly.express as px

from data_processor import ColumnTypes, QualityReport


@dataclass(frozen=True)
class AnalysisSummary:
    descriptive_stats: pd.DataFrame
    category_summaries: pd.DataFrame
    time_summaries: pd.DataFrame
    chart_data: pd.DataFrame


def build_analysis_summary(
    df: pd.DataFrame,
    quality_report: QualityReport,
    column_types: ColumnTypes,
) -> AnalysisSummary:
    descriptive_stats = _build_descriptive_stats(df, column_types.numeric)
    category_summaries = _build_category_summaries(df, column_types.categorical)
    time_summaries = _build_time_summaries(df, column_types.datetime, column_types.numeric)
    chart_data = _build_chart_data(df, quality_report, column_types)

    return AnalysisSummary(
        descriptive_stats=descriptive_stats,
        category_summaries=category_summaries,
        time_summaries=time_summaries,
        chart_data=chart_data,
    )


def create_plotly_charts(df: pd.DataFrame, column_types: ColumnTypes) -> list[tuple[str, object]]:
    charts: list[tuple[str, object]] = []

    for column in column_types.numeric[:2]:
        charts.append((f"{column} 分布直方图", px.histogram(df, x=column, title=f"{column} 分布直方图")))
        charts.append((f"{column} 箱线图", px.box(df, y=column, title=f"{column} 箱线图")))

    for column in column_types.categorical[:2]:
        counts = df[column].value_counts().head(10).reset_index()
        counts.columns = [column, "数量"]
        charts.append((f"{column} 类别分布", px.bar(counts, x=column, y="数量", title=f"{column} 类别分布 Top 10")))

    if column_types.datetime and column_types.numeric:
        date_column = column_types.datetime[0]
        numeric_column = column_types.numeric[0]
        trend = (
            df[[date_column, numeric_column]]
            .dropna()
            .set_index(date_column)
            .resample("D")[numeric_column]
            .mean()
            .reset_index()
        )
        if not trend.empty:
            charts.append((f"{numeric_column} 时间趋势", px.line(trend, x=date_column, y=numeric_column, title=f"{numeric_column} 时间趋势")))

    if len(column_types.numeric) >= 2:
        x_column, y_column = column_types.numeric[:2]
        charts.append((f"{x_column} 与 {y_column} 关系", px.scatter(df, x=x_column, y=y_column, title=f"{x_column} 与 {y_column} 关系")))

    return charts


def _build_descriptive_stats(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if not numeric_columns:
        return pd.DataFrame(columns=["字段", "均值", "中位数", "最大值", "最小值", "标准差"])

    stats = df[numeric_columns].agg(["mean", "median", "max", "min", "std"]).T.reset_index()
    stats.columns = ["字段", "均值", "中位数", "最大值", "最小值", "标准差"]
    numeric_stat_columns = ["均值", "中位数", "最大值", "最小值", "标准差"]
    stats[numeric_stat_columns] = stats[numeric_stat_columns].round(2)
    return stats


def _build_category_summaries(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in categorical_columns:
        for value, count in df[column].value_counts(dropna=False).head(10).items():
            rows.append({"字段": column, "类别": value, "数量": int(count)})
    return pd.DataFrame(rows, columns=["字段", "类别", "数量"])


def _build_time_summaries(
    df: pd.DataFrame,
    datetime_columns: list[str],
    numeric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_column in datetime_columns:
        date_series = df[date_column].dropna()
        if date_series.empty:
            continue

        row: dict[str, object] = {
            "日期字段": date_column,
            "最早日期": date_series.min(),
            "最晚日期": date_series.max(),
            "记录数量": int(date_series.count()),
        }
        if numeric_columns:
            numeric_column = numeric_columns[0]
            row["关联数值字段"] = numeric_column
            row["日均值"] = round(float(df[numeric_column].mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_chart_data(
    df: pd.DataFrame,
    quality_report: QualityReport,
    column_types: ColumnTypes,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column in column_types.numeric:
        rows.append({"字段": column, "推荐图表": "直方图 / 箱线图", "原因": "数值字段适合查看分布与异常值"})

    for column in column_types.categorical:
        rows.append({"字段": column, "推荐图表": "柱状图", "原因": "类别字段适合查看频数分布"})

    for column in column_types.datetime:
        rows.append({"字段": column, "推荐图表": "折线图", "原因": "日期字段适合查看时间趋势"})

    if len(column_types.numeric) >= 2:
        rows.append(
            {
                "字段": f"{column_types.numeric[0]} + {column_types.numeric[1]}",
                "推荐图表": "散点图",
                "原因": "两个数值字段适合查看相关关系",
            }
        )

    if not quality_report.outlier_summary.empty:
        top_outlier = quality_report.outlier_summary.iloc[0]
        rows.append(
            {
                "字段": top_outlier["字段"],
                "推荐图表": "箱线图",
                "原因": f"该字段异常值数量最高：{top_outlier['异常值数量']}",
            }
        )

    return pd.DataFrame(rows, columns=["字段", "推荐图表", "原因"])
