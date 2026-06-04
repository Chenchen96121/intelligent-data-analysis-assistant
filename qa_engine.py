import re

import pandas as pd

from analyzer import AnalysisSummary
from data_processor import ColumnTypes, QualityReport


class RuleBasedQAEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        quality_report: QualityReport,
        column_types: ColumnTypes,
        analysis: AnalysisSummary,
    ) -> None:
        self.df = df
        self.quality_report = quality_report
        self.column_types = column_types
        self.analysis = analysis
        self._question_text = ""

    def answer(self, question: str) -> str:
        normalized = question.strip().lower()
        self._question_text = normalized

        if not normalized:
            return "请输入一个数据分析问题。"

        if any(keyword in normalized for keyword in ["总结", "概览", "summary"]):
            return self._dataset_summary()

        if any(keyword in normalized for keyword in ["缺失", "missing"]):
            return self._missing_answer()

        if any(keyword in normalized for keyword in ["异常", "outlier"]):
            return self._outlier_answer()

        if any(keyword in normalized for keyword in ["平均", "均值", "mean", "average"]):
            return self._numeric_metric_answer("均值")

        if any(keyword in normalized for keyword in ["最大", "max"]):
            return self._numeric_metric_answer("最大值")

        if any(keyword in normalized for keyword in ["最小", "min"]):
            return self._numeric_metric_answer("最小值")

        if any(keyword in normalized for keyword in ["中位", "median"]):
            return self._numeric_metric_answer("中位数")

        if any(keyword in normalized for keyword in ["字段", "列", "columns"]):
            return self._column_answer()

        return "暂未识别该问题。你可以尝试询问：缺失值最多的列是什么、哪个字段异常值最多、某个数值字段平均值是多少，或生成数据总结。"

    def _dataset_summary(self) -> str:
        return (
            f"当前数据清洗后共有 {len(self.df)} 行、{self.df.shape[1]} 个字段；"
            f"其中数值字段 {len(self.column_types.numeric)} 个，类别字段 {len(self.column_types.categorical)} 个，"
            f"日期字段 {len(self.column_types.datetime)} 个。"
            f"检测到重复行 {self.quality_report.duplicate_rows} 行。"
        )

    def _missing_answer(self) -> str:
        if self.quality_report.missing_summary.empty:
            return "当前数据没有字段可用于缺失值统计。"
        top_row = self.quality_report.missing_summary.iloc[0]
        return f"缺失值最多的字段是「{top_row['字段']}」，缺失 {top_row['缺失值数量']} 个，缺失率为 {top_row['缺失率']}%。"

    def _outlier_answer(self) -> str:
        if self.quality_report.outlier_summary.empty:
            return "当前数据没有数值字段，因此没有异常值统计结果。"
        top_row = self.quality_report.outlier_summary.iloc[0]
        return f"异常值最多的字段是「{top_row['字段']}」，共检测到 {top_row['异常值数量']} 个异常值。"

    def _numeric_metric_answer(self, metric: str) -> str:
        if self.analysis.descriptive_stats.empty:
            return "当前数据没有数值字段，无法计算该指标。"

        column = self._find_column_in_question()
        stats = self.analysis.descriptive_stats
        if column is not None and column in stats["字段"].values:
            row = stats[stats["字段"] == column].iloc[0]
            return f"字段「{column}」的{metric}为 {row[metric]}。"

        first_row = stats.iloc[0]
        return f"未匹配到具体字段，默认返回「{first_row['字段']}」的{metric}：{first_row[metric]}。"

    def _column_answer(self) -> str:
        return (
            f"识别到数值字段：{_format_list(self.column_types.numeric)}；"
            f"类别字段：{_format_list(self.column_types.categorical)}；"
            f"日期字段：{_format_list(self.column_types.datetime)}。"
        )

    def _find_column_in_question(self) -> str | None:
        question_columns = sorted(self.df.columns, key=len, reverse=True)
        for column in question_columns:
            if re.search(re.escape(column.lower()), self._question_text):
                return column
        return None


def _format_list(values: list[str]) -> str:
    return "、".join(values) if values else "无"
