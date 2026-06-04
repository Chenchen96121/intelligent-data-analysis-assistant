from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd


@dataclass(frozen=True)
class ColumnTypes:
    numeric: list[str]
    categorical: list[str]
    datetime: list[str]


@dataclass(frozen=True)
class QualityReport:
    duplicate_rows: int
    missing_summary: pd.DataFrame
    outlier_summary: pd.DataFrame
    outlier_masks: dict[str, pd.Series]


def read_excel_file(file: str | BinaryIO) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=0)
    df = _deduplicate_column_names(df)
    return df


def detect_column_types(df: pd.DataFrame) -> ColumnTypes:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical = [
        column
        for column in df.columns
        if column not in numeric and column not in datetime
    ]
    return ColumnTypes(numeric=numeric, categorical=categorical, datetime=datetime)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    if df.empty:
        return df.copy(), QualityReport(
            duplicate_rows=0,
            missing_summary=_build_missing_summary(df),
            outlier_summary=pd.DataFrame(columns=["字段", "异常值数量", "异常值比例"]),
            outlier_masks={},
        )

    cleaned = _normalize_dataframe(df)
    duplicate_rows = int(cleaned.duplicated().sum())
    missing_summary = _build_missing_summary(cleaned)

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = _fill_missing_values(cleaned)
    outlier_masks, outlier_summary = _detect_outliers(cleaned)

    return cleaned, QualityReport(
        duplicate_rows=duplicate_rows,
        missing_summary=missing_summary,
        outlier_summary=outlier_summary,
        outlier_masks=outlier_masks,
    )


def _deduplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    new_columns: list[str] = []
    for raw_column in df.columns:
        column = str(raw_column).strip() or "未命名字段"
        count = seen.get(column, 0)
        new_columns.append(column if count == 0 else f"{column}_{count + 1}")
        seen[column] = count + 1
    df = df.copy()
    df.columns = new_columns
    return df


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _deduplicate_column_names(df.copy())

    for column in normalized.columns:
        if normalized[column].dtype == "object":
            stripped = normalized[column].astype("string").str.strip()
            normalized[column] = stripped.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            converted_date = pd.to_datetime(normalized[column], errors="coerce")
            if converted_date.notna().mean() >= 0.8:
                normalized[column] = converted_date

    return normalized


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    for column in filled.columns:
        series = filled[column]
        if pd.api.types.is_numeric_dtype(series):
            fill_value = series.median()
            filled[column] = series.fillna(fill_value if pd.notna(fill_value) else 0)
        elif pd.api.types.is_datetime64_any_dtype(series):
            mode = series.mode(dropna=True)
            filled[column] = series.fillna(mode.iloc[0] if not mode.empty else pd.Timestamp("1970-01-01"))
        else:
            mode = series.mode(dropna=True)
            filled[column] = series.fillna(mode.iloc[0] if not mode.empty else "未知")
    return filled


def _build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    missing_counts = df.isna().sum()
    if total_rows == 0:
        missing_rates = missing_counts.astype(float)
    else:
        missing_rates = (missing_counts / total_rows * 100).round(2)

    return pd.DataFrame(
        {
            "字段": missing_counts.index,
            "缺失值数量": missing_counts.values,
            "缺失率": missing_rates.values,
        }
    ).sort_values("缺失值数量", ascending=False, ignore_index=True)


def _detect_outliers(df: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    masks: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            mask = pd.Series(False, index=df.index)
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                mask = pd.Series(False, index=df.index)
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (df[column] < lower) | (df[column] > upper)

        masks[column] = mask.fillna(False)
        count = int(masks[column].sum())
        rows.append(
            {
                "字段": column,
                "异常值数量": count,
                "异常值比例": round(count / len(df) * 100, 2) if len(df) else 0,
            }
        )

    return masks, pd.DataFrame(rows).sort_values("异常值数量", ascending=False, ignore_index=True)
