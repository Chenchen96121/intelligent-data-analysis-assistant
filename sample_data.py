from pathlib import Path

import pandas as pd


def create_sample_data() -> None:
    data = pd.DataFrame(
        {
            "日期": pd.date_range("2026-01-01", periods=16, freq="D").tolist(),
            "地区": ["华东", "华南", "华北", "华东", "西南", "华南", "华东", "华北", "西北", "华南", "华东", "华北", None, "华东", "华南", "华南"],
            "销售额": [1200, 1350, 980, 1420, None, 1600, 1550, 1490, 5000, 1320, 1280, 1400, 1375, 1450, 1510, 1510],
            "订单数": [12, 15, 9, 16, 11, 18, 17, None, 14, 13, 12, 15, 16, 17, 18, 18],
            "客户类型": ["新客户", "老客户", "老客户", "新客户", "新客户", "老客户", None, "老客户", "新客户", "老客户", "新客户", "老客户", "新客户", "老客户", "老客户", "老客户"],
        }
    )

    data = pd.concat([data, data.iloc[[15]]], ignore_index=True)
    output_path = Path("sample_sales_data.xlsx")
    data.to_excel(output_path, index=False)
    csv_output_path = Path("sample_sales_data.csv")
    data.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"已生成示例数据：{output_path.resolve()}")
    print(f"已生成 CSV 示例数据：{csv_output_path.resolve()}")


if __name__ == "__main__":
    create_sample_data()
