# 智能数据分析助手

一个 Streamlit 数据分析项目，支持 Excel / CSV 上传、数据清洗、统计分析、图表推荐、规则式自然语言问答，以及 Excel 分析包导出。

## 功能

- 支持上传 `.xlsx`、`.xls`、`.csv` 文件
- 自动读取 Excel 第一个工作表，CSV 支持常见 UTF-8 / GBK 中文编码
- 自动识别数值列、类别列和日期列
- 检测缺失值、重复行和 IQR 异常值
- 生成描述统计、类别频数和时间趋势摘要
- 自动推荐直方图、箱线图、柱状图、折线图和散点图
- 使用规则式 Mock 实现自然语言数据问答
- 支持 OpenAI 兼容 API 大模型问答测试，页面内提供小白接入教程
- 导出 `智能数据分析报告.xlsx`

## 运行

```bash
pip install -r requirements.txt
python sample_data.py
streamlit run app.py
```

打开页面后上传 `sample_sales_data.xlsx` 或 `sample_sales_data.csv` 即可演示。

## 项目结构

- `app.py`：Streamlit 页面入口
- `data_processor.py`：Excel / CSV 读取、数据清洗、字段识别、数据质量检测
- `analyzer.py`：统计分析与图表推荐
- `qa_engine.py`：规则式自然语言问答
- `llm_client.py`：OpenAI 兼容 API 大模型问答调用
- `report_generator.py`：Excel 分析包生成
- `sample_data.py`：生成演示用 Excel / CSV 数据
- `tests/test_core.py`：核心功能测试

## 项目总结

智能数据分析助手：基于 Python、Pandas 和 Streamlit 开发数据分析平台，实现 Excel / CSV 文件上传、自动清洗、异常值与缺失值识别、统计分析、图表推荐和 Excel 报告导出；设计规则式自然语言问答模块，并扩展 OpenAI 兼容 API 大模型问答测试能力，支持用户通过中文问题获取数据概览、缺失值、异常值和指标统计结果。
