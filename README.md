# News-Driven Market Trend Index (S&P500)

把全球宏观新闻信息流压缩为每日 `TrendScore`（`-100` 到 `+100`），用于观察美股大盘风险偏好变化。

## 功能概览
- RSS 抓取新闻元数据
- 网页正文抽取（`trafilatura`）
- 新闻去重（标题/内容相似）
- 规则化主题识别、方向判断、相关性评估、影响强度计算
- 聚合为日度 `TrendScore`
- 输出 JSON 结构化结果与 Markdown 日报
- 提供 Streamlit Dashboard 可视化与一键运行

## 项目结构
- `config/feeds.yaml`: RSS 源配置
- `config/scoring.yaml`: 主题权重、关键词、区间阈值与模型参数
- `src/news_trends/`: 核心流程代码
- `scripts/run_daily.py`: CLI 运行入口
- `scripts/dashboard.py`: Streamlit 仪表盘
- `data/raw/`: 原始统计与中间文件
- `data/processed/`: 每日评分 JSON 输出
- `reports/`: 每日 Markdown 日报
- `tests/`: 单元测试

## 快速开始
### 1) 安装依赖
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 运行一天数据
```bash
python scripts/run_daily.py --date 2026-02-18
```

不传 `--date` 时，默认使用所选时区当天日期（CLI 默认运行机器本地时区，可用 `--timezone` 指定）。

### 3) 启动 Dashboard
```bash
streamlit run scripts/dashboard.py
```
浏览器打开 `http://localhost:8501`。

## CLI 用法
```bash
python scripts/run_daily.py [--date YYYY-MM-DD] [--timezone IANA_TZ]
```

返回值会打印为 JSON，核心字段包括：
- `date`: 目标日期
- `raw_articles`: RSS 原始文章数
- `date_filtered_articles`: 日期筛选后文章数
- `deduped_articles`: 去重后文章数
- `trend_score`: 当日总分
- `report_path`: Markdown 报告路径
- `json_path`: JSON 结果路径

## 输出说明
每次运行会生成以下文件：
- `reports/YYYY-MM-DD.md`
- `data/processed/YYYY-MM-DD_scored.json`
- `data/raw/YYYY-MM-DD_count.txt`（流程计数统计）

`data/processed/YYYY-MM-DD_scored.json` 包含：
- `trend_score`、`market_regime`、`summary`
- `articles[]` 明细（主题、方向、相关性、影响、单条分数、链接等）

## Docker
### 构建镜像
```bash
docker build -t news-trends:latest .
```

### 直接运行（当天）
```bash
docker run --rm \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  news-trends:latest
```

### 指定日期运行
```bash
docker run --rm \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  news-trends:latest \
  python scripts/run_daily.py --date 2026-02-18
```

### Docker Compose
```bash
docker compose run --rm news_trends
docker compose run --rm news_trends python scripts/run_daily.py --date 2026-02-18
docker compose up dashboard
```

## 配置说明
### `config/feeds.yaml`
维护 RSS 源列表（名称、分类、URL）。

### `config/scoring.yaml`
维护打分规则，例如：
- 主题权重与关键词
- 方向判定规则
- `rules.regime_thresholds`（市场区间阈值，必填）
- `sentiment.server_url` / `sentiment.model`（Dashboard 模型连通性检查使用）

## Dashboard 说明
- 左侧顶部提供时区下拉框，运行日期默认显示该时区当天
- 左侧可直接触发“跑今天”或“跑指定日”
- 展示历史 `TrendScore` 折线、主题分布、方向分布
- 展示 `|article_score|` 排序后的 Top News
- 提供“检查LLM模型”按钮，验证 `scoring.yaml` 中模型接口连通性

## 测试
```bash
pytest -q
```

## 注意事项
- 本项目输出的是宏观环境指标，不是直接交易信号。
- 日期过滤与“跑今天”均基于所选时区（Dashboard 由侧边栏选择；CLI 可用 `--timezone` 指定）。
- 数据来源为公开 RSS 与网页正文抽取，稳定性受源站质量影响。
