# News-Driven Market Trend Index (S&P500)

把全球宏观新闻信息流压缩为每日 `TrendScore`（`-100` 到 `+100`）。

## 功能
- RSS 抓取 + 网页正文抽取（`trafilatura`）
- 标题/内容去重
- 规则化主题识别、方向判断、相关性评估、影响强度计算
- 聚合为日度 `TrendScore`
- 生成 Markdown 日报

## 目录
- `config/feeds.yaml`：RSS 源
- `config/scoring.yaml`：主题权重、词典、规则参数
- `src/news_trends/`：核心代码
- `reports/`：日报输出
- `data/`：中间数据（raw / processed）
- `scripts/run_daily.py`：运行入口

## 安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行
```bash
python scripts/run_daily.py --date 2026-02-18
```

不传 `--date` 默认使用今天（本地时区）。

## Docker 运行
1. 构建镜像：
```bash
docker build -t news-trends:latest .
```

2. 直接运行（当天）：
```bash
docker run --rm \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  news-trends:latest
```

3. 指定日期运行：
```bash
docker run --rm \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  news-trends:latest \
  python scripts/run_daily.py --date 2026-02-18
```

4. 使用 docker compose：
```bash
docker compose run --rm news_trends
docker compose run --rm news_trends python scripts/run_daily.py --date 2026-02-18
```

## Dashboard（网页版）
先至少生成一天数据：
```bash
python scripts/run_daily.py --date 2026-02-18
```

本地启动 Dashboard：
```bash
streamlit run scripts/dashboard.py
```

Docker 启动 Dashboard：
```bash
docker compose up dashboard
```
浏览器打开：`http://localhost:8501`
在页面左侧可以通过按钮直接触发“跑今天”或“跑指定日”数据，无需再单独执行脚本。

## 输出
运行后将生成：
- `reports/YYYY-MM-DD.md`
- `data/processed/YYYY-MM-DD_scored.json`

## 说明
- 该系统是宏观环境指标，不直接输出交易信号。
- 全流程使用免费公开源（RSS + 网页抓取）。
