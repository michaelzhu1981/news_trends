"""Streamlit 仪表盘：运行日报流程、查看 TrendScore 历史与 Top News。"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from time import perf_counter, sleep
from zoneinfo import ZoneInfo, available_timezones

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# 项目根目录与源码路径
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_trends.config import PROCESSED_DIR, load_scoring_config  # noqa: E402
from news_trends.pipeline import run_daily_pipeline  # noqa: E402


def _load_json_reports() -> list[dict]:
    """加载 data/processed 下所有 *_scored.json，按日期排序。"""
    files = sorted(PROCESSED_DIR.glob("*_scored.json"))
    rows: list[dict] = []
    for file in files:
        try:
            rows.append(json.loads(file.read_text(encoding="utf-8")))
        except Exception:
            continue
    rows.sort(key=lambda x: x.get("date", ""))
    return rows


def _history_frame(data: list[dict]) -> pd.DataFrame:
    """将 JSON 报告列表转为历史趋势 DataFrame。"""
    if not data:
        return pd.DataFrame(columns=["date", "trend_score", "article_count", "market_regime", "summary"])
    return pd.DataFrame(
        [
            {
                "date": d.get("date"),
                "trend_score": d.get("trend_score", 0.0),
                "article_count": d.get("article_count", 0),
                "market_regime": d.get("market_regime", "中性"),
                "summary": d.get("summary", ""),
            }
            for d in data
        ]
    )


def _selected_day_articles(day_payload: dict) -> pd.DataFrame:
    """从单日报告 payload 中提取 articles 转为 DataFrame。"""
    articles = day_payload.get("articles", [])
    if not articles:
        return pd.DataFrame(
            columns=[
                "source",
                "title",
                "topic",
                "direction_label",
                "impact",
                "relevance",
                "article_score",
                "url",
            ]
        )
    return pd.DataFrame(articles)


def _regime_color(score: float) -> str:
    """根据 TrendScore 返回 regime 对应的颜色十六进制。"""
    if score >= 40:
        return "#117a65"
    if score >= 10:
        return "#1f8f4e"
    if score > -10:
        return "#9a7d0a"
    if score > -40:
        return "#af601a"
    return "#922b21"


def _regime_threshold_lines() -> list[float]:
    """读取 scoring.yaml 中 rules.regime_thresholds，并返回可用于绘图的阈值线。"""
    cfg = load_scoring_config()
    rules = cfg.get("rules", {})
    thresholds = rules.get("regime_thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("scoring.yaml 中需定义 rules.regime_thresholds")

    lines: list[float] = []
    for item in thresholds:
        if not isinstance(item, dict):
            continue
        min_score = item.get("min_score")
        if isinstance(min_score, (int, float)):
            lines.append(float(min_score))

    if not lines:
        raise ValueError("rules.regime_thresholds 里至少需要一个 min_score")
    return lines


def _safe_ratio(processed: int, total: int) -> float:
    """安全计算进度比例，返回 [0, 1] 区间。"""
    if total <= 0:
        return 0.0
    return min(1.0, max(0.0, processed / total))


def _format_elapsed(seconds: float) -> str:
    """将秒数格式化为 mm:ss / hh:mm:ss。"""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _timezone_options() -> list[str]:
    """返回可选时区列表。"""
    try:
        options = sorted(available_timezones())
    except Exception:
        options = []
    if not options:
        options = ["UTC", "Asia/Shanghai", "America/New_York"]
    return options


def _default_timezone_name(options: list[str]) -> str:
    """优先使用 NEWS_TRENDS_TZ，否则使用系统本地时区。"""
    tz_name = os.environ.get("NEWS_TRENDS_TZ", "").strip()
    if tz_name in options:
        return tz_name
    local_tz = datetime.now().astimezone().tzinfo
    local_name = getattr(local_tz, "key", "")
    if local_name in options:
        return local_name
    if "Asia/Shanghai" in options:
        return "Asia/Shanghai"
    return options[0]


def _timezone_label(tz_name: str) -> str:
    """展示时区标签：IANA 名称 + UTC 偏移。"""
    try:
        tz = ZoneInfo(tz_name)
        offset = datetime.now(tz).utcoffset()
        if offset is None:
            return f"{tz_name} (UTC+00:00)"
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        hours, minutes = divmod(abs(total_minutes), 60)
        return f"{tz_name} (UTC{sign}{hours:02d}:{minutes:02d})"
    except Exception:
        return tz_name


def _display_timezone(tz_name: str) -> ZoneInfo:
    """返回用于 UI 时间展示的时区。"""
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("UTC")


def _check_sentiment_model_status() -> tuple[bool, str]:
    """仅检查 LLM 模型接口连接是否正常。"""
    cfg = load_scoring_config()
    sentiment = cfg.get("sentiment", {})
    server = str(sentiment.get("server_url", "")).rstrip("/")
    model = str(sentiment.get("model", "")).strip()
    timeout = float(sentiment.get("timeout_seconds", 10))

    if not server:
        return False, "配置缺失：sentiment.server_url 为空"

    try:
        models_resp = requests.get(f"{server}/v1/models", timeout=timeout)
        models_resp.raise_for_status()
        model_text = model or "未配置"
        return True, f"模型接口连接正常（/v1/models），模型：{model_text}"
    except Exception as exc:
        model_text = model or "未配置"
        return False, f"模型接口连接失败（模型：{model_text}）：{exc}"


def main() -> None:
    """入口：配置页面、侧边栏运行控制、历史图表与 Top News 表格。"""
    st.set_page_config(page_title="News Trend Dashboard", page_icon="📈", layout="wide")
    tz_options = _timezone_options()
    if "selected_timezone" not in st.session_state:
        st.session_state["selected_timezone"] = _default_timezone_name(tz_options)
    if st.session_state["selected_timezone"] not in tz_options:
        tz_options.insert(0, st.session_state["selected_timezone"])
    selected_tz_name = st.sidebar.selectbox(
        "时区",
        options=tz_options,
        index=tz_options.index(st.session_state["selected_timezone"]),
        key="selected_timezone",
        format_func=_timezone_label,
    )
    display_tz = _display_timezone(selected_tz_name)
    today_in_selected_tz = datetime.now(display_tz).date()
    if st.session_state.get("run_date_timezone") != selected_tz_name:
        st.session_state["run_date"] = today_in_selected_tz
        st.session_state["run_date_timezone"] = selected_tz_name

    st.title("S&P500 新闻宏观趋势 Dashboard")
    st.caption("把全球信息流压缩成每日宏观情绪指数")

    st.sidebar.header("数据运行")
    run_date = st.sidebar.date_input("运行日期", format="YYYY-MM-DD", key="run_date")
    run_col1, run_col2 = st.sidebar.columns(2)
    run_today = run_col1.button("跑今天", use_container_width=True)
    run_selected = run_col2.button("跑指定日", use_container_width=True)

    run_status_box = st.container(border=True)
    running_placeholder = run_status_box.empty()
    progress_placeholder = run_status_box.empty()
    table_placeholder = run_status_box.empty()

    if "run_status_text" not in st.session_state:
        st.session_state["run_status_text"] = "**当前状态**: 未运行"
    if "run_progress_ratio" not in st.session_state:
        st.session_state["run_progress_ratio"] = 0.0
    if "run_progress_text" not in st.session_state:
        st.session_state["run_progress_text"] = "等待点击运行按钮"
    if "run_task_rows" not in st.session_state:
        st.session_state["run_task_rows"] = []
    if "run_in_progress" not in st.session_state:
        st.session_state["run_in_progress"] = False
    if "run_started_at" not in st.session_state:
        st.session_state["run_started_at"] = 0.0
    if "run_current_task" not in st.session_state:
        st.session_state["run_current_task"] = "未知任务"
    if "run_current_state_cn" not in st.session_state:
        st.session_state["run_current_state_cn"] = "进行中"
    if "run_current_processed" not in st.session_state:
        st.session_state["run_current_processed"] = 0
    if "run_current_total" not in st.session_state:
        st.session_state["run_current_total"] = 1
    if "run_current_message" not in st.session_state:
        st.session_state["run_current_message"] = "等待进度事件..."
    if "run_task_logs" not in st.session_state:
        st.session_state["run_task_logs"] = {}
    if "run_task_order" not in st.session_state:
        st.session_state["run_task_order"] = []
    if "run_event_queue" not in st.session_state:
        st.session_state["run_event_queue"] = None
    if "run_thread" not in st.session_state:
        st.session_state["run_thread"] = None
    if "run_last_result" not in st.session_state:
        st.session_state["run_last_result"] = None

    def _start_run(target_date: str | None) -> None:
        st.session_state["run_in_progress"] = True
        st.session_state["run_started_at"] = perf_counter()
        st.session_state["run_status_text"] = "任务已启动，等待进度事件..."
        st.session_state["run_progress_ratio"] = 0.0
        st.session_state["run_progress_text"] = "准备开始..."
        st.session_state["run_task_rows"] = []
        st.session_state["run_task_logs"] = {}
        st.session_state["run_task_order"] = []
        st.session_state["run_current_task"] = "未知任务"
        st.session_state["run_current_state_cn"] = "进行中"
        st.session_state["run_current_processed"] = 0
        st.session_state["run_current_total"] = 1
        st.session_state["run_current_message"] = "等待进度事件..."
        st.session_state["run_last_result"] = None

        event_queue: Queue[dict] = Queue()
        st.session_state["run_event_queue"] = event_queue

        def _worker() -> None:
            try:
                def on_progress(event: dict) -> None:
                    event_queue.put({"type": "progress", "event": event})

                result = run_daily_pipeline(
                    target_date=target_date,
                    timezone_name=selected_tz_name,
                    progress_callback=on_progress,
                )
                event_queue.put({"type": "done", "result": result})
            except Exception as exc:  # pragma: no cover - UI 异常提示路径
                event_queue.put({"type": "error", "error": str(exc)})

        thread = threading.Thread(target=_worker, daemon=True)
        st.session_state["run_thread"] = thread
        thread.start()

    if (run_today or run_selected) and not st.session_state["run_in_progress"]:
        target_date = None if run_today else run_date.strftime("%Y-%m-%d")
        _start_run(target_date)

    if st.session_state["run_in_progress"]:
        task_logs: dict[str, dict] = st.session_state["run_task_logs"]
        task_order: list[str] = st.session_state["run_task_order"]
        event_queue = st.session_state["run_event_queue"]

        while event_queue is not None:
            try:
                payload = event_queue.get_nowait()
            except Empty:
                break

            event_type = payload.get("type")
            if event_type == "progress":
                event = payload.get("event", {})
                task = str(event.get("task", "未知任务"))
                task_id = str(event.get("task_id", task))
                source = str(event.get("source", "-"))
                status = str(event.get("status", "running"))
                message = str(event.get("message", ""))
                processed = int(event.get("processed", 0))
                total = int(event.get("total", 1))

                state_cn = "进行中" if status == "running" else "完成"
                elapsed_text = _format_elapsed(perf_counter() - st.session_state["run_started_at"])
                progress_ratio = _safe_ratio(processed, total)
                progress_text = f"{task} {processed}/{total}"

                now = datetime.now(display_tz).strftime("%H:%M:%S")
                row = {
                    "time": now,
                    "task": task,
                    "source": source,
                    "status": state_cn,
                    "progress": f"{processed}/{total}",
                    "elapsed": elapsed_text,
                    "message": message,
                }
                if task_id not in task_logs:
                    task_order.append(task_id)
                task_logs[task_id] = row

                st.session_state["run_current_task"] = task
                st.session_state["run_current_state_cn"] = state_cn
                st.session_state["run_current_processed"] = processed
                st.session_state["run_current_total"] = total
                st.session_state["run_current_message"] = message
                st.session_state["run_progress_ratio"] = progress_ratio
                st.session_state["run_progress_text"] = progress_text
                st.session_state["run_task_rows"] = [task_logs[t] for t in task_order]
            elif event_type == "done":
                total_elapsed = _format_elapsed(perf_counter() - st.session_state["run_started_at"])
                result = payload.get("result", {})
                st.session_state["run_in_progress"] = False
                st.session_state["run_status_text"] = f"任务执行完成，总耗时：{total_elapsed}"
                st.session_state["run_progress_ratio"] = 1.0
                st.session_state["run_progress_text"] = "全部任务完成"
                st.session_state["run_last_result"] = result
            elif event_type == "error":
                error = str(payload.get("error", "未知错误"))
                st.session_state["run_in_progress"] = False
                st.session_state["run_status_text"] = f"任务执行失败：{error}"
                st.session_state["run_progress_text"] = "执行失败"

    if st.session_state["run_in_progress"]:
        elapsed_text = _format_elapsed(perf_counter() - st.session_state["run_started_at"])
        status_text = (
            f"**正在处理**: `{st.session_state['run_current_task']}` | "
            f"**状态**: `{st.session_state['run_current_state_cn']}` | "
            f"**进度**: `{st.session_state['run_current_processed']}/{st.session_state['run_current_total']}` | "
            f"**已运行**: `{elapsed_text}`\n\n{st.session_state['run_current_message']}"
        )
        st.session_state["run_status_text"] = status_text
        running_placeholder.markdown(status_text)
    else:
        status_text = str(st.session_state["run_status_text"])
        if status_text.startswith("任务执行完成"):
            running_placeholder.success(status_text)
        elif status_text.startswith("任务执行失败"):
            running_placeholder.error(status_text)
        else:
            running_placeholder.markdown(status_text)

    progress_placeholder.progress(
        float(st.session_state["run_progress_ratio"]),
        text=str(st.session_state["run_progress_text"]),
    )
    table_placeholder.dataframe(
        pd.DataFrame(
            st.session_state["run_task_rows"],
            columns=["time", "task", "source", "status", "progress", "elapsed", "message"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.session_state["run_last_result"]:
        result = st.session_state["run_last_result"]
        tz_value = result.get("timezone", selected_tz_name)
        st.sidebar.success(
            f"完成: {result['date']} ({tz_value}) | score={result['trend_score']} | articles={result['deduped_articles']}"
        )

    if st.session_state["run_in_progress"]:
        sleep(1)
        st.rerun()

    payloads = _load_json_reports()
    if not payloads:
        st.warning("未发现历史数据。请点击左侧按钮运行数据抓取。")
        st.stop()

    history = _history_frame(payloads)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.sort_values("date")

    date_options = [d.get("date", "") for d in payloads if d.get("date")]
    selected_date = st.sidebar.selectbox("选择日期", options=date_options, index=len(date_options) - 1)
    st.sidebar.divider()
    st.sidebar.subheader("模型状态")
    check_model = st.sidebar.button("检查LLM模型", use_container_width=True)
    if check_model:
        ok, msg = _check_sentiment_model_status()
        st.session_state["model_status_ok"] = ok
        st.session_state["model_status_msg"] = msg

    if "model_status_ok" in st.session_state and "model_status_msg" in st.session_state:
        if st.session_state["model_status_ok"]:
            st.sidebar.success(st.session_state["model_status_msg"])
        else:
            st.sidebar.error(st.session_state["model_status_msg"])
    else:
        st.sidebar.caption("点击按钮检查当前配置LLM模型是否可用")
    selected = next((x for x in payloads if x.get("date") == selected_date), payloads[-1])

    trend_score = float(selected.get("trend_score", 0.0))
    regime = selected.get("market_regime", "中性")
    article_count = int(selected.get("article_count", 0))
    summary = selected.get("summary", "")

    c1, c2, c3 = st.columns(3)
    c1.metric("TrendScore", f"{trend_score:.2f}")
    c2.metric("Market Regime", regime)
    c3.metric("Articles", str(article_count))
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:10px;background:{_regime_color(trend_score)}20;"
        f"border:1px solid {_regime_color(trend_score)};'>"
        f"<b>Summary:</b> {summary}</div>",
        unsafe_allow_html=True,
    )

    st.subheader("TrendScore 历史走势")
    history_plot = history.copy()
    history_plot["date_only"] = pd.to_datetime(history_plot["date"], errors="coerce").dt.normalize()
    fig_line = px.line(
        history_plot,
        x="date_only",
        y="trend_score",
        markers=True,
        title="Daily TrendScore",
        color_discrete_sequence=["#1f77b4"],
    )
    regime_line_colors = ["#117a65", "#1f8f4e", "#9a7d0a", "#af601a", "#922b21"]
    for idx, threshold in enumerate(_regime_threshold_lines()):
        fig_line.add_hline(
            y=threshold,
            line_dash="dot",
            line_color=regime_line_colors[min(idx, len(regime_line_colors) - 1)],
        )
    fig_line.update_xaxes(title_text="date", type="date", dtick=86400000, tickformat="%Y-%m-%d")
    fig_line.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_line, use_container_width=True)

    articles = _selected_day_articles(selected)
    if articles.empty:
        st.info("该日期没有文章明细。")
        st.stop()

    left, right = st.columns(2)
    topic_df = articles.groupby("topic", as_index=False).size().sort_values("size", ascending=False)
    fig_topic = px.bar(
        topic_df,
        x="topic",
        y="size",
        title=f"{selected_date} 主题分布",
        color="size",
        color_continuous_scale="Blues",
    )
    fig_topic.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    left.plotly_chart(fig_topic, use_container_width=True)

    direction_df = articles.groupby("direction_label", as_index=False).size()
    fig_dir = px.pie(
        direction_df,
        names="direction_label",
        values="size",
        title=f"{selected_date} 方向分布",
        color="direction_label",
        color_discrete_map={"正面": "#1f8f4e", "中性": "#b7950b", "负面": "#922b21"},
    )
    fig_dir.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
    right.plotly_chart(fig_dir, use_container_width=True)

    st.subheader("Top News（按 |article_score| 排序）")
    if "market_scope_label" not in articles.columns:
        if "market_scope" in articles.columns:
            articles["market_scope_label"] = articles["market_scope"].map(
                {"macro": "宏观大盘", "asset_specific": "个股/个别资产"}
            ).fillna("未知")
        else:
            articles["market_scope_label"] = "未知"
            articles["market_scope"] = "unknown"
    articles["abs_score"] = articles["article_score"].abs()
    display_df = (
        articles.sort_values("abs_score", ascending=False)[
            [
                "source",
                "title",
                "topic",
                "direction_label",
                "market_scope_label",
                "impact",
                "relevance",
                "article_score",
                "url",
            ]
        ]
        .head(30)
        .reset_index(drop=True)
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "source": "Source",
            "title": st.column_config.TextColumn("Title", width="large"),
            "topic": "Topic",
            "direction_label": "Direction",
            "market_scope_label": "MarketScope",
            "impact": st.column_config.NumberColumn("Impact", format="%.2f"),
            "relevance": st.column_config.NumberColumn("Relevance", format="%.2f"),
            "article_score": st.column_config.NumberColumn("ArticleScore", format="%.3f"),
            "url": st.column_config.LinkColumn("Link"),
        },
        hide_index=True,
    )


if __name__ == "__main__":
    main()
