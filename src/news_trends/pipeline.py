"""主流程管道：RSS 抓取 -> 正文提取 -> 日期过滤 -> 去重 -> 打分 -> 输出。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Callable

from .config import RAW_DIR, load_feeds, load_scoring_config
from .dedup import deduplicate_articles
from .fetcher import enrich_articles_with_content_async, fetch_all_rss_entries
from .models import Article
from .report import write_json, write_markdown
from .scoring import score_articles


# 进度回调：接收任务事件字典
ProgressCallback = Callable[[dict], None]


def _emit_progress(callback: ProgressCallback | None, event: dict) -> None:
    """安全调用进度回调，忽略异常。"""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        return


def _article_on_target_date(article: Article, target_date: str, local_tz) -> bool:
    """判断文章发布时间（转本地时区后）是否等于目标日期。"""
    if article.published_at is None:
        return False

    dt = article.published_at
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    local_dt = dt.astimezone(local_tz)
    return local_dt.strftime("%Y-%m-%d") == target_date


def _filter_articles_by_target_date(articles: list[Article], target_date: str) -> list[Article]:
    """按本地时区筛选出发表日期为 target_date 的文章。"""
    local_tz = datetime.now().astimezone().tzinfo
    return [a for a in articles if _article_on_target_date(a, target_date, local_tz)]


def run_daily_pipeline(
    target_date: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str | int | float]:
    """执行完整日报流程：抓取 RSS -> 正文提取 -> 日期过滤 -> 去重 -> 打分 -> 写入 JSON/Markdown。返回统计信息字典。"""
    date_str = target_date or datetime.now().strftime("%Y-%m-%d")
    _emit_progress(
        progress_callback,
        {
            "task": "初始化",
            "task_id": "初始化",
            "status": "running",
            "message": f"目标日期 {date_str}",
            "processed": 0,
            "total": 1,
        },
    )

    feeds = load_feeds()
    scoring_cfg = load_scoring_config()
    _emit_progress(
        progress_callback,
        {
            "task": "初始化",
            "task_id": "初始化",
            "status": "completed",
            "message": f"加载配置完成，feeds={len(feeds)}",
            "processed": 1,
            "total": 1,
        },
    )

    all_articles = asyncio.run(fetch_all_rss_entries(feeds, progress_callback=progress_callback))

    _emit_progress(
        progress_callback,
        {
            "task": "正文提取",
            "task_id": "正文提取",
            "status": "running",
            "message": f"开始提取正文，共 {len(all_articles)} 条",
            "processed": 0,
            "total": max(1, len(all_articles)),
        },
    )
    enriched = asyncio.run(
        enrich_articles_with_content_async(
            all_articles,
            progress_callback=progress_callback,
        )
    )
    _emit_progress(
        progress_callback,
        {
            "task": "正文提取",
            "task_id": "正文提取",
            "status": "completed",
            "message": f"正文提取完成，共 {len(enriched)} 条",
            "processed": max(1, len(all_articles)),
            "total": max(1, len(all_articles)),
        },
    )

    _emit_progress(
        progress_callback,
        {
            "task": "日期过滤",
            "task_id": "日期过滤",
            "status": "running",
            "message": f"按本地时区筛选 {date_str} 新闻",
            "processed": 0,
            "total": 1,
        },
    )
    filtered = _filter_articles_by_target_date(enriched, date_str)
    _emit_progress(
        progress_callback,
        {
            "task": "日期过滤",
            "task_id": "日期过滤",
            "status": "completed",
            "message": f"筛选后 {len(filtered)}/{len(enriched)} 条",
            "processed": 1,
            "total": 1,
        },
    )

    _emit_progress(
        progress_callback,
        {
            "task": "去重",
            "task_id": "去重",
            "status": "running",
            "message": f"去重前 {len(filtered)} 条",
            "processed": 0,
            "total": 1,
        },
    )
    deduped = deduplicate_articles(filtered)
    _emit_progress(
        progress_callback,
        {
            "task": "去重",
            "task_id": "去重",
            "status": "completed",
            "message": f"去重后 {len(deduped)} 条",
            "processed": 1,
            "total": 1,
        },
    )

    _emit_progress(
        progress_callback,
        {"task": "打分", "task_id": "打分", "status": "running", "message": "开始分类与打分", "processed": 0, "total": 1},
    )
    result = score_articles(date_str, deduped, scoring_cfg)
    _emit_progress(
        progress_callback,
        {
            "task": "打分",
            "task_id": "打分",
            "status": "completed",
            "message": f"TrendScore={result.trend_score}",
            "processed": 1,
            "total": 1,
        },
    )

    _emit_progress(
        progress_callback,
        {"task": "输出", "task_id": "输出", "status": "running", "message": "写入 JSON/Markdown 报告", "processed": 0, "total": 1},
    )
    json_path = write_json(result, scoring_cfg)
    report_path = write_markdown(result, scoring_cfg)
    _emit_progress(
        progress_callback,
        {
            "task": "输出",
            "task_id": "输出",
            "status": "completed",
            "message": f"报告已生成: {date_str}",
            "processed": 1,
            "total": 1,
        },
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_count_path = RAW_DIR / f"{date_str}_count.txt"
    raw_count_path.write_text(
        (
            f"raw_articles={len(all_articles)}\n"
            f"enriched={len(enriched)}\n"
            f"date_filtered={len(filtered)}\n"
            f"deduped={len(deduped)}\n"
        ),
        encoding="utf-8",
    )
    _emit_progress(
        progress_callback,
        {"task": "完成", "task_id": "完成", "status": "completed", "message": "全部流程完成", "processed": 1, "total": 1},
    )

    return {
        "date": date_str,
        "raw_articles": len(all_articles),
        "date_filtered_articles": len(filtered),
        "deduped_articles": len(deduped),
        "trend_score": result.trend_score,
        "report_path": str(report_path),
        "json_path": str(json_path),
    }
