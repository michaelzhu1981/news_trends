"""主流程管道：RSS 抓取 -> 正文提取 -> 日期过滤 -> 去重 -> 打分 -> 输出。"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
import re
from zoneinfo import ZoneInfo
from typing import Callable
from pathlib import Path

from .config import RAW_DIR, load_feeds, load_scoring_config
from .dedup import deduplicate_articles
from .fetcher import enrich_articles_with_content_async, fetch_all_rss_entries
from .models import Article
from .report import append_trend_history, write_json, write_markdown
from .scoring import score_articles


# 进度回调：接收任务事件字典
ProgressCallback = Callable[[dict], None]
_FILENAME_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


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


def _resolve_timezone(timezone_name: str | None):
    """解析时区；若未指定则使用运行环境本地时区。"""
    if timezone_name:
        try:
            return ZoneInfo(timezone_name)
        except Exception as exc:
            raise ValueError(f"无效时区: {timezone_name}") from exc

    local_tz = datetime.now().astimezone().tzinfo
    return local_tz or UTC


def _filter_articles_by_target_date(articles: list[Article], target_date: str, local_tz) -> list[Article]:
    """按指定时区筛选出发表日期为 target_date 的文章。"""
    return [a for a in articles if _article_on_target_date(a, target_date, local_tz)]


def _cleanup_recent_files(base_dir: Path, keep_days: int, anchor_date: date) -> int:
    """清理目录中过期文件：仅保留从 anchor_date 往前 keep_days 天内的文件。"""
    if keep_days <= 0 or not base_dir.exists():
        return 0

    cutoff = anchor_date - timedelta(days=keep_days - 1)
    removed = 0

    for path in base_dir.iterdir():
        if not path.is_file():
            continue
        matched = _FILENAME_DATE_RE.match(path.name)
        if not matched:
            continue
        try:
            file_date = date.fromisoformat(matched.group(1))
        except ValueError:
            continue
        if file_date < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def _latest_dated_file(base_dir: Path) -> date | None:
    """返回目录内按文件名前缀识别出的最新日期。"""
    if not base_dir.exists():
        return None

    latest: date | None = None
    for path in base_dir.iterdir():
        if not path.is_file():
            continue
        matched = _FILENAME_DATE_RE.match(path.name)
        if not matched:
            continue
        try:
            file_date = date.fromisoformat(matched.group(1))
        except ValueError:
            continue
        if latest is None or file_date > latest:
            latest = file_date
    return latest


def run_daily_pipeline(
    target_date: str | None = None,
    timezone_name: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str | int | float]:
    """执行完整日报流程：抓取 RSS -> 正文提取 -> 日期过滤 -> 去重 -> 打分 -> 写入 JSON/Markdown。返回统计信息字典。"""
    local_tz = _resolve_timezone(timezone_name)
    timezone_label = getattr(local_tz, "key", str(local_tz))
    date_str = target_date or datetime.now(local_tz).strftime("%Y-%m-%d")
    _emit_progress(
        progress_callback,
        {
            "task": "初始化",
            "task_id": "初始化",
            "status": "running",
            "message": f"目标日期 {date_str}（时区 {timezone_label}）",
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
            "message": f"按时区 {timezone_label} 筛选 {date_str} 新闻",
            "processed": 0,
            "total": 1,
        },
    )
    filtered = _filter_articles_by_target_date(enriched, date_str, local_tz)
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
    deduped = deduplicate_articles(filtered, scoring_cfg)
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
    history_path = append_trend_history(result, scoring_cfg)
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

    anchor_candidates = [
        date.fromisoformat(date_str),
        _latest_dated_file(json_path.parent),
        _latest_dated_file(RAW_DIR),
        _latest_dated_file(report_path.parent),
    ]
    anchor_day = max(d for d in anchor_candidates if d is not None)
    removed_processed = _cleanup_recent_files(json_path.parent, keep_days=3, anchor_date=anchor_day)
    removed_raw = _cleanup_recent_files(RAW_DIR, keep_days=3, anchor_date=anchor_day)
    removed_reports = _cleanup_recent_files(report_path.parent, keep_days=3, anchor_date=anchor_day)
    _emit_progress(
        progress_callback,
        {
            "task": "完成",
            "task_id": "完成",
            "status": "completed",
            "message": (
                "全部流程完成"
                f"（清理: processed={removed_processed}, raw={removed_raw}, reports={removed_reports}）"
            ),
            "processed": 1,
            "total": 1,
        },
    )

    return {
        "date": date_str,
        "timezone": timezone_label,
        "raw_articles": len(all_articles),
        "date_filtered_articles": len(filtered),
        "deduped_articles": len(deduped),
        "trend_score": result.trend_score,
        "report_path": str(report_path),
        "json_path": str(json_path),
        "history_path": str(history_path),
    }
