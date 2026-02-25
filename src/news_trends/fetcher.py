"""RSS 抓取与正文提取模块。"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Callable, Iterable

import aiohttp
import feedparser
import requests
import trafilatura
from dateutil import parser as date_parser

from .models import Article


# HTTP 请求 User-Agent（部分源会校验）
USER_AGENT = "Mozilla/5.0 (compatible; NewsTrendBot/1.0; +https://example.local)"
# 请求超时秒数
TIMEOUT = 15
# 异步抓取并发上限
DEFAULT_FETCH_CONCURRENCY = 20
# 进程池默认 worker 数
DEFAULT_PARSE_WORKERS = max(1, min(8, (os.cpu_count() or 1)))
# 正文异常长度上限（字符）
MAX_CONTENT_CHARS = 15_000
# 进度回调类型：接收一个事件字典
ProgressCallback = Callable[[dict], None]


def _parse_published(entry: dict) -> datetime | None:
    """从 RSS 条目中解析发布时间，优先 published，其次 updated、created。"""
    candidates = [
        entry.get("published"),
        entry.get("updated"),
        entry.get("created"),
    ]
    for value in candidates:
        if value:
            try:
                return date_parser.parse(value)
            except Exception:
                continue
    return None


def fetch_rss_entries(feed_name: str, category: str, url: str) -> list[Article]:
    """抓取指定 RSS 源，返回 Article 列表。失败时返回空列表。"""
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    parsed = feedparser.parse(resp.content)
    articles: list[Article] = []

    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        summary = (entry.get("summary") or "").strip()
        if not title or not link:
            continue

        articles.append(
            Article(
                source=feed_name,
                source_category=category,
                title=title,
                url=link,
                published_at=_parse_published(entry),
                summary=summary,
            )
        )

    return articles


def fetch_full_text(url: str) -> str:
    """从 URL 抓取页面并用 trafilatura 提取正文。失败时返回空字符串。"""
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        downloaded = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        return _truncate_abnormal_content(downloaded or "")
    except Exception:
        return ""


def _emit_progress(callback: ProgressCallback | None, event: dict) -> None:
    """安全调用进度回调，忽略异常。"""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        return


async def _http_get_text(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """异步获取页面文本，失败返回空字符串。"""
    try:
        if semaphore is not None:
            async with semaphore:
                async with session.get(url, timeout=TIMEOUT) as resp:
                    if resp.status >= 400:
                        return ""
                    return await resp.text()
        async with session.get(url, timeout=TIMEOUT) as resp:
            if resp.status >= 400:
                return ""
            return await resp.text()
    except Exception:
        return ""


def _extract_from_html(html: str) -> str:
    """进程池 worker：从 HTML 提取正文。"""
    if not html:
        return ""
    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        return _truncate_abnormal_content(text or "")
    except Exception:
        return ""


def _truncate_abnormal_content(text: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    """对异常超长正文做截断，避免后续处理和模型请求异常膨胀。"""
    cleaned = text.strip()
    if max_chars < 1:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return "." * max_chars
    return f"{cleaned[: max_chars - 3].rstrip()}..."


async def _fetch_rss_entries_async(
    feed_name: str,
    category: str,
    url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> list[Article]:
    """异步抓取指定 RSS 源，返回 Article 列表。失败时返回空列表。"""
    content = await _http_get_text(session, url, semaphore=semaphore)
    if not content:
        return []

    parsed = feedparser.parse(content)
    articles: list[Article] = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        summary = (entry.get("summary") or "").strip()
        if not title or not link:
            continue
        articles.append(
            Article(
                source=feed_name,
                source_category=category,
                title=title,
                url=link,
                published_at=_parse_published(entry),
                summary=summary,
            )
        )
    return articles


async def fetch_all_rss_entries(
    feeds: list[dict],
    progress_callback: ProgressCallback | None = None,
    concurrency: int = DEFAULT_FETCH_CONCURRENCY,
) -> list[Article]:
    """并发抓取所有 RSS 源。"""
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    headers = {"User-Agent": USER_AGENT}
    semaphore = asyncio.Semaphore(max(1, concurrency))
    all_articles: list[Article] = []
    total_feeds = len(feeds)

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async def _run_one(idx: int, feed: dict) -> list[Article]:
            name = feed.get("name", "unknown")
            category = feed.get("category", "general")
            url = feed.get("url", "")
            if not url:
                return []
            _emit_progress(
                progress_callback,
                {
                    "task": "RSS抓取",
                    "task_id": f"RSS::{name}",
                    "source": name,
                    "status": "running",
                    "message": f"[{idx}/{total_feeds}] {name}",
                    "processed": idx - 1,
                    "total": total_feeds,
                },
            )
            entries = await _fetch_rss_entries_async(name, category, url, session=session, semaphore=semaphore)
            _emit_progress(
                progress_callback,
                {
                    "task": "RSS抓取",
                    "task_id": f"RSS::{name}",
                    "source": name,
                    "status": "completed",
                    "message": f"[{idx}/{total_feeds}] {name} -> {len(entries)} 条",
                    "processed": idx,
                    "total": total_feeds,
                },
            )
            return entries

        tasks = [asyncio.create_task(_run_one(idx, feed)) for idx, feed in enumerate(feeds, start=1)]
        for task in asyncio.as_completed(tasks):
            all_articles.extend(await task)
    return all_articles


async def enrich_articles_with_content_async(
    articles: Iterable[Article],
    progress_callback: ProgressCallback | None = None,
    fetch_concurrency: int = DEFAULT_FETCH_CONCURRENCY,
    parse_workers: int = DEFAULT_PARSE_WORKERS,
) -> list[Article]:
    """异步抓取正文页面 + 多进程解析正文。"""
    article_list = list(articles)
    total = len(article_list)
    if total == 0:
        return []

    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    headers = {"User-Agent": USER_AGENT}
    fetch_sem = asyncio.Semaphore(max(1, fetch_concurrency))

    total_steps = total * 2
    html_list: list[str] = [""] * total
    async def _fetch_one(idx: int, article: Article, session: aiohttp.ClientSession) -> tuple[int, str]:
        html = await _http_get_text(session, article.url, semaphore=fetch_sem)
        return idx, html

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        fetch_tasks = [
            asyncio.create_task(_fetch_one(idx, article, session))
            for idx, article in enumerate(article_list, start=1)
        ]
        for fetched_count, done in enumerate(asyncio.as_completed(fetch_tasks), start=1):
            idx, html = await done
            html_list[idx - 1] = html
            _emit_progress(
                progress_callback,
                {
                    "task": "正文提取",
                    "task_id": f"正文::抓取::{idx}",
                    "source": article_list[idx - 1].source,
                    "status": "running",
                    "message": f"抓取页面 {fetched_count}/{total}",
                    "processed": fetched_count,
                    "total": total_steps,
                },
            )

    loop = asyncio.get_running_loop()
    enriched_by_index: list[Article | None] = [None] * total

    async def _parse_one(idx: int, article: Article, html: str) -> tuple[int, Article]:
        content = await loop.run_in_executor(pool, _extract_from_html, html)
        article.content = content or article.summary
        return idx, article

    with ProcessPoolExecutor(max_workers=max(1, parse_workers)) as pool:
        parse_tasks = [
            asyncio.create_task(_parse_one(idx, article, html))
            for idx, (article, html) in enumerate(zip(article_list, html_list), start=1)
        ]
        for processed, done in enumerate(asyncio.as_completed(parse_tasks), start=1):
            idx, article = await done
            enriched_by_index[idx - 1] = article
            _emit_progress(
                progress_callback,
                {
                    "task": "正文提取",
                    "task_id": f"正文::{idx}",
                    "source": article.source,
                    "status": "running",
                    "message": article.title[:90],
                    "processed": total + processed,
                    "total": total_steps,
                },
            )
    return [article for article in enriched_by_index if article is not None]


def enrich_articles_with_content(
    articles: Iterable[Article],
    progress_callback: ProgressCallback | None = None,
) -> list[Article]:
    """兼容旧调用：默认走异步抓取 + 多进程解析。"""
    return asyncio.run(
        enrich_articles_with_content_async(
            articles,
            progress_callback=progress_callback,
        )
    )
