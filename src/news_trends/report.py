"""报告输出模块：JSON 与 Markdown。"""

from __future__ import annotations

import json
from html import unescape
import re
from pathlib import Path
from typing import Any

from .config import PROCESSED_DIR, REPORT_DIR
from .models import DailyResult
from .scoring import classify_market_regime


def _top_news(result: DailyResult, k: int = 12):
    """按 |article_score| 选 Top News，不做展示层去重。"""
    ranked = sorted(result.scored_articles, key=lambda x: abs(x.article_score), reverse=True)
    return ranked[:k]


def _regime_thresholds_from_cfg(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """从配置中提取 regime_thresholds，必须在 scoring.yaml 的 rules.regime_thresholds 中定义。"""
    rules = cfg.get("rules", {}) or {}
    thresholds = rules.get("regime_thresholds")
    if not thresholds or not isinstance(thresholds, list):
        raise ValueError("scoring.yaml 中需定义 rules.regime_thresholds")
    return thresholds


def _format_bound(value: float | int) -> str:
    """将区间边界格式化为带符号的紧凑数字字符串。"""
    num = float(value)
    if num.is_integer():
        return f"{num:+.0f}"
    return f"{num:+.2f}".rstrip("0").rstrip(".")


def _regime_explanations(regime_thresholds: list[dict[str, Any]]) -> list[str]:
    """按阈值配置生成“区间解释”行。"""
    explanations: list[str] = []
    prev_min_score: float | None = 100.0

    for item in regime_thresholds:
        label = str(item.get("label", "")).strip()
        if not label:
            continue

        min_score = item.get("min_score")
        if min_score is None:
            explanations.append(f"- -100 ~ {_format_bound(prev_min_score)}: {label}")
            prev_min_score = None
            continue

        lower = float(min_score)
        upper = prev_min_score if prev_min_score is not None else 100.0
        explanations.append(f"- {_format_bound(lower)} ~ {_format_bound(upper)}: {label}")
        prev_min_score = lower

    return explanations


def _strip_links(text: str) -> str:
    """清洗摘要中的链接、HTML 残留与异常字符，保留可读文本。"""
    cleaned = text or ""
    cleaned = unescape(cleaned)
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = re.sub(r"<a\s+[^>]*href=[\"'][^\"']+[\"'][^>]*>(.*?)</a>", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"(?:https?://|www\.)\S+", "", cleaned)
    cleaned = re.sub(r"</?font\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\bFinancialContent\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def write_json(result: DailyResult, cfg: dict[str, Any]) -> Path:
    """将 DailyResult 写入 data/processed/{date}_scored.json。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{result.date}_scored.json"

    payload = {
        "date": result.date,
        "trend_score": result.trend_score,
        "article_count": result.article_count,
        "summary": result.summary,
        "market_regime": classify_market_regime(result.trend_score, _regime_thresholds_from_cfg(cfg)),
        "articles": [
            {
                "source": a.source,
                "title": a.title,
                "url": a.url,
                "topic": a.topic,
                "direction": a.direction,
                "direction_label": a.direction_label,
                "market_scope": a.market_scope,
                "market_scope_label": a.market_scope_label,
                "relevance": a.relevance,
                "impact": a.impact,
                "scope_weight": a.scope_weight,
                "article_score": a.article_score,
                "published_at": a.published_at.isoformat() if a.published_at else None,
                "summary": (a.summary or "")[:400],
            }
            for a in result.scored_articles
        ],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def write_markdown(result: DailyResult, cfg: dict[str, Any]) -> Path:
    """将 DailyResult 写入 reports/{date}.md，包含总览、区间解释与 Top News。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"{result.date}.md"
    regime_thresholds = _regime_thresholds_from_cfg(cfg)
    regime = classify_market_regime(result.trend_score, regime_thresholds)
    top_news = _top_news(result)

    lines: list[str] = []
    lines.append(f"# S&P500 新闻宏观趋势评分日报 - {result.date}")
    lines.append("")
    lines.append("## 今日总览")
    lines.append(f"- TrendScore: **{result.trend_score}**")
    lines.append(f"- Market Regime: **{regime}**")
    lines.append(f"- Articles: **{result.article_count}**")
    lines.append(f"- Summary: {_strip_links(result.summary)}")
    lines.append("")
    lines.append("## 区间解释")
    lines.extend(_regime_explanations(regime_thresholds))
    lines.append("")
    lines.append("## Top News")

    if not top_news:
        lines.append("- 当日无可用新闻。")
    else:
        for item in top_news:
            lines.append(f"### {item.title}")
            lines.append(f"- 摘要: {_strip_links((item.summary or '').strip())}")
            lines.append(f"- 方向: {item.direction_label}")
            lines.append(f"- Market Scope: {item.market_scope_label} ({item.market_scope})")
            lines.append(f"- 影响强度: {item.impact}")
            lines.append(f"- 主题: {item.topic}")
            lines.append(f"- 相关性: {item.relevance}")
            lines.append(f"- 链接: {item.url}")
            lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    return out_path
