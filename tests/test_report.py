"""报告模块测试。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_trends.config import load_scoring_config
from news_trends.models import DailyResult, ScoredArticle
from news_trends import report


def test_write_markdown_uses_regime_thresholds_from_config(tmp_path, monkeypatch):
    """验证：区间解释由 scoring.yaml 的 regime_thresholds 配置驱动。"""
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)
    cfg = load_scoring_config()
    result = DailyResult(
        date="2026-02-20",
        trend_score=35.0,
        article_count=0,
        summary="test-summary",
        scored_articles=[],
    )

    out = report.write_markdown(result, cfg)
    text = out.read_text(encoding="utf-8")

    assert "## 区间解释" in text
    assert "- +80 ~ +100: 强 Risk-On" in text
    assert "- +40 ~ +80: 偏多" in text
    assert "- -10 ~ +40: 中性" in text
    assert "- -40 ~ -10: 偏空" in text
    assert "- -100 ~ -40: 强 Risk-Off" in text


def test_write_markdown_summary_removes_links(tmp_path, monkeypatch):
    """验证：报告中的摘要不包含链接。"""
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)
    cfg = load_scoring_config()
    result = DailyResult(
        date="2026-02-20",
        trend_score=5.0,
        article_count=1,
        summary="主线见[这里](https://example.com/x) 以及 https://foo.bar/a",
        scored_articles=[
            ScoredArticle(
                source="test",
                source_category="general",
                title="t1",
                url="https://example.com/news/1",
                published_at=None,
                summary='详见<a href="https://a.com">公告</a> 和 https://b.com',
            )
        ],
    )

    out = report.write_markdown(result, cfg)
    text = out.read_text(encoding="utf-8")

    assert "- Summary: 主线见这里 以及" in text
    assert "- 摘要: 详见公告 和" in text
    assert "https://foo.bar/a" not in text
    assert "https://b.com" not in text
    assert "- 链接: https://example.com/news/1" in text


def test_write_markdown_summary_removes_html_entities_and_noise(tmp_path, monkeypatch):
    """验证：摘要中的 HTML entity、标签与异常来源标识被清理。"""
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)
    cfg = load_scoring_config()
    result = DailyResult(
        date="2026-02-20",
        trend_score=5.0,
        article_count=1,
        summary='市场回暖&nbsp;&nbsp;<font color="#6f6f6f">FinancialContent</font>',
        scored_articles=[],
    )

    out = report.write_markdown(result, cfg)
    text = out.read_text(encoding="utf-8")

    assert "- Summary: 市场回暖" in text
    assert "&nbsp;" not in text
    assert "<font" not in text
    assert "FinancialContent" not in text


def test_write_markdown_top_news_keeps_similar_titles(tmp_path, monkeypatch):
    """验证：Top News 展示不去重，相似标题可同时出现。"""
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)
    cfg = load_scoring_config()
    result = DailyResult(
        date="2026-02-20",
        trend_score=20.0,
        article_count=3,
        summary="test-summary",
        scored_articles=[
            ScoredArticle(
                source="x",
                source_category="finance",
                title="Treasury Yields Rise as Fed Signals No Rush to Cut",
                url="https://example.com/1",
                published_at=None,
                summary="s1",
                topic="fed",
                article_score=0.9,
                direction=1,
                direction_label="正面",
                market_scope="macro",
                market_scope_label="宏观大盘",
                relevance=0.6,
                impact=0.9,
                scope_weight=1.25,
            ),
            ScoredArticle(
                source="y",
                source_category="finance",
                title="Treasury Yields Move Higher as Fed Signals No Rush to Cut",
                url="https://example.com/2",
                published_at=None,
                summary="s2",
                topic="fed",
                article_score=0.8,
                direction=1,
                direction_label="正面",
                market_scope="macro",
                market_scope_label="宏观大盘",
                relevance=0.6,
                impact=0.9,
                scope_weight=1.25,
            ),
            ScoredArticle(
                source="z",
                source_category="finance",
                title="Oil prices jump on Middle East tensions",
                url="https://example.com/3",
                published_at=None,
                summary="s3",
                topic="energy",
                article_score=-0.7,
                direction=-1,
                direction_label="负面",
                market_scope="macro",
                market_scope_label="宏观大盘",
                relevance=0.6,
                impact=0.9,
                scope_weight=1.25,
            ),
        ],
    )

    out = report.write_markdown(result, cfg)
    text = out.read_text(encoding="utf-8")

    assert "### Treasury Yields Rise as Fed Signals No Rush to Cut" in text
    assert "### Treasury Yields Move Higher as Fed Signals No Rush to Cut" in text
    assert "### Oil prices jump on Middle East tensions" in text
