"""去重模块测试。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_trends.dedup import deduplicate_articles
from news_trends.models import Article


def _article(title: str, url: str, content: str = "") -> Article:
    return Article(
        source="test",
        source_category="finance",
        title=title,
        url=url,
        published_at=None,
        summary="",
        content=content,
    )


def _sentiment_cfg() -> dict:
    return {
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5",
        "max_retries": 0,
        "dedup_pair_llm_system_prompt": "pair system",
        "dedup_pair_llm_user_template": "Candidates:\n{candidate_pairs_json}",
    }


def test_deduplicate_articles_uses_pair_llm_only(monkeypatch):
    articles = [
        _article("Fed hints at rate cuts in 2026", "https://example.com/1", "a"),
        _article("Federal Reserve hints at rate cuts in 2026", "https://example.com/2", "b"),
        _article("Oil jumps after OPEC output decision", "https://example.com/3", "c"),
    ]
    cfg = {"sentiment": _sentiment_cfg()}
    seen: list[str] = []

    class _Resp:
        def __init__(self, content: str):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    def _fake_post(*args, **kwargs):
        seen.append(kwargs["json"]["messages"][1]["content"])
        return _Resp('{"duplicate_pair_ids":[0]}')

    monkeypatch.setattr("news_trends.dedup.requests.post", _fake_post)
    deduped = deduplicate_articles(articles, cfg)

    assert len(seen) == 1
    assert "Fed hints at rate cuts in 2026" in seen[0]
    assert "Federal Reserve hints at rate cuts in 2026" in seen[0]
    assert "Oil jumps after OPEC output decision" not in seen[0]
    assert [x.url for x in deduped] == ["https://example.com/2", "https://example.com/3"]


def test_deduplicate_articles_fallback_to_local_obvious_dedup_when_llm_fails(monkeypatch):
    articles = [
        _article("Treasury yields rise as Fed signals rate cut", "https://example.com/1", "a"),
        _article("Treasury yields rise as Fed signals rate cut", "https://example.com/2", "b"),
    ]
    cfg = {"sentiment": _sentiment_cfg()}

    def _fake_post(*args, **kwargs):
        raise RuntimeError("mock llm failure")

    monkeypatch.setattr("news_trends.dedup.requests.post", _fake_post)
    deduped = deduplicate_articles(articles, cfg)

    assert [x.url for x in deduped] == ["https://example.com/1"]


def test_deduplicate_articles_handles_same_headline_different_source_suffix(monkeypatch):
    articles = [
        _article(
            "Real estate stocks slump as Iran conflict pushes Treasury yields up, muddles path to rate cuts - Seeking Alpha",
            "https://example.com/1",
        ),
        _article(
            "Real estate stocks slump as Iran conflict pushes Treasury yields up, muddles path to rate cuts - MSN",
            "https://example.com/2",
        ),
    ]
    cfg = {"sentiment": _sentiment_cfg()}

    def _fake_post(*args, **kwargs):
        raise RuntimeError("mock llm failure")

    monkeypatch.setattr("news_trends.dedup.requests.post", _fake_post)
    deduped = deduplicate_articles(articles, cfg)
    assert len(deduped) == 1


def test_deduplicate_articles_prefers_more_informative_title(monkeypatch):
    articles = [
        _article("Trump raises tariffs", "https://example.com/1"),
        _article("Trump raises tariffs to 15% on imports from all countries", "https://example.com/2"),
    ]
    cfg = {"sentiment": _sentiment_cfg()}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"duplicate_pair_ids":[0]}'}}]}

    monkeypatch.setattr("news_trends.dedup.requests.post", lambda *args, **kwargs: _Resp())
    deduped = deduplicate_articles(articles, cfg)
    assert [x.url for x in deduped] == ["https://example.com/2"]
