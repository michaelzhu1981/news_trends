"""打分配置与 score_articles 测试。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_trends.models import Article
from news_trends.scoring import _contains_any, calc_impact, detect_directions_llm, detect_market_scopes, score_articles
from news_trends.config import load_scoring_config


def test_scoring_range():
    """验证：TrendScore 在 [-100, 100] 区间，且 article_count 正确。"""
    cfg = load_scoring_config()
    cfg.setdefault("sentiment", {})
    cfg["sentiment"]["backend"] = "rule"
    articles = [
        Article(
            source="x",
            source_category="finance",
            title="Fed signals rate cut and cooling inflation",
            url="https://example.com/1",
            published_at=None,
            summary="soft landing hope",
            content="Fed suggests rate cut with cooling inflation; equities rebound.",
        ),
        Article(
            source="y",
            source_category="geopolitics",
            title="War escalates and oil spikes",
            url="https://example.com/2",
            published_at=None,
            summary="risk-off",
            content="war escalates, energy shock, stock market pressured",
        ),
    ]

    result = score_articles("2026-02-18", articles, cfg)
    assert -100 <= result.trend_score <= 100
    assert result.article_count == 2


def test_contains_any_matches_word_boundaries():
    """验证：关键词按词边界/短语匹配，避免子串误命中。"""
    text = "the market is in risk off mode during emergency meeting"
    terms = ["risk-off", "risk", "off", "merg"]
    normalized_terms = [x.replace("-", " ") for x in terms]
    assert _contains_any(text, normalized_terms) == 3


def test_calc_impact_uses_diminishing_boost():
    """验证：impact 冲击词加成采用 ln(1+hits) 边际递减。"""
    impact = calc_impact(
        topic="general",
        text="shock emergency crash downgrade",
        topic_weights={"general": 0.4},
        boost_terms=["shock", "emergency", "crash", "downgrade"],
        event_cluster_count_today=1,
        impact_boost_per_term=0.1,
        same_topic_frequency_boost=0.0,
        max_impact=1.0,
    )
    assert round(impact, 6) == round(0.4 + 0.1 * 1.6094379124341003, 6)


def test_same_topic_frequency_boost_uses_event_cluster_size():
    """验证：频次加成按事件簇频次，而不是主题频次。"""
    cfg = load_scoring_config()
    cfg.setdefault("sentiment", {})
    cfg["sentiment"]["backend"] = "rule"
    cfg["rules"]["same_topic_frequency_boost"] = 0.2
    cfg["rules"]["impact_boost_per_term"] = 0.0
    cfg["rules"]["same_event_decay_base"] = 1.0
    cfg["rules"]["same_event_decay_min_factor"] = 1.0

    articles = [
        Article(
            source="a",
            source_category="finance",
            title="Fed announces policy path update",
            url="https://example.com/a",
            published_at=None,
            summary="market watches fed guidance",
            content="federal reserve updates policy communication",
        ),
        Article(
            source="b",
            source_category="finance",
            title="Fed discusses balance sheet reduction plan",
            url="https://example.com/b",
            published_at=None,
            summary="market watches fed communication",
            content="federal reserve discusses balance sheet runoff",
        ),
    ]

    result = score_articles("2026-02-18", articles, cfg)
    impacts = [x.impact for x in result.scored_articles]
    assert impacts == [0.9, 0.9]


def test_llm_chat_parse_json_direction(monkeypatch):
    """验证：llm_chat 能解析 chat/completions JSON 输出。"""
    cfg = {
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "direction_mapping": {"negative": -1, "neutral": 0, "positive": 1},
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"direction":"positive"}',
                        }
                    }
                ]
            }

    monkeypatch.setattr("news_trends.scoring.requests.post", lambda *args, **kwargs: _Resp())
    dirs = detect_directions_llm(["headline"], cfg)
    assert dirs == [1]


def test_llm_user_template_with_json_braces(monkeypatch):
    """验证：llm_user_template 中 JSON 花括号不会导致 format 报错。"""
    cfg = {
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "llm_user_template": (
            "Classify and return JSON only: "
            '{"direction":"negative|neutral|positive","market_scope":"macro|asset_specific"}\nText:\n{text}'
        ),
        "direction_mapping": {"negative": -1, "neutral": 0, "positive": 1},
    }
    seen = {"called": False, "user_content": ""}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"direction":"neutral"}'}}]}

    def _fake_post(*args, **kwargs):
        seen["called"] = True
        seen["user_content"] = kwargs["json"]["messages"][1]["content"]
        return _Resp()

    monkeypatch.setattr("news_trends.scoring.requests.post", _fake_post)
    dirs = detect_directions_llm(["headline"], cfg)
    assert seen["called"] is True
    assert '{"direction":"negative|neutral|positive","market_scope":"macro|asset_specific"}' in seen["user_content"]
    assert "headline" in seen["user_content"]
    assert dirs == [0]


def test_llm_chat_parse_scope(monkeypatch):
    """验证：llm_chat 能解析 market_scope 输出。"""
    cfg = {
        "backend": "llm_chat",
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "scope_mapping": {"macro": "macro", "asset_specific": "asset_specific"},
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"direction":"positive","market_scope":"macro"}',
                        }
                    }
                ]
            }

    monkeypatch.setattr("news_trends.scoring.requests.post", lambda *args, **kwargs: _Resp())
    scopes = detect_market_scopes(["headline"], {"sentiment": cfg})
    assert scopes == ["macro"]


def test_score_articles_with_llm_backend(monkeypatch):
    """验证：score_articles 在 llm_chat 后端下输出方向。"""
    cfg = load_scoring_config()
    cfg["sentiment"] = {
        "backend": "llm_chat",
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "direction_mapping": {"negative": -1, "neutral": 0, "positive": 1},
        "scope_mapping": {"macro": "macro", "asset_specific": "asset_specific"},
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"direction":"negative","market_scope":"asset_specific"}',
                        }
                    }
                ]
            }

    monkeypatch.setattr("news_trends.scoring.requests.post", lambda *args, **kwargs: _Resp())
    articles = [
        Article("a", "finance", "t1", "https://e.com/1", None, "s", "c"),
    ]
    result = score_articles("2026-02-18", articles, cfg)
    assert [x.direction for x in result.scored_articles] == [-1]
    assert [x.market_scope for x in result.scored_articles] == ["asset_specific"]


def test_score_articles_skip_failed_llm_item(monkeypatch):
    """验证：单条新闻调用 LLM 失败时会跳过，不中断整体流程。"""
    cfg = load_scoring_config()
    cfg["sentiment"] = {
        "backend": "llm_chat",
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "direction_mapping": {"negative": -1, "neutral": 0, "positive": 1},
        "scope_mapping": {"macro": "macro", "asset_specific": "asset_specific"},
    }
    articles = [
        Article("a", "finance", "ok", "https://e.com/1", None, "s", "good content"),
        Article("b", "finance", "bad", "https://e.com/2", None, "s", "bad content"),
    ]

    class _Resp:
        def __init__(self, content: str):
            self._content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    def _fake_post(*args, **kwargs):
        user_content = kwargs["json"]["messages"][1]["content"]
        if "\nbad\n" in f"\n{user_content}\n":
            raise RuntimeError("mock llm failure")
        return _Resp('{"direction":"positive","market_scope":"macro"}')

    monkeypatch.setattr("news_trends.scoring.requests.post", _fake_post)
    result = score_articles("2026-02-18", articles, cfg)

    assert result.article_count == 1
    assert len(result.scored_articles) == 1
    assert result.scored_articles[0].title == "ok"


def test_llm_chat_parse_channels_to_direction(monkeypatch):
    """验证：llm_chat 可从 channels 聚合得到方向。"""
    cfg = {
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "channel_weights": {
            "growth_impulse": 0.30,
            "inflation_pressure": -0.25,
            "policy_path": 0.25,
            "risk_premium": 0.20,
            "cost_shock": 0.20,
            "financial_conditions": 0.20,
        },
        "direction_threshold": {"positive": 0.20, "negative": -0.20},
        "low_confidence_to_neutral": 0.55,
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"market_scope":"macro","confidence":0.9,"channels":'
                                '{"growth_impulse":0,"inflation_pressure":1,"policy_path":0,"risk_premium":-1,"cost_shock":-1,"financial_conditions":0}}'
                            ),
                        }
                    }
                ]
            }

    monkeypatch.setattr("news_trends.scoring.requests.post", lambda *args, **kwargs: _Resp())
    dirs = detect_directions_llm(["Middle East conflict drives oil higher"], cfg)
    assert dirs == [-1]


def test_llm_chat_low_confidence_channels_to_neutral(monkeypatch):
    """验证：低置信度时通道结果强制归为中性。"""
    cfg = {
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "low_confidence_to_neutral": 0.80,
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"market_scope":"macro","confidence":0.5,"channels":'
                                '{"growth_impulse":1,"inflation_pressure":-1,"policy_path":1,"risk_premium":1,"cost_shock":1,"financial_conditions":1}}'
                            ),
                        }
                    }
                ]
            }

    monkeypatch.setattr("news_trends.scoring.requests.post", lambda *args, **kwargs: _Resp())
    dirs = detect_directions_llm(["headline"], cfg)
    assert dirs == [0]


def test_llm_input_max_chars_truncates_user_text(monkeypatch):
    """验证：发送给 LLM 的文本会按 input_max_chars 裁剪。"""
    cfg = {
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "input_max_chars": 20,
    }
    seen = {"user_content": ""}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"direction":"neutral","market_scope":"macro"}'}}]}

    def _fake_post(*args, **kwargs):
        seen["user_content"] = kwargs["json"]["messages"][1]["content"]
        return _Resp()

    monkeypatch.setattr("news_trends.scoring.requests.post", _fake_post)
    detect_directions_llm(["abcdefghijklmnopqrstuvwxyz"], cfg)
    assert "abcdefghijklmnopq..." in seen["user_content"]


def test_conflict_energy_guardrail_downgrades_positive(monkeypatch):
    """验证：冲突+能源涨价语义下，guardrail 会把 positive 降级。"""
    cfg = {
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "guardrail_conflict_energy_enabled": True,
        "guardrail_conflict_energy_positive_to": "neutral",
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"direction":"positive","market_scope":"macro"}'}}]}

    monkeypatch.setattr("news_trends.scoring.requests.post", lambda *args, **kwargs: _Resp())
    dirs = detect_directions_llm(["U.S. gas prices jump as Middle East conflict drives oil higher"], cfg)
    assert dirs == [0]


def test_neutral_risk_event_guardrail_downgrades_to_negative(monkeypatch):
    """验证：风险事件语义下，中性会被 guardrail 下调为负面。"""
    cfg = {
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "guardrail_neutral_risk_event_enabled": True,
        "guardrail_neutral_risk_event_to": "negative",
        "guardrail_neutral_risk_event_min_hits": 2,
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"direction":"neutral","market_scope":"macro"}'}}]}

    monkeypatch.setattr("news_trends.scoring.requests.post", lambda *args, **kwargs: _Resp())
    dirs = detect_directions_llm(
        ["Blue Owl private-credit stumble revives fears of another Bear Stearns moment"],
        cfg,
    )
    assert dirs == [-1]


def test_neutral_risk_event_guardrail_respects_min_hits(monkeypatch):
    """验证：命中不足阈值时，不触发中性->负面 guardrail。"""
    cfg = {
        "server_url": "http://127.0.0.1:1234",
        "model": "qwen2.5-7b-instruct",
        "max_retries": 0,
        "guardrail_neutral_risk_event_enabled": True,
        "guardrail_neutral_risk_event_min_hits": 3,
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"direction":"neutral","market_scope":"macro"}'}}]}

    monkeypatch.setattr("news_trends.scoring.requests.post", lambda *args, **kwargs: _Resp())
    dirs = detect_directions_llm(["Credit markets show fears"], cfg)
    assert dirs == [0]


def test_macro_scope_weight_higher_than_asset_scope():
    """验证：同方向同影响下，宏观新闻 article_score 权重高于个别资产。"""
    cfg = load_scoring_config()
    cfg.setdefault("sentiment", {})
    cfg["sentiment"]["backend"] = "rule"
    cfg["sentiment"]["macro_scope_terms"] = ["federal reserve", "inflation", "s&p 500"]
    cfg["sentiment"]["asset_scope_terms"] = ["earnings", "company", "shares"]
    cfg["rules"]["macro_scope_weight"] = 1.3
    cfg["rules"]["asset_scope_weight"] = 1.0

    # 两条新闻都偏负面；第一条偏宏观，第二条偏个股
    articles = [
        Article(
            source="macro",
            source_category="finance",
            title="Federal Reserve warns inflation surge",
            url="https://example.com/macro",
            published_at=None,
            summary="S&P 500 risk-off",
            content="inflation surge may pressure yields and equities",
        ),
        Article(
            source="asset",
            source_category="finance",
            title="Company issues profit warning",
            url="https://example.com/asset",
            published_at=None,
            summary="shares fall after earnings miss",
            content="company earnings disappoint and guidance cut",
        ),
    ]

    result = score_articles("2026-02-18", articles, cfg)
    macro = result.scored_articles[0]
    asset = result.scored_articles[1]
    assert macro.market_scope == "macro"
    assert asset.market_scope == "asset_specific"
    assert abs(macro.article_score) > abs(asset.article_score)


def test_same_event_cluster_decay_reduces_repeated_headline_scores():
    """验证：同事件近似标题会触发边际递减，后续 article_score 更低。"""
    cfg = load_scoring_config()
    cfg.setdefault("sentiment", {})
    cfg["sentiment"]["backend"] = "rule"
    cfg["rules"]["same_topic_frequency_boost"] = 0.0
    cfg["rules"]["same_event_decay_base"] = 0.7
    cfg["rules"]["same_event_decay_min_factor"] = 0.35
    cfg["rules"]["event_title_similarity_threshold"] = 0.8
    cfg["rules"]["event_token_jaccard_threshold"] = 0.45

    articles = [
        Article(
            source="a",
            source_category="finance",
            title="Treasury yields rise as Fed signals rate cut path",
            url="https://example.com/a",
            published_at=None,
            summary="stock market reacts to treasury yield move",
            content="rate cut and cooling inflation support risk sentiment",
        ),
        Article(
            source="b",
            source_category="finance",
            title="Treasury yields move higher as Fed signals rate cut path",
            url="https://example.com/b",
            published_at=None,
            summary="stock market reacts to treasury yield move",
            content="rate cut and cooling inflation support risk sentiment",
        ),
        Article(
            source="c",
            source_category="finance",
            title="Treasury yields climb further on Fed rate cut expectations",
            url="https://example.com/c",
            published_at=None,
            summary="stock market reacts to treasury yield move",
            content="rate cut and cooling inflation support risk sentiment",
        ),
    ]

    result = score_articles("2026-02-18", articles, cfg)
    scores = [x.article_score for x in result.scored_articles]
    assert len(scores) == 3
    assert scores[0] > scores[1] > scores[2]
