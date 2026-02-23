"""新闻打分与宏观情绪计算模块。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import requests

from .dedup import normalize_text
from .models import Article, DailyResult, ScoredArticle

# 方向对应的中文标签
DIRECTION_LABEL = {
    -1: "负面",
    0: "中性",
    1: "正面",
}
MARKET_SCOPE_LABEL = {
    "macro": "宏观大盘",
    "asset_specific": "个股/个别资产",
}

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "us",
    "with",
}


def _contains_any(text: str, terms: list[str]) -> int:
    """统计文本命中的关键词个数（按词边界/短语匹配，避免子串误命中）。"""
    text_tokens = [tok for tok in text.split() if tok]
    if not text_tokens:
        return 0

    hits = 0
    for term in terms:
        term_tokens = [tok for tok in term.split() if tok]
        if not term_tokens:
            continue
        span = len(term_tokens)
        if span > len(text_tokens):
            continue
        matched = any(
            text_tokens[idx : idx + span] == term_tokens for idx in range(len(text_tokens) - span + 1)
        )
        if matched:
            hits += 1
    return hits


def detect_topic(text: str, topic_keywords: dict[str, list[str]]) -> str:
    """根据关键词命中数检测主题，命中最多者胜出；无命中则返回 general。"""
    topic_hits: dict[str, int] = {}
    for topic, keywords in topic_keywords.items():
        topic_hits[topic] = _contains_any(text, keywords)

    ranked = sorted(topic_hits.items(), key=lambda x: x[1], reverse=True)
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]
    return "general"


def detect_direction_rule(text: str, positive_terms: list[str], negative_terms: list[str], neutral_band: int = 1) -> int:
    """规则法检测情绪方向：正负面命中数差值在 neutral_band 内为中性，否则取较多一方。返回 -1/0/1。"""
    pos_hits = _contains_any(text, positive_terms)
    neg_hits = _contains_any(text, negative_terms)

    if abs(pos_hits - neg_hits) <= neutral_band:
        return 0
    return 1 if pos_hits > neg_hits else -1


def _label_to_direction(label: str, mapping: dict[str, int]) -> int:
    """把模型标签映射到 -1/0/1。"""
    key = label.strip().lower()
    if key in mapping:
        return mapping[key]

    if "neg" in key:
        return -1
    if "pos" in key:
        return 1
    if "neu" in key:
        return 0
    if key.endswith("0"):
        return -1
    if key.endswith("1"):
        return 0
    if key.endswith("2"):
        return 1
    return 0


def _extract_json_like(text: str) -> Any:
    """从模型文本中提取 JSON（兼容 markdown 代码块）。"""
    raw = text.strip()
    match = _JSON_FENCE_RE.search(raw)
    if match:
        raw = match.group(1).strip()
    return json.loads(raw)


def _parse_llm_direction_response(content: str, direction_mapping: dict[str, int]) -> int:
    """解析 LLM 返回文本到 -1/0/1。"""
    if not content or not content.strip():
        raise ValueError("llm 返回为空")

    parsed: Any
    try:
        parsed = _extract_json_like(content)
    except Exception:
        parsed = content.strip()

    if isinstance(parsed, dict):
        for key in ("direction", "label", "sentiment"):
            value = parsed.get(key)
            if isinstance(value, str):
                return _label_to_direction(value, direction_mapping)
            if isinstance(value, (int, float)):
                num = int(value)
                if num in (-1, 0, 1):
                    return num
        raise ValueError("llm 返回字典缺少可识别字段")

    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, (int, float)):
            num = int(first)
            if num in (-1, 0, 1):
                return num
        if isinstance(first, str):
            return _label_to_direction(first, direction_mapping)
        if isinstance(first, dict):
            return _parse_llm_direction_response(json.dumps(first, ensure_ascii=True), direction_mapping)
        raise ValueError("llm 返回数组格式不可识别")

    if isinstance(parsed, (int, float)):
        num = int(parsed)
        if num in (-1, 0, 1):
            return num

    if isinstance(parsed, str):
        stripped = parsed.strip()
        if stripped in {"-1", "0", "1"}:
            return int(stripped)
        return _label_to_direction(stripped, direction_mapping)

    raise ValueError("llm 返回格式不可识别")


def _label_to_scope(label: str, mapping: dict[str, str]) -> str:
    """把模型标签映射到 macro / asset_specific。"""
    key = label.strip().lower()
    if key in mapping:
        mapped = mapping[key].strip().lower()
        if mapped in MARKET_SCOPE_LABEL:
            return mapped

    if any(k in key for k in ("macro", "broad market", "index-level", "market-wide")):
        return "macro"
    if any(k in key for k in ("asset", "single stock", "single-asset", "idiosyncratic", "company")):
        return "asset_specific"
    return "asset_specific"


def _parse_llm_scope_response(content: str, scope_mapping: dict[str, str]) -> str:
    """解析 LLM 返回文本到 macro / asset_specific。"""
    if not content or not content.strip():
        raise ValueError("llm 返回为空")

    parsed: Any
    try:
        parsed = _extract_json_like(content)
    except Exception:
        parsed = content.strip()

    if isinstance(parsed, dict):
        for key in ("market_scope", "scope", "scope_label"):
            value = parsed.get(key)
            if isinstance(value, str):
                return _label_to_scope(value, scope_mapping)
        raise ValueError("llm 返回字典缺少可识别 scope 字段")

    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, str):
            return _label_to_scope(first, scope_mapping)
        if isinstance(first, dict):
            return _parse_llm_scope_response(json.dumps(first, ensure_ascii=True), scope_mapping)
        raise ValueError("llm 返回数组格式不可识别")

    if isinstance(parsed, str):
        return _label_to_scope(parsed.strip(), scope_mapping)

    raise ValueError("llm 返回格式不可识别")


def _detect_direction_and_scope_llm(texts: list[str], sentiment_cfg: dict[str, Any]) -> list[tuple[int, str]]:
    """通过 LM Studio OpenAI 兼容 chat/completions 接口检测方向与新闻范围。"""
    if not texts:
        return []

    server_url = str(sentiment_cfg.get("server_url", "")).rstrip("/")
    model_id = str(sentiment_cfg.get("model", ""))
    timeout = float(sentiment_cfg.get("timeout_seconds", 10))
    max_retries = max(0, int(sentiment_cfg.get("max_retries", 2)))
    retry_backoff_seconds = max(0.0, float(sentiment_cfg.get("retry_backoff_seconds", 0.3)))
    temperature = float(sentiment_cfg.get("llm_temperature", 0))
    max_tokens = max(1, int(sentiment_cfg.get("llm_max_tokens", 16)))

    direction_mapping = {
        str(k).lower(): int(v)
        for k, v in sentiment_cfg.get(
            "direction_mapping",
            {
                "negative": -1,
                "neutral": 0,
                "positive": 1,
                "label_0": -1,
                "label_1": 0,
                "label_2": 1,
            },
        ).items()
    }
    scope_mapping = {
        str(k).lower(): str(v).lower()
        for k, v in sentiment_cfg.get(
            "scope_mapping",
            {
                "macro": "macro",
                "broad_market": "macro",
                "market_wide": "macro",
                "asset_specific": "asset_specific",
                "single_asset": "asset_specific",
                "single_stock": "asset_specific",
            },
        ).items()
    }
    system_prompt = str(
        sentiment_cfg.get(
            "llm_system_prompt",
            (
                "You are a financial news classifier for short-term market trend scoring. "
                "Return two labels: direction and market_scope. "
                "direction must be one of negative|neutral|positive. "
                "market_scope must be one of macro|asset_specific."
            ),
        )
    )
    user_template = str(
        sentiment_cfg.get(
            "llm_user_template",
            "Classify the following market news text. Respond with JSON only: {\"direction\":\"negative|neutral|positive\",\"market_scope\":\"macro|asset_specific\"}.\nText:\n{text}",
        )
    )

    if not server_url or not model_id:
        raise ValueError("llm 配置缺少 server_url 或 model")
    if timeout <= 0:
        raise ValueError("llm 配置 timeout_seconds 必须大于 0")

    def _render_user_prompt(template: str, article_text: str) -> str:
        """仅替换 {text} 占位符，避免把 JSON 花括号误当成 format 变量。"""
        if "{text}" in template:
            return template.replace("{text}", article_text)
        return f"{template}\n{article_text}"

    results: list[tuple[int, str]] = []
    for text in texts:
        last_error: Exception | None = None
        resp = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    f"{server_url}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": _render_user_prompt(user_template, text)},
                        ],
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise RuntimeError(f"llm 请求失败，已重试 {max_retries} 次") from exc
                time.sleep(retry_backoff_seconds * (2**attempt))

        if resp is None:
            raise RuntimeError("llm 请求失败：未获得响应") from last_error

        payload = resp.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("llm 返回异常：choices 为空")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("llm 返回异常：缺少 message.content")
        direction = _parse_llm_direction_response(content, direction_mapping)
        try:
            market_scope = _parse_llm_scope_response(content, scope_mapping)
        except Exception:
            market_scope = detect_market_scope_rule(text, sentiment_cfg)
        results.append((direction, market_scope))

    return results


def detect_directions_llm(texts: list[str], sentiment_cfg: dict[str, Any]) -> list[int]:
    """通过 LM Studio OpenAI 兼容 chat/completions 接口检测方向。"""
    return [direction for direction, _ in _detect_direction_and_scope_llm(texts, sentiment_cfg)]


def detect_market_scope_rule(text: str, cfg: dict[str, Any]) -> str:
    """规则法检测新闻范围：macro 或 asset_specific。"""
    macro_terms = [str(x).lower() for x in cfg.get("macro_scope_terms", [])]
    asset_terms = [str(x).lower() for x in cfg.get("asset_scope_terms", [])]

    macro_hits = _contains_any(text, macro_terms)
    asset_hits = _contains_any(text, asset_terms)
    has_dollar_ticker = bool(re.search(r"(?:^|\s)\$[a-z]{1,6}\b", text))

    if macro_hits > asset_hits:
        return "macro"
    if asset_hits > macro_hits or has_dollar_ticker:
        return "asset_specific"
    return "asset_specific"


def detect_directions(
    texts: list[str],
    positive_terms: list[str],
    negative_terms: list[str],
    rules: dict[str, Any],
    cfg: dict[str, Any],
) -> list[int]:
    """统一方向检测入口：支持 rule / llm_chat。"""
    sentiment_cfg = cfg.get("sentiment", {})
    backend = str(sentiment_cfg.get("backend", "rule")).lower()

    if backend == "llm_chat":
        return detect_directions_llm(texts, sentiment_cfg)

    return [
        detect_direction_rule(
            text,
            positive_terms,
            negative_terms,
            neutral_band=rules.get("neutral_band", 1),
        )
        for text in texts
    ]


def detect_market_scopes(
    texts: list[str],
    cfg: dict[str, Any],
) -> list[str]:
    """统一范围检测入口：支持 llm_chat / rule。"""
    sentiment_cfg = cfg.get("sentiment", {})
    backend = str(sentiment_cfg.get("backend", "rule")).lower()
    if backend == "llm_chat":
        return [scope for _, scope in _detect_direction_and_scope_llm(texts, sentiment_cfg)]
    return [detect_market_scope_rule(text, sentiment_cfg) for text in texts]


def calc_relevance(text: str, relevance_terms: list[str], base: float, boost: float, max_score: float) -> float:
    """计算市场相关性得分。

    原理:
    - relevance = base + 命中的市场相关关键词数 * boost，并裁剪到 [0, max_score]。

    意义:
    - 用于回答“这条新闻是否真的和市场价格相关”，避免把泛资讯当作交易信号。
    - 分值越高，越说明该新闻对资产价格形成影响的可能性更高。
    """
    hits = _contains_any(text, relevance_terms)
    score = base + hits * boost
    return max(0.0, min(max_score, score))


def calc_impact(
    topic: str,
    text: str,
    topic_weights: dict[str, float],
    boost_terms: list[str],
    event_cluster_count_today: int,
    impact_boost_per_term: float,
    same_topic_frequency_boost: float,
    max_impact: float,
) -> float:
    """计算影响强度。

    原理:
    - impact = 主题基础权重
      + ln(1 + 冲击类词命中数) * impact_boost_per_term
      + max(0, 同事件簇当日频次 - 1) * same_topic_frequency_boost
    - 最终裁剪到 [0, max_impact]。

    意义:
    - 用于回答“这条新闻力度有多大”。
    - 区分“同样相关但冲击程度不同”的新闻；分值越高，潜在波动/风险偏好切换越强。
    """
    base = topic_weights.get(topic, topic_weights.get("general", 0.4))
    boost_hits = _contains_any(text, boost_terms)
    boosts = math.log1p(boost_hits) * impact_boost_per_term
    freq_boost = max(0, event_cluster_count_today - 1) * same_topic_frequency_boost
    return max(0.0, min(max_impact, base + boosts + freq_boost))


def _title_tokens(title: str) -> set[str]:
    """提取用于事件聚类的标题关键词。"""
    normalized = normalize_text(title)
    return {tok for tok in normalized.split() if len(tok) >= 3 and tok not in _TITLE_STOPWORDS}


def _leading_signature(title: str) -> tuple[str, ...]:
    """标题前两个有效 token，作为强锚点签名。"""
    normalized = normalize_text(title)
    tokens = [tok for tok in normalized.split() if len(tok) >= 3 and tok not in _TITLE_STOPWORDS]
    return tuple(tokens[:2])


def _token_jaccard(left: set[str], right: set[str]) -> float:
    """计算两个 token 集合的 Jaccard 相似度。"""
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return inter / union


def _build_event_cluster_ids(
    articles: list[Article],
    topics: list[str],
    *,
    title_similarity_threshold: float,
    token_jaccard_threshold: float,
) -> list[int]:
    """按标题近似度把新闻聚为事件簇，返回每条新闻的 cluster_id。"""
    cluster_ids: list[int] = []
    cluster_reps: list[tuple[str, set[str], tuple[str, ...]]] = []
    # rep: (normalized_title, title_tokens, leading_signature)

    for article, _topic in zip(articles, topics):
        norm_title = normalize_text(article.title)
        tokens = _title_tokens(article.title)
        leading_sig = _leading_signature(article.title)
        best_cluster = -1
        best_score = -1.0

        for cid, (rep_title, rep_tokens, rep_leading_sig) in enumerate(cluster_reps):
            token_sim = _token_jaccard(tokens, rep_tokens)
            title_sim = SequenceMatcher(None, norm_title, rep_title).ratio()
            same_leading = len(leading_sig) == 2 and leading_sig == rep_leading_sig
            if not same_leading and token_sim < token_jaccard_threshold and title_sim < title_similarity_threshold:
                continue
            score = 1.0 if same_leading else max(token_sim, title_sim)
            if score > best_score:
                best_score = score
                best_cluster = cid

        if best_cluster < 0:
            best_cluster = len(cluster_reps)
            cluster_reps.append((norm_title, tokens, leading_sig))

        cluster_ids.append(best_cluster)

    return cluster_ids


def classify_market_regime(score: float, regime_thresholds: list[dict[str, Any]]) -> str:
    """根据 TrendScore 划分市场 regime：区间由 rules.regime_thresholds（scoring.yaml）定义。

    解读方式:
    - 分数正负号代表方向:
      - score > 0: 偏 Risk-On（风险偏好上升，通常偏多）
      - score < 0: 偏 Risk-Off（风险偏好下降，通常偏空）
    - 分数绝对值代表强度:
      - |score| 越接近 0，情绪越中性或多空更分散
      - |score| 越大，说明同方向新闻越集中、影响越强

    按 min_score 从高到低匹配，最后一项可不写 min_score 表示兜底。
    """
    for item in regime_thresholds:
        min_score = item.get("min_score")
        if min_score is None:
            return str(item.get("label", ""))
        if score >= min_score:
            return str(item.get("label", ""))
    return str(regime_thresholds[-1].get("label", "")) if regime_thresholds else ""


def summarize_topics(scored_articles: list[ScoredArticle]) -> str:
    """汇总当日主导主题，取前二高频主题用「与」连接，如「inflation与geopolitics主导」。"""
    if not scored_articles:
        return "当日无有效新闻数据"

    top_topics = Counter(a.topic for a in scored_articles).most_common(2)
    names = "与".join(topic for topic, _ in top_topics)
    return f"{names}主导"


def score_articles(date_str: str, articles: list[Article], cfg: dict) -> DailyResult:
    """对文章列表进行主题检测、方向检测、相关性与影响打分，并汇总成当日 DailyResult。TrendScore 经 tanh 映射到 [-100, 100]。"""
    topic_weights = cfg["topic_weights"]
    topic_keywords = cfg["topic_keywords"]
    positive_terms = [normalize_text(x) for x in cfg["positive_terms"]]
    negative_terms = [normalize_text(x) for x in cfg["negative_terms"]]
    relevance_terms = [normalize_text(x) for x in cfg["market_relevance_terms"]]
    boost_terms = [normalize_text(x) for x in cfg["impact_boost_terms"]]
    rules = cfg["rules"]
    macro_scope_weight = float(rules.get("macro_scope_weight", 1.25))
    asset_scope_weight = float(rules.get("asset_scope_weight", 0.3))
    if macro_scope_weight <= 0 or asset_scope_weight <= 0:
        raise ValueError("scope 权重必须大于 0")
    same_event_decay_base = float(rules.get("same_event_decay_base", 1.0))
    same_event_decay_min_factor = float(rules.get("same_event_decay_min_factor", 0.35))
    event_title_similarity_threshold = float(rules.get("event_title_similarity_threshold", 0.8))
    event_token_jaccard_threshold = float(rules.get("event_token_jaccard_threshold", 0.45))
    if not (0 < same_event_decay_base <= 1):
        raise ValueError("rules.same_event_decay_base 必须在 (0, 1] 范围内")
    if not (0 < same_event_decay_min_factor <= 1):
        raise ValueError("rules.same_event_decay_min_factor 必须在 (0, 1] 范围内")
    if not (0 <= event_title_similarity_threshold <= 1):
        raise ValueError("rules.event_title_similarity_threshold 必须在 [0, 1] 范围内")
    if not (0 <= event_token_jaccard_threshold <= 1):
        raise ValueError("rules.event_token_jaccard_threshold 必须在 [0, 1] 范围内")
    scope_weights = {
        "macro": macro_scope_weight,
        "asset_specific": asset_scope_weight,
    }

    texts = [normalize_text(f"{a.title} {a.summary} {a.content}") for a in articles]
    sentiment_cfg = cfg.get("sentiment", {})
    backend = str(sentiment_cfg.get("backend", "rule")).lower()
    used_articles = articles
    used_texts = texts
    if backend == "llm_chat":
        # LLM 后端逐条处理：单条失败时跳过该新闻，不中断整日流程。
        used_articles = []
        used_texts = []
        direction_scope_pairs: list[tuple[int, str]] = []
        for article, text in zip(articles, texts):
            try:
                pair = _detect_direction_and_scope_llm([text], sentiment_cfg)[0]
            except Exception:
                continue
            used_articles.append(article)
            used_texts.append(text)
            direction_scope_pairs.append(pair)
        directions = [x[0] for x in direction_scope_pairs]
        market_scopes = [x[1] for x in direction_scope_pairs]
    else:
        directions = detect_directions(used_texts, positive_terms, negative_terms, rules, cfg)
        market_scopes = detect_market_scopes(used_texts, cfg)

    detected_topics = [detect_topic(text, topic_keywords) for text in used_texts]
    event_cluster_ids = _build_event_cluster_ids(
        used_articles,
        detected_topics,
        title_similarity_threshold=event_title_similarity_threshold,
        token_jaccard_threshold=event_token_jaccard_threshold,
    )
    event_cluster_size_counter = Counter(event_cluster_ids)
    event_cluster_counter: Counter[int] = Counter()

    scored: list[ScoredArticle] = []
    total_raw_score = 0.0

    for article, text, topic, direction, market_scope, cluster_id in zip(
        used_articles, used_texts, detected_topics, directions, market_scopes, event_cluster_ids
    ):
        relevance = calc_relevance(
            text,
            relevance_terms,
            base=rules["base_relevance"],
            boost=rules["relevance_hit_boost"],
            max_score=rules["max_relevance"],
        )
        impact = calc_impact(
            topic,
            text,
            topic_weights,
            boost_terms,
            event_cluster_size_counter[cluster_id],
            impact_boost_per_term=rules["impact_boost_per_term"],
            same_topic_frequency_boost=rules["same_topic_frequency_boost"],
            max_impact=rules["max_impact"],
        )
        # article_score 的原理与意义:
        # - 原理: direction(-1/0/1) * impact(力度) * relevance(市场相关性) * scope_weight(范围权重)。
        # - 意义: 给出单篇新闻对当日情绪的净贡献值。
        #   正值偏 Risk-On，负值偏 Risk-Off，绝对值越大说明贡献越强。
        scope_weight = scope_weights.get(market_scope, asset_scope_weight)
        base_score = direction * impact * relevance * scope_weight
        event_cluster_counter[cluster_id] += 1
        cluster_rank = event_cluster_counter[cluster_id]
        event_decay_factor = max(same_event_decay_min_factor, same_event_decay_base ** (cluster_rank - 1))
        article_score = base_score * event_decay_factor
        total_raw_score += article_score

        scored.append(
            ScoredArticle(
                source=article.source,
                source_category=article.source_category,
                title=article.title,
                url=article.url,
                published_at=article.published_at,
                summary=article.summary,
                content=article.content,
                topic=topic,
                direction=direction,
                relevance=round(relevance, 4),
                impact=round(impact, 4),
                market_scope=market_scope,
                market_scope_label=MARKET_SCOPE_LABEL.get(market_scope, market_scope),
                scope_weight=round(scope_weight, 4),
                article_score=round(article_score, 4),
                direction_label=DIRECTION_LABEL[direction],
                content_hash=hashlib.md5(text.encode("utf-8")).hexdigest(),
                title_hash=hashlib.md5(normalize_text(article.title).encode("utf-8")).hexdigest(),
            )
        )

    # TrendScore 映射与含义:
    # - 先把全部 article_score 求和得到 total_raw_score（净情绪贡献）。
    # - 再通过 tanh(k * raw_score) 非线性压缩到 [-1, 1]，并放大到 [-100, 100]。
    # - k 越小，越不容易出现接近 ±100 的饱和分数；k 越大，分数更敏感。
    # - 正值表示偏 Risk-On，负值表示偏 Risk-Off。
    # - 绝对值越大表示情绪强度越高；接近 0 表示中性或分歧更大。
    trend_score_tanh_k = float(rules.get("trend_score_tanh_k", 1.0))
    if trend_score_tanh_k <= 0:
        raise ValueError("rules.trend_score_tanh_k 必须大于 0")
    trend_score = round(100 * math.tanh(trend_score_tanh_k * total_raw_score), 2)

    return DailyResult(
        date=date_str,
        trend_score=trend_score,
        article_count=len(scored),
        summary=summarize_topics(scored),
        scored_articles=scored,
    )
