"""文章去重模块。"""

from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from typing import Any

import requests

from .models import Article


# 多空格压缩
SPACE_RE = re.compile(r"\s+")
# 非字母数字中文字符（保留英文、数字、中文）
NON_WORD_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff ]", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
TITLE_SPLIT_RE = re.compile(r"\s(?:-+|[|–—])\s")
LIKELY_PUBLISHER_TOKEN_RE = re.compile(r"^[a-z&.]+$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """标准化文本：转小写、去特殊字符、压缩空格。"""
    text = text.lower().strip()
    text = NON_WORD_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text


def _extract_json_like(text: str) -> Any:
    """从模型文本中提取 JSON（兼容 markdown 代码块）。"""
    raw = text.strip()
    match = _JSON_FENCE_RE.search(raw)
    if match:
        raw = match.group(1).strip()
    return json.loads(raw)


def _parse_duplicate_pair_ids_from_llm(content: str, total_pairs: int) -> list[int] | None:
    """解析候选对去重响应，支持 duplicate_pair_ids。"""
    if not content or not content.strip():
        return None
    try:
        parsed = _extract_json_like(content)
    except Exception:
        return None

    def _clean_ids(raw: Any) -> list[int]:
        if not isinstance(raw, list):
            return []
        ids: list[int] = []
        for item in raw:
            if isinstance(item, int) and 0 <= item < total_pairs:
                ids.append(item)
        return sorted(set(ids))

    if isinstance(parsed, dict):
        ids = _clean_ids(parsed.get("duplicate_pair_ids"))
        if ids:
            return ids
    if isinstance(parsed, list):
        ids = _clean_ids(parsed)
        if ids:
            return ids
    return []


def _tokenize_for_dedup(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {tok for tok in normalized.split(" ") if tok}


def _normalize_title_core_for_dedup(title: str) -> str:
    """提取标题核心文本：剥离常见「- Publisher」尾部来源片段。"""
    raw = " ".join((title or "").strip().split())
    if not raw:
        return ""

    parts = TITLE_SPLIT_RE.split(raw)
    if len(parts) >= 2 and _looks_like_publisher_segment(parts[-1]):
        raw = " - ".join(parts[:-1]).strip()
    return normalize_text(raw)


def _looks_like_publisher_segment(segment: str) -> bool:
    """判断标题尾段是否更像媒体来源名，而非正文语义的一部分。"""
    normalized = normalize_text(segment)
    tokens = [tok for tok in normalized.split(" ") if tok]
    if not (1 <= len(tokens) <= 6):
        return False
    if NUMBER_RE.search(normalized):
        return False
    # 来源名通常是短词组，几乎全由字母/符号词构成。
    if not all(LIKELY_PUBLISHER_TOKEN_RE.match(tok) for tok in tokens):
        return False
    return True


def _truncate_title_for_llm(title: str, max_chars: int) -> str:
    """控制传给 LLM 的标题长度，避免请求 token 过大。"""
    if max_chars <= 0 or len(title) <= max_chars:
        return title
    return title[:max_chars].rstrip() + "..."


def _require_prompt_config(sentiment_cfg: dict[str, Any], key: str) -> str:
    """强制从配置读取 prompt，缺失时抛出配置错误。"""
    value = sentiment_cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"缺少配置项 sentiment.{key}，请在 config/scoring.yaml 中设置。")
    return value


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return inter / union


def _build_candidate_pairs(titles: list[str], cfg: dict[str, Any] | None = None) -> list[tuple[int, int]]:
    """规则阶段：筛出值得交给 LLM 判断的标题候选对。"""
    sentiment_cfg = (cfg or {}).get("sentiment") or {}
    jaccard_threshold = float(sentiment_cfg.get("dedup_candidate_jaccard_threshold", 0.30))
    min_overlap_tokens = max(2, int(sentiment_cfg.get("dedup_candidate_min_overlap_tokens", 3)))
    substring_min_tokens = max(2, int(sentiment_cfg.get("dedup_candidate_substring_min_tokens", 3)))

    normalized = [normalize_text(t) for t in titles]
    normalized_cores = [_normalize_title_core_for_dedup(t) for t in titles]
    tokenized = [_tokenize_for_dedup(t) for t in titles]
    tokenized_cores = [_tokenize_for_dedup(t) for t in normalized_cores]
    numbers = [set(NUMBER_RE.findall(x)) for x in normalized]
    pair_scores: dict[tuple[int, int], float] = {}

    def register_pair(i: int, j: int, score: float) -> None:
        key = (i, j)
        prev = pair_scores.get(key)
        if prev is None or score > prev:
            pair_scores[key] = score

    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            text_i = normalized[i]
            text_j = normalized[j]
            tokens_i = tokenized[i]
            tokens_j = tokenized[j]
            overlap = tokens_i & tokens_j
            overlap_count = len(overlap)
            jaccard = _jaccard_similarity(tokens_i, tokens_j)
            core_i = normalized_cores[i]
            core_j = normalized_cores[j]
            core_tokens_i = tokenized_cores[i]
            core_tokens_j = tokenized_cores[j]
            core_jaccard = _jaccard_similarity(core_tokens_i, core_tokens_j)

            if text_i == text_j:
                register_pair(i, j, 1.0)
                continue

            if core_i and core_i == core_j:
                register_pair(i, j, 0.99)
                continue

            if (text_i in text_j or text_j in text_i) and min(len(tokens_i), len(tokens_j)) >= substring_min_tokens:
                register_pair(i, j, 0.95)
                continue

            if core_i and core_j and (core_i in core_j or core_j in core_i) and min(
                len(core_tokens_i), len(core_tokens_j)
            ) >= substring_min_tokens:
                register_pair(i, j, 0.93)
                continue

            if min(len(tokens_i), len(tokens_j)) >= substring_min_tokens and (
                tokens_i <= tokens_j or tokens_j <= tokens_i
            ):
                register_pair(i, j, 0.90 + min(0.05, jaccard * 0.05))
                continue

            if min(len(core_tokens_i), len(core_tokens_j)) >= substring_min_tokens and (
                core_tokens_i <= core_tokens_j or core_tokens_j <= core_tokens_i
            ):
                register_pair(i, j, 0.88 + min(0.05, core_jaccard * 0.05))
                continue

            if overlap_count >= min_overlap_tokens and jaccard >= jaccard_threshold:
                register_pair(i, j, min(0.87, 0.60 + jaccard * 0.30))
                continue

            # 标题共享数字锚点且文字高度重叠，容易是同一事件不同改写
            if numbers[i] and numbers[i] == numbers[j] and overlap_count >= (min_overlap_tokens + 1):
                register_pair(i, j, min(0.75, 0.55 + jaccard * 0.20))

    ranked = sorted(pair_scores.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [pair for pair, _score in ranked]


def _select_cluster_representative(cluster: list[int], titles: list[str]) -> int:
    """从重复簇中选择保留标题：优先信息量更高，其次 id 更小。"""
    def score(idx: int) -> tuple[int, int, int]:
        title = titles[idx]
        tokens = _tokenize_for_dedup(title)
        has_num = 1 if NUMBER_RE.search(title) else 0
        # tokens 多、包含数字通常更具体
        return (len(tokens), has_num, -idx)

    return max(cluster, key=score)


def _build_keep_ids_from_duplicate_pairs(titles: list[str], duplicate_pair_ids: list[int], pairs: list[tuple[int, int]]) -> list[int]:
    """根据重复 pair 的并查集聚类，构造保留索引。"""
    n = len(titles)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa = find(a)
        pb = find(b)
        if pa != pb:
            parent[pb] = pa

    for pair_id in duplicate_pair_ids:
        i, j = pairs[pair_id]
        union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    keep: list[int] = []
    for cluster in clusters.values():
        if len(cluster) == 1:
            keep.append(cluster[0])
        else:
            keep.append(_select_cluster_representative(cluster, titles))
    return sorted(set(keep))


def _is_obvious_duplicate_pair(title_a: str, title_b: str) -> bool:
    """本地快速判定明显重复（完全相同/仅附带来源后缀）。"""
    a = normalize_text(title_a)
    b = normalize_text(title_b)
    a_core = _normalize_title_core_for_dedup(title_a)
    b_core = _normalize_title_core_for_dedup(title_b)
    if not a or not b:
        return False
    if a == b:
        return True
    if a_core and a_core == b_core:
        return True

    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if short not in long:
        short, long = (a_core, b_core) if len(a_core) <= len(b_core) else (b_core, a_core)
        if not short or short not in long:
            short_tokens = _tokenize_for_dedup(a_core)
            long_tokens = _tokenize_for_dedup(b_core)
            if not short_tokens or not long_tokens:
                return False

            nums_a = set(NUMBER_RE.findall(a_core))
            nums_b = set(NUMBER_RE.findall(b_core))
            if nums_a and nums_b and nums_a != nums_b:
                return False

            # 两边都不是子串关系时，用高阈值“近乎同一句”做兜底。
            overlap = short_tokens & long_tokens
            jaccard = _jaccard_similarity(short_tokens, long_tokens)
            ratio = SequenceMatcher(None, a_core, b_core).ratio()
            if len(overlap) >= 5 and jaccard >= 0.75 and ratio >= 0.92:
                return True
            return False

    short_tokens = _tokenize_for_dedup(short)
    long_tokens = _tokenize_for_dedup(long)
    if not short_tokens:
        return False

    # 仅在长标题比短标题多很少 token 时，视为“附带来源后缀”
    if short_tokens <= long_tokens and (len(long_tokens) - len(short_tokens)) <= 7:
        return True
    return False


def _deduplicate_keep_indices_two_stage_with_llm(titles: list[str], cfg: dict[str, Any] | None = None) -> list[int] | None:
    """两阶段去重：规则筛候选对 + LLM 判定候选对。"""
    if not titles:
        return []

    sentiment_cfg = (cfg or {}).get("sentiment") or {}
    if sentiment_cfg.get("dedup_llm_enabled", True) is False:
        return None

    server_url = str(sentiment_cfg.get("server_url", "")).rstrip("/")
    model_id = str(sentiment_cfg.get("model", ""))
    if not server_url or not model_id:
        return None

    pairs = _build_candidate_pairs(titles, cfg)
    obvious_duplicate_pair_ids = [
        pair_id for pair_id, (i, j) in enumerate(pairs) if _is_obvious_duplicate_pair(titles[i], titles[j])
    ]
    obvious_set = set(obvious_duplicate_pair_ids)
    llm_pair_global_ids = [pair_id for pair_id in range(len(pairs)) if pair_id not in obvious_set]
    llm_pairs = [pairs[pair_id] for pair_id in llm_pair_global_ids]

    if not pairs:
        return list(range(len(titles)))
    if not llm_pairs:
        return _build_keep_ids_from_duplicate_pairs(titles, obvious_duplicate_pair_ids, pairs)

    timeout = float(sentiment_cfg.get("timeout_seconds", 10))
    max_retries = max(0, int(sentiment_cfg.get("max_retries", 2)))
    retry_backoff_seconds = max(0.0, float(sentiment_cfg.get("retry_backoff_seconds", 0.3)))
    temperature = float(sentiment_cfg.get("llm_temperature", 0))
    max_tokens = max(1, int(sentiment_cfg.get("llm_max_tokens", 4096)))
    max_candidate_pairs = max(1, int(sentiment_cfg.get("dedup_max_candidate_pairs", 220)))
    max_title_chars = max(30, int(sentiment_cfg.get("dedup_title_max_chars", 180)))

    limited_pair_global_ids = llm_pair_global_ids[:max_candidate_pairs]
    limited_pairs = [pairs[pair_id] for pair_id in limited_pair_global_ids]
    indexed_pairs = [
        {
            "pair_id": pair_id,
            "a_id": i,
            "b_id": j,
            "a_title": _truncate_title_for_llm(titles[i], max_title_chars),
            "b_title": _truncate_title_for_llm(titles[j], max_title_chars),
        }
        for pair_id, (i, j) in enumerate(limited_pairs)
    ]
    pairs_json = json.dumps(indexed_pairs, ensure_ascii=False)

    system_prompt = _require_prompt_config(sentiment_cfg, "dedup_pair_llm_system_prompt")
    user_template = _require_prompt_config(sentiment_cfg, "dedup_pair_llm_user_template")
    user_prompt = user_template.replace("{candidate_pairs_json}", pairs_json)

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
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            break
        except Exception:
            if attempt >= max_retries:
                if obvious_duplicate_pair_ids:
                    return _build_keep_ids_from_duplicate_pairs(titles, obvious_duplicate_pair_ids, pairs)
                return None
            time.sleep(retry_backoff_seconds * (2**attempt))

    if resp is None:
        return None

    try:
        payload = resp.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            return None

        duplicate_pair_ids = _parse_duplicate_pair_ids_from_llm(content, len(indexed_pairs))
        if duplicate_pair_ids is None:
            if obvious_duplicate_pair_ids:
                return _build_keep_ids_from_duplicate_pairs(titles, obvious_duplicate_pair_ids, pairs)
            return None

        llm_duplicate_global_pairs = [limited_pair_global_ids[idx] for idx in duplicate_pair_ids]
        merged_duplicate_pair_ids = sorted(set(obvious_duplicate_pair_ids + llm_duplicate_global_pairs))
        keep_ids = _build_keep_ids_from_duplicate_pairs(titles, merged_duplicate_pair_ids, pairs)
        return keep_ids
    except Exception:
        if obvious_duplicate_pair_ids:
            return _build_keep_ids_from_duplicate_pairs(titles, obvious_duplicate_pair_ids, pairs)
        return None


def deduplicate_articles(articles: list[Article], cfg: dict[str, Any] | None = None) -> list[Article]:
    """去重：仅使用候选对 + dedup_pair_llm 判定重复。"""
    if not articles:
        return []

    titles = [a.title for a in articles]
    base_keep_indices = _deduplicate_keep_indices_two_stage_with_llm(titles, cfg)
    if base_keep_indices is None:
        base_keep_indices = list(range(len(articles)))
    return [articles[i] for i in base_keep_indices]
