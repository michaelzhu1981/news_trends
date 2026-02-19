"""文章去重模块。"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher

from .models import Article


# 多空格压缩
SPACE_RE = re.compile(r"\s+")
# 非字母数字中文字符（保留英文、数字、中文）
NON_WORD_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff ]", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """标准化文本：转小写、去特殊字符、压缩空格。"""
    text = text.lower().strip()
    text = NON_WORD_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text


def content_hash(text: str) -> str:
    """对标准化后的文本生成 MD5 哈希，用于内容去重。"""
    norm = normalize_text(text)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def title_similar(a: str, b: str, threshold: float = 0.88) -> bool:
    """判断两个标题是否相似（SequenceMatcher 相似度 >= threshold）。"""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio() >= threshold


def deduplicate_articles(articles: list[Article]) -> list[Article]:
    """去重：内容哈希相同或标题高度相似的文章只保留一条。"""
    unique: list[Article] = []
    seen_content: set[str] = set()

    for art in articles:
        body = art.content if art.content else art.summary
        c_hash = content_hash(body)
        if c_hash in seen_content:
            continue

        duplicate_title = any(title_similar(art.title, existing.title) for existing in unique)
        if duplicate_title:
            continue

        seen_content.add(c_hash)
        unique.append(art)

    return unique
