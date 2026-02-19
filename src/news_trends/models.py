"""数据模型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    """原始文章数据。

    Attributes:
        source: 来源名称（如 RSS 源名）
        source_category: 来源分类（如 finance, geopolitics）
        title: 标题
        url: 链接
        published_at: 发布时间（可为空）
        summary: 摘要
        content: 正文内容
    """

    source: str
    source_category: str
    title: str
    url: str
    published_at: Optional[datetime]
    summary: str = ""
    content: str = ""


@dataclass
class ScoredArticle(Article):
    """带评分的文章，继承 Article 并扩展打分相关字段。

    Attributes:
        topic: 检测到的主题（如 general, inflation 等）
        direction: 情绪方向（-1 负面, 0 中性, 1 正面）
        relevance: 市场相关性得分
        impact: 影响强度得分
        market_scope: 新闻范围标签（macro 宏观大盘, asset_specific 个股/个别资产）
        market_scope_label: 新闻范围文字标签（宏观大盘/个股或个别资产）
        scope_weight: 范围权重（宏观通常高于个股）
        article_score: 单篇文章综合得分
        direction_label: 方向文字标签（负面/中性/正面）
        content_hash: 正文哈希（用于去重等）
        title_hash: 标题哈希
    """

    topic: str = "general"
    direction: int = 0
    relevance: float = 0.0
    impact: float = 0.0
    market_scope: str = "asset_specific"
    market_scope_label: str = "个股/个别资产"
    scope_weight: float = 1.0
    article_score: float = 0.0
    direction_label: str = "中性"
    content_hash: str = ""
    title_hash: str = ""


@dataclass
class DailyResult:
    """某一天的汇总结果。

    Attributes:
        date: 日期字符串（YYYY-MM-DD）
        trend_score: 当日 TrendScore（-100 ~ +100）
        article_count: 有效文章数量
        summary: 当日主题摘要
        scored_articles: 带评分的文章列表
    """

    date: str
    trend_score: float
    article_count: int
    summary: str
    scored_articles: list[ScoredArticle] = field(default_factory=list)
