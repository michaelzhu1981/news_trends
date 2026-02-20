"""管道模块测试：日期过滤逻辑。"""

from datetime import UTC, datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_trends.models import Article
from news_trends.pipeline import _filter_articles_by_target_date


def _mk_article(ts: datetime | None) -> Article:
    """构造测试用 Article。"""
    return Article(
        source="test",
        source_category="general",
        title="t",
        url="https://example.com",
        published_at=ts,
        summary="",
        content="",
    )


def test_filter_articles_by_target_date_uses_local_timezone():
    """验证：仅保留本地时区下发表日期为今日的文章；无时间戳的排除。"""
    local_tz = ZoneInfo("Asia/Shanghai")
    in_day = _mk_article(datetime.now(UTC))
    out_day = _mk_article(datetime(2020, 1, 1, tzinfo=UTC))
    missing_time = _mk_article(None)
    target = datetime.now(local_tz).strftime("%Y-%m-%d")

    filtered = _filter_articles_by_target_date([in_day, out_day, missing_time], target, local_tz)

    assert len(filtered) == 1
    assert filtered[0].url == in_day.url


def test_filter_articles_by_target_date_treats_naive_as_utc():
    """验证：无时区时间的 datetime 被视为 UTC 处理。"""
    local_tz = ZoneInfo("Asia/Shanghai")
    aware_dt = datetime(2026, 2, 17, 12, 0, 0, tzinfo=UTC)
    target = aware_dt.astimezone(local_tz).strftime("%Y-%m-%d")
    naive_utc = _mk_article(datetime(2026, 2, 17, 12, 0, 0))
    aware_utc = _mk_article(aware_dt)

    filtered = _filter_articles_by_target_date([naive_utc, aware_utc], target, local_tz)

    assert len(filtered) == 2
