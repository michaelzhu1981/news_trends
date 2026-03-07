"""管道模块测试：日期过滤与文件清理逻辑。"""

from datetime import UTC, date, datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_trends.models import Article
from news_trends.pipeline import _cleanup_recent_files, _filter_articles_by_target_date, _latest_dated_file


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


def test_cleanup_recent_files_keeps_only_recent_three_days(tmp_path):
    """验证：仅删除超过 3 天窗口的日期文件，不影响非日期文件。"""
    keep_names = [
        "2026-03-05_scored.json",
        "2026-03-06_scored.json",
        "2026-03-07_scored.json",
        "README.txt",
    ]
    remove_names = ["2026-03-04_scored.json", "2026-03-01_scored.json"]

    for name in keep_names + remove_names:
        (tmp_path / name).write_text("x", encoding="utf-8")

    removed = _cleanup_recent_files(tmp_path, keep_days=3, anchor_date=date(2026, 3, 7))

    assert removed == 2
    assert (tmp_path / "2026-03-05_scored.json").exists()
    assert (tmp_path / "2026-03-06_scored.json").exists()
    assert (tmp_path / "2026-03-07_scored.json").exists()
    assert (tmp_path / "README.txt").exists()
    assert not (tmp_path / "2026-03-04_scored.json").exists()
    assert not (tmp_path / "2026-03-01_scored.json").exists()


def test_latest_dated_file_ignores_non_date_files(tmp_path):
    """验证：最新日期识别只基于日期前缀文件。"""
    (tmp_path / "2026-03-03_scored.json").write_text("x", encoding="utf-8")
    (tmp_path / "2026-03-07.md").write_text("x", encoding="utf-8")
    (tmp_path / "not-a-date.txt").write_text("x", encoding="utf-8")

    latest = _latest_dated_file(tmp_path)

    assert latest == date(2026, 3, 7)
