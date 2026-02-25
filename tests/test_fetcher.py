"""fetcher 模块测试。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_trends import fetcher


def test_fetch_full_text_truncates_abnormal_long_content(monkeypatch):
    class _Resp:
        text = "<html>fake</html>"

        def raise_for_status(self):
            return None

    long_text = "a" * (fetcher.MAX_CONTENT_CHARS + 100)
    monkeypatch.setattr("news_trends.fetcher.requests.get", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr("news_trends.fetcher.trafilatura.extract", lambda *args, **kwargs: long_text)

    content = fetcher.fetch_full_text("https://example.com")

    assert len(content) == fetcher.MAX_CONTENT_CHARS
    assert content.endswith("...")


def test_extract_from_html_keeps_normal_length_content(monkeypatch):
    text = "normal content"
    monkeypatch.setattr("news_trends.fetcher.trafilatura.extract", lambda *args, **kwargs: text)

    content = fetcher._extract_from_html("<html>fake</html>")

    assert content == text
