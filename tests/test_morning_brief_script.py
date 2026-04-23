"""Unit tests for the ``morning_brief`` skill script.

Covers the Phase 8 argv-only skill convention:

* Each HTTP-facing section (quote, weather, GitHub, HN, TechCrunch, NZ news)
  has a happy path and a failure path that degrades to an inline ⚠️ banner
  rather than crashing the brief.
* ZenQuotes failure falls back to a deterministic local quote (random seeded
  so the test is repeatable).
* MetOcean weather is skipped with a polite notice when the API key is not
  set; temperatures are converted Kelvin → °C and wind m/s → km/h.
* NZ news failover: RNZ failure falls through to Stuff NZ.
* The RSS parser accepts both RSS 2.0 (``<item><title>...``) and Atom
  (``<entry><title>...`` with href in the ``link`` attribute).
* ``build_brief`` assembles frontmatter + a header + every section in order.
* ``_destination`` creates ``<vault_root>/Daily/YYYY/MM/DD-daily-news.md``.
* ``main`` writes the file, echoes to stdout, and exits 0. Validation errors
  on ``--date`` or missing ``--vault-root`` exit 2 without side effects.

Network is mocked with respx — zero real I/O.
"""
from __future__ import annotations

import importlib.util
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "runtime" / "skills" / "_bundle" / "morning_brief" / "morning_brief.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("morning_brief_bundle", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


morning_brief = _load_module()

pytestmark = pytest.mark.unit


NZDT = ZoneInfo("Pacific/Auckland")


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> Any:
    """Fresh sync httpx.Client with the module's production settings."""
    return httpx.Client(
        timeout=morning_brief.REQUEST_TIMEOUT,
        headers={"User-Agent": morning_brief.USER_AGENT},
    )


@pytest.fixture
def when_nz() -> datetime:
    # A known NZDT date so path construction is deterministic.
    return datetime(2026, 4, 19, 7, 30, tzinfo=NZDT)


@pytest.fixture(autouse=True)
def _clear_metoceankey(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default: no weather key. Individual tests set it when they want
    # to exercise the weather path.
    monkeypatch.delenv("METSERVICE_API_KEY", raising=False)


# ── Quote ────────────────────────────────────────────────────────────────────


def test_fetch_quote_happy_path(client: httpx.Client) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://zenquotes.io/api/random").mock(
            return_value=httpx.Response(
                200, json=[{"q": "Stay hungry.", "a": "Steve Jobs"}]
            )
        )
        out = morning_brief.fetch_quote(client)
    assert 'Stay hungry.' in out
    assert '— Steve Jobs' in out


def test_fetch_quote_falls_back_on_http_error(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deterministic fallback — freeze random.choice so the test isn't flaky.
    monkeypatch.setattr(
        random, "choice", lambda seq: ("Deterministic wisdom.", "Test Author")
    )
    with respx.mock() as mock:
        mock.get("https://zenquotes.io/api/random").mock(
            return_value=httpx.Response(500, text="boom")
        )
        out = morning_brief.fetch_quote(client)
    assert "Deterministic wisdom." in out
    assert "Test Author" in out


def test_fetch_quote_falls_back_on_bad_payload(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        random, "choice", lambda seq: ("Fallback quote.", "Fallback Author")
    )
    with respx.mock() as mock:
        # Empty list — triggers the "unexpected shape" path.
        mock.get("https://zenquotes.io/api/random").mock(
            return_value=httpx.Response(200, json=[])
        )
        out = morning_brief.fetch_quote(client)
    assert "Fallback quote." in out


# ── Weather (MetOcean) ───────────────────────────────────────────────────────


def _fake_metocean_payload() -> dict[str, Any]:
    # 3 days x 2 variables; temperatures in Kelvin, wind in m/s.
    return {
        "dimensions": {
            "time": {
                "data": [
                    "2026-04-19T00:00:00Z",
                    "2026-04-20T00:00:00Z",
                    "2026-04-21T00:00:00Z",
                ]
            }
        },
        "variables": {
            "air.temperature.at-2m": {"data": [288.15, 289.15, 290.15]},  # 15/16/17°C
            "wind.speed.at-10m": {"data": [5.0, 7.0, 10.0]},  # 18/25/36 km/h
            "precipitation.rate": {"data": [0.0, 0.0, 0.1]},
        },
    }


def test_fetch_weather_skips_without_api_key(client: httpx.Client) -> None:
    # _clear_metoceankey fixture ensures the env var is unset.
    out = morning_brief.fetch_weather(client)
    assert "Weather unavailable" in out
    assert "METSERVICE_API_KEY" in out


def test_fetch_weather_formats_kelvin_and_mps(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METSERVICE_API_KEY", "test-key-xyz")
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://forecast-v2.metoceanapi.com/point/time").mock(
            return_value=httpx.Response(200, json=_fake_metocean_payload())
        )
        out = morning_brief.fetch_weather(client)
    # 288.15K - 273.15 = 15.0°C; 5.0 m/s * 3.6 = 18 km/h
    assert "15.0°C" in out
    assert "18 km/h" in out
    # Both configured locations rendered.
    assert "Auckland" in out
    assert "Conifer Grove" in out


def test_fetch_weather_degrades_on_http_error(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METSERVICE_API_KEY", "test-key-xyz")
    with respx.mock() as mock:
        mock.post("https://forecast-v2.metoceanapi.com/point/time").mock(
            return_value=httpx.Response(503)
        )
        out = morning_brief.fetch_weather(client)
    # Section is still produced with a warning per-location.
    assert "## ☁️ Weather" in out
    assert "Unavailable" in out


# ── GitHub Trending ──────────────────────────────────────────────────────────


_GITHUB_FIXTURE_HTML = """
<html><body>
<article class="Box-row">
  <h2><a href="/foo/bar">foo / bar</a></h2>
  <p>A cool thing</p>
  <a href="/foo/bar/stargazers">1,234</a>
</article>
<article class="Box-row">
  <h2><a href="/baz/qux">baz / qux</a></h2>
  <p>Another cool thing</p>
  <a href="/baz/qux/stargazers">42</a>
</article>
</body></html>
"""


def test_fetch_github_trending_parses_top_repos(client: httpx.Client) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://github.com/trending?since=daily").mock(
            return_value=httpx.Response(200, text=_GITHUB_FIXTURE_HTML)
        )
        out = morning_brief.fetch_github_trending(client)
    assert "foo/bar" in out
    assert "baz/qux" in out
    assert "⭐ 1234" in out  # commas stripped
    assert "A cool thing" in out


def test_fetch_github_trending_warns_on_failure(client: httpx.Client) -> None:
    with respx.mock() as mock:
        mock.get("https://github.com/trending?since=daily").mock(
            return_value=httpx.Response(500)
        )
        out = morning_brief.fetch_github_trending(client)
    assert "GitHub Trending unavailable" in out


def test_fetch_github_trending_warns_when_markup_changes(
    client: httpx.Client,
) -> None:
    # Layout change → no article.Box-row elements → friendly warning.
    with respx.mock() as mock:
        mock.get("https://github.com/trending?since=daily").mock(
            return_value=httpx.Response(200, text="<html><body>empty</body></html>")
        )
        out = morning_brief.fetch_github_trending(client)
    assert "GitHub Trending unavailable" in out


# ── Hacker News ──────────────────────────────────────────────────────────────


def test_fetch_hacker_news_formats_top_hits(client: httpx.Client) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "title": "Thing happened",
                            "url": "https://example.com/a",
                            "objectID": "1",
                            "points": 100,
                            "num_comments": 20,
                        },
                        {
                            "title": "Discuss only",
                            "url": None,
                            "objectID": "2",
                            "points": 50,
                            "num_comments": 30,
                        },
                    ]
                },
            )
        )
        out = morning_brief.fetch_hacker_news(client)
    assert "Thing happened" in out
    assert "(100 pts, 20 comments)" in out
    # Missing URL falls back to item page.
    assert "news.ycombinator.com/item?id=2" in out


def test_fetch_hacker_news_warns_on_empty(client: httpx.Client) -> None:
    with respx.mock() as mock:
        mock.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(200, json={"hits": []})
        )
        out = morning_brief.fetch_hacker_news(client)
    assert "Hacker News unavailable" in out


# ── TechCrunch AI ────────────────────────────────────────────────────────────


_TECHCRUNCH_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>OpenAI launches new GPT feature</title>
    <link>https://techcrunch.com/openai</link>
    <pubDate>Sun, 19 Apr 2026 12:00:00 +0000</pubDate>
  </item>
  <item>
    <title>Unrelated startup news</title>
    <link>https://techcrunch.com/startup</link>
    <pubDate>Sun, 19 Apr 2026 10:00:00 +0000</pubDate>
  </item>
</channel></rss>
"""


def test_fetch_techcrunch_prefers_ai_items(client: httpx.Client) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://techcrunch.com/feed/").mock(
            return_value=httpx.Response(200, text=_TECHCRUNCH_RSS)
        )
        out = morning_brief.fetch_techcrunch_ai(client)
    # Only the AI-matching item is surfaced — the unrelated one is filtered.
    assert "OpenAI launches new GPT feature" in out
    assert "Unrelated startup" not in out


def test_fetch_techcrunch_warns_when_feed_empty(client: httpx.Client) -> None:
    with respx.mock() as mock:
        mock.get("https://techcrunch.com/feed/").mock(
            return_value=httpx.Response(
                200,
                text='<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>',
            )
        )
        out = morning_brief.fetch_techcrunch_ai(client)
    assert "TechCrunch AI unavailable" in out


# ── NZ News ──────────────────────────────────────────────────────────────────


_RNZ_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>NZ headline</title>
    <link>https://rnz.co.nz/a</link>
    <pubDate>Sun, 19 Apr 2026 07:00:00 +1200</pubDate>
  </item>
</channel></rss>
"""


def test_fetch_nz_news_prefers_rnz(client: httpx.Client) -> None:
    with respx.mock() as mock:
        mock.get("https://www.rnz.co.nz/rss/news.xml").mock(
            return_value=httpx.Response(200, text=_RNZ_RSS)
        )
        # Stuff is not reached because RNZ succeeded.
        out = morning_brief.fetch_nz_news(client)
    assert "RNZ News" in out
    assert "NZ headline" in out


def test_fetch_nz_news_falls_back_to_stuff(client: httpx.Client) -> None:
    stuff_rss = _RNZ_RSS.replace("rnz.co.nz", "stuff.co.nz")
    with respx.mock() as mock:
        mock.get("https://www.rnz.co.nz/rss/news.xml").mock(
            return_value=httpx.Response(500)
        )
        mock.get("https://www.stuff.co.nz/rss").mock(
            return_value=httpx.Response(200, text=stuff_rss)
        )
        out = morning_brief.fetch_nz_news(client)
    assert "Stuff NZ" in out


def test_fetch_nz_news_warns_when_both_fail(client: httpx.Client) -> None:
    with respx.mock() as mock:
        mock.get("https://www.rnz.co.nz/rss/news.xml").mock(
            return_value=httpx.Response(500)
        )
        mock.get("https://www.stuff.co.nz/rss").mock(
            return_value=httpx.Response(500)
        )
        out = morning_brief.fetch_nz_news(client)
    assert "NZ News unavailable" in out


# ── RSS parser edge cases ────────────────────────────────────────────────────


def test_rss_items_handles_atom_feed() -> None:
    atom = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Atom entry</title>
        <link href="https://example.com/atom-a"/>
        <updated>2026-04-19T07:00:00Z</updated>
      </entry>
    </feed>
    """
    items = morning_brief._rss_items(atom)
    assert len(items) == 1
    assert items[0]["title"] == "Atom entry"
    assert items[0]["link"] == "https://example.com/atom-a"


def test_rss_items_missing_fields_default_placeholders() -> None:
    rss = """<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <item></item>
    </channel></rss>
    """
    items = morning_brief._rss_items(rss)
    assert items == [{"title": "Untitled", "link": "#", "pubdate": "?"}]


# ── Assembly + destination path ──────────────────────────────────────────────


def test_destination_path_shape(tmp_path: Path, when_nz: datetime) -> None:
    # Don't pre-create — `_destination` must mkdir.
    dest = morning_brief._destination(tmp_path, when_nz)
    assert dest == tmp_path / "Daily" / "2026" / "04" / "19-daily-news.md"
    assert dest.parent.is_dir()


def test_build_brief_assembles_all_sections(
    client: httpx.Client, when_nz: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        random, "choice", lambda seq: ("Deterministic.", "Tester")
    )
    # Force ZenQuotes to fail so we exercise the fallback path
    # deterministically; everything else returns a canned success.
    with respx.mock() as mock:
        mock.get("https://zenquotes.io/api/random").mock(
            return_value=httpx.Response(500)
        )
        mock.get("https://github.com/trending?since=daily").mock(
            return_value=httpx.Response(200, text=_GITHUB_FIXTURE_HTML)
        )
        mock.get("https://hn.algolia.com/api/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "title": "HN story",
                            "url": "https://example.com/hn",
                            "objectID": "10",
                            "points": 1,
                            "num_comments": 0,
                        }
                    ]
                },
            )
        )
        mock.get("https://techcrunch.com/feed/").mock(
            return_value=httpx.Response(200, text=_TECHCRUNCH_RSS)
        )
        mock.get("https://www.rnz.co.nz/rss/news.xml").mock(
            return_value=httpx.Response(200, text=_RNZ_RSS)
        )
        out = morning_brief.build_brief(client, when_nz)

    assert out.startswith("---\n")  # frontmatter
    assert "# Daily News - 2026-04-19" in out
    assert "Deterministic." in out  # fallback quote rendered
    assert "Weather unavailable" in out  # no METSERVICE_API_KEY
    assert "foo/bar" in out  # github
    assert "HN story" in out
    assert "OpenAI launches" in out  # techcrunch AI filter
    assert "RNZ News" in out


# ── CLI / main ───────────────────────────────────────────────────────────────


def _all_http_mocked(mock: Any) -> None:
    """Register minimal successful mocks for every outbound call in main()."""
    mock.get("https://zenquotes.io/api/random").mock(
        return_value=httpx.Response(
            200, json=[{"q": "CLI quote", "a": "CLI Author"}]
        )
    )
    mock.get("https://github.com/trending?since=daily").mock(
        return_value=httpx.Response(200, text=_GITHUB_FIXTURE_HTML)
    )
    mock.get("https://hn.algolia.com/api/v1/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": "t",
                        "url": "https://example.com",
                        "objectID": "x",
                        "points": 1,
                        "num_comments": 0,
                    }
                ]
            },
        )
    )
    mock.get("https://techcrunch.com/feed/").mock(
        return_value=httpx.Response(200, text=_TECHCRUNCH_RSS)
    )
    mock.get("https://www.rnz.co.nz/rss/news.xml").mock(
        return_value=httpx.Response(200, text=_RNZ_RSS)
    )


def test_main_writes_file_and_prints_to_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Freeze NZ "now" deterministically so we can assert the dest path.
    class _FrozenDT(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            if tz is UTC:
                return datetime(2026, 4, 18, 19, 0, tzinfo=UTC)
            # NZDT equivalent = UTC + 12h = 2026-04-19 07:00
            return datetime(2026, 4, 19, 7, 0, tzinfo=tz)

    monkeypatch.setattr(morning_brief, "datetime", _FrozenDT)

    with respx.mock() as mock:
        _all_http_mocked(mock)
        rc = morning_brief.main(["--vault-root", str(tmp_path)])

    assert rc == 0
    dest = tmp_path / "Daily" / "2026" / "04" / "19-daily-news.md"
    assert dest.is_file()
    contents = dest.read_text(encoding="utf-8")
    assert "# Daily News - 2026-04-19" in contents
    # Same markdown also goes to stdout so the bot can reply with it.
    captured = capsys.readouterr()
    assert "# Daily News - 2026-04-19" in captured.out


def test_main_explicit_date_override_is_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with respx.mock() as mock:
        _all_http_mocked(mock)
        rc = morning_brief.main(
            ["--vault-root", str(tmp_path), "--date", "2026-01-15"]
        )
    assert rc == 0
    assert (tmp_path / "Daily" / "2026" / "01" / "15-daily-news.md").is_file()


def test_main_rejects_bad_date_format(tmp_path: Path) -> None:
    rc = morning_brief.main(
        ["--vault-root", str(tmp_path), "--date", "not-a-date"]
    )
    assert rc == 2
    # No file written on validation failure.
    assert not (tmp_path / "Daily").exists()


def test_main_rejects_missing_vault_root(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    rc = morning_brief.main(["--vault-root", str(missing)])
    assert rc == 2
