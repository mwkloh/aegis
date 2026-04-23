"""Morning brief — daily quote, NZ weather, GitHub Trending, Hacker News, TechCrunch AI, NZ news.

Writes Markdown to ``<vault_root>/Daily/YYYY/MM/DD-daily-news.md`` and prints the
full Markdown to stdout. Never raises — each section degrades to an inline warning
so the brief always produces *something*.

Ported from ``~/.kai/workspace/skills/morning_brief/daily_news_aggregator.py``:
dropped the four AI-newsletter RSS feeds, dropped direct Telegram delivery (the
AEGIS bot owns the Telegram surface), rewrote HTTP on ``httpx``, swapped RSS
parsing onto ``defusedxml.ElementTree`` for safety.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import defusedxml.ElementTree as DET  # type: ignore[import-untyped]
import httpx
from bs4 import BeautifulSoup

NZDT = ZoneInfo("Pacific/Auckland")
REQUEST_TIMEOUT = 15.0
USER_AGENT = "AEGIS-MorningBrief/1.0"
_PUBDATE_MAX_CHARS = 16
_MIN_FULL_NAME_PARTS = 2
_TOP_N = 5

FALLBACK_QUOTES: tuple[tuple[str, str], ...] = (
    ("The only way to do great work is to love what you do.", "Steve Jobs"),
    ("In the middle of every difficulty lies opportunity.", "Albert Einstein"),
    ("It does not matter how slowly you go as long as you do not stop.", "Confucius"),
    ("Life is what happens when you're busy making other plans.", "John Lennon"),
    ("The future belongs to those who believe in the beauty of their dreams.", "Eleanor Roosevelt"),
    ("Strive not to be a success, but rather to be of value.", "Albert Einstein"),
    ("You miss 100% of the shots you don't take.", "Wayne Gretzky"),
    ("Whether you think you can or you think you can't, you're right.", "Henry Ford"),
    (
        "The best time to plant a tree was 20 years ago. The second best time is now.",
        "Chinese Proverb",
    ),
    ("An unexamined life is not worth living.", "Socrates"),
)

_LOCATIONS: dict[str, dict[str, float]] = {
    "Auckland": {"lat": -36.8485, "lon": 174.7633},
    "Conifer Grove": {"lat": -37.0530, "lon": 174.9080},
}

AI_KEYWORDS = (
    "ai",
    "artificial intelligence",
    "machine learning",
    "llm",
    "gpt",
    "openai",
    "gemini",
    "claude",
)

log = logging.getLogger("morning_brief")


# ── Section error helper ─────────────────────────────────────────────────────

def _section_error(name: str, exc: Exception) -> str:
    log.warning("section %s failed: %s", name, exc)
    return f"⚠️ *{name} unavailable*: {exc}\n"


# ── RSS helper (defusedxml) ──────────────────────────────────────────────────

def _rss_items(xml_text: str) -> list[dict[str, str]]:
    """Parse RSS 2.0 or Atom feeds into ``[{title, link, pubdate}]`` dicts."""
    root = DET.fromstring(xml_text)
    items: list[dict[str, str]] = []
    for elem in root.iter():
        tag = str(elem.tag).rsplit("}", 1)[-1]
        if tag in ("item", "entry"):
            items.append(_rss_fields(elem))
    return items


def _rss_fields(node: Any) -> dict[str, str]:
    title = ""
    link = ""
    pubdate = ""
    for child in node:
        tag = str(child.tag).rsplit("}", 1)[-1]
        text = (child.text or "").strip()
        if tag == "title" and not title:
            title = text
        elif tag == "link" and not link:
            link = text or child.attrib.get("href", "")
        elif tag in ("pubDate", "published", "updated") and not pubdate:
            pubdate = text
    return {"title": title or "Untitled", "link": link or "#", "pubdate": pubdate or "?"}


# ── Section fetchers ─────────────────────────────────────────────────────────

def fetch_quote(client: httpx.Client) -> str:
    try:
        resp = client.get("https://zenquotes.io/api/random")
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list):
            q = data[0].get("q", "")
            a = data[0].get("a", "Unknown")
            if q:
                return f'💬 *"{q}"* — {a}\n'
        raise ValueError("unexpected ZenQuotes response shape")
    except Exception as exc:
        log.info("ZenQuotes fallback: %s", exc)
        q, a = random.choice(FALLBACK_QUOTES)  # noqa: S311  # nosec B311
        return f'💬 *"{q}"* — {a}\n'


def _metocean_fetch(
    client: httpx.Client, api_key: str, lat: float, lon: float
) -> dict[str, Any]:
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {
        "points": [{"lon": lon, "lat": lat}],
        "variables": [
            "air.temperature.at-2m",
            "wind.speed.at-10m",
            "precipitation.rate",
        ],
        "time": {"from": now_iso, "interval": "24h", "repeat": 3},
    }
    resp = client.post(
        "https://forecast-v2.metoceanapi.com/point/time",
        content=json.dumps(body),
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    return payload


def _format_metocean(name: str, data: dict[str, Any]) -> str:
    lines = [f"### {name}"]
    try:
        times = data["dimensions"]["time"]["data"]
        variables = data["variables"]
        temps = variables.get("air.temperature.at-2m", {}).get("data", [])
        winds = variables.get("wind.speed.at-10m", {}).get("data", [])
        for i, t_iso in enumerate(times[:3]):
            dt = datetime.fromisoformat(t_iso.replace("Z", "+00:00"))
            nz = dt.astimezone(NZDT)
            day_label = nz.strftime("%a %d %b")
            temp_str = f"{temps[i] - 273.15:.1f}°C" if i < len(temps) else "?"
            wind_str = f"{winds[i] * 3.6:.0f} km/h" if i < len(winds) else "?"
            lines.append(f"📅 **{day_label}** — 🌡️ {temp_str} | 💨 {wind_str}")
    except Exception as exc:
        lines.append(f"⚠️ Could not parse weather data: {exc}")
    return "\n".join(lines)


def fetch_weather(client: httpx.Client) -> str:
    api_key = os.environ.get("METSERVICE_API_KEY", "")
    if not api_key:
        log.info("METSERVICE_API_KEY unset — skipping weather")
        return "## ☁️ Weather\n\n*Weather unavailable — `METSERVICE_API_KEY` not configured.*\n\n"
    out = ["## ☁️ Weather\n"]
    for display_name, coords in _LOCATIONS.items():
        try:
            data = _metocean_fetch(client, api_key, coords["lat"], coords["lon"])
            out.append(_format_metocean(display_name, data))
            out.append("")
        except Exception as exc:
            out.append(f"### {display_name}\n⚠️ Unavailable: {exc}\n")
    return "\n".join(out) + "\n"


def fetch_github_trending(client: httpx.Client) -> str:
    try:
        resp = client.get("https://github.com/trending?since=daily")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        repos = soup.select("article.Box-row")
        if not repos:
            raise ValueError("no trending articles found — GitHub markup may have changed")
        lines = ["## 🔥 GitHub Trending\n"]
        for i, repo in enumerate(repos[:_TOP_N], 1):
            h2 = repo.select_one("h2 a")
            if not h2:
                continue
            full_name = h2.get_text(separator="/", strip=True).replace(" ", "")
            parts = [p for p in full_name.split("/") if p]
            if len(parts) >= _MIN_FULL_NAME_PARTS:
                full_name = "/".join(parts[-2:])
            desc_el = repo.select_one("p")
            desc = desc_el.get_text(strip=True) if desc_el else "No description"
            stars = "?"
            for a in repo.select("a"):
                href = a.get("href", "")
                if isinstance(href, str) and href.endswith("/stargazers"):
                    stars = a.get_text(strip=True).replace(",", "").strip()
                    break
            url = f"https://github.com/{full_name}"
            lines.append(f"{i}. **{full_name}** ⭐ {stars}")
            lines.append(f"   > {desc}")
            lines.append(f"   🔗 {url}\n")
        return "\n".join(lines) + "\n"
    except Exception as exc:
        return _section_error("GitHub Trending", exc)


def fetch_hacker_news(client: httpx.Client) -> str:
    try:
        resp = client.get(
            "https://hn.algolia.com/api/v1/search",
            params={"tags": "front_page", "hitsPerPage": 5},
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        if not hits:
            raise ValueError("HN returned no hits")
        lines = ["## 📰 Hacker News Top 5\n"]
        for i, hit in enumerate(hits[:_TOP_N], 1):
            title = hit.get("title") or "Untitled"
            obj_id = hit.get("objectID", "")
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={obj_id}"
            pts = hit.get("points", 0)
            comments = hit.get("num_comments", 0)
            lines.append(f"{i}. **{title}** ({pts} pts, {comments} comments)")
            lines.append(f"   🔗 {url}\n")
        return "\n".join(lines) + "\n"
    except Exception as exc:
        return _section_error("Hacker News", exc)


def fetch_techcrunch_ai(client: httpx.Client) -> str:
    try:
        resp = client.get("https://techcrunch.com/feed/")
        resp.raise_for_status()
        items = _rss_items(resp.text)
        if not items:
            raise ValueError("TechCrunch feed empty")
        ai_items = [it for it in items if any(kw in it["title"].lower() for kw in AI_KEYWORDS)]
        selected = (ai_items or items)[:_TOP_N]
        lines = ["## 🤖 TechCrunch AI\n"]
        for i, it in enumerate(selected, 1):
            pubdate = it["pubdate"][:_PUBDATE_MAX_CHARS]
            lines.append(f"{i}. **{it['title']}**")
            lines.append(f"   🔗 {it['link']} | 📅 {pubdate}\n")
        return "\n".join(lines) + "\n"
    except Exception as exc:
        return _section_error("TechCrunch AI", exc)


def fetch_nz_news(client: httpx.Client) -> str:
    sources = (
        ("RNZ News", "https://www.rnz.co.nz/rss/news.xml"),
        ("Stuff NZ", "https://www.stuff.co.nz/rss"),
    )
    last_exc: Exception = ValueError("no NZ sources tried")
    for source_name, url in sources:
        try:
            resp = client.get(url)
            resp.raise_for_status()
            items = _rss_items(resp.text)
            if not items:
                raise ValueError("empty RSS feed")
            lines = [f"## 🇳🇿 NZ News ({source_name})\n"]
            for i, it in enumerate(items[:_TOP_N], 1):
                pubdate = it["pubdate"][:_PUBDATE_MAX_CHARS]
                lines.append(f"{i}. **{it['title']}**")
                lines.append(f"   🔗 {it['link']} | 📅 {pubdate}\n")
            return "\n".join(lines) + "\n"
        except Exception as exc:
            log.warning("NZ news source %s failed: %s", source_name, exc)
            last_exc = exc
    return _section_error("NZ News", last_exc)


# ── Assembly ─────────────────────────────────────────────────────────────────

def build_brief(client: httpx.Client, when_nz: datetime) -> str:
    date_str = when_nz.strftime("%Y-%m-%d")
    frontmatter = (
        "---\n"
        f"created: {date_str}\n"
        f"updated: {date_str}\n"
        "type: log\n"
        "tags:\n  - daily\n  - media\n  - aegis\n"
        "sources:\n"
        "  - github-trending\n"
        "  - hacker-news\n"
        "  - techcrunch\n"
        "  - rnz\n"
        "---\n"
    )
    sections = [
        frontmatter,
        f"# Daily News - {date_str}\n",
        fetch_quote(client),
        fetch_weather(client),
        fetch_github_trending(client),
        fetch_hacker_news(client),
        fetch_techcrunch_ai(client),
        fetch_nz_news(client),
    ]
    return "\n".join(sections)


def _destination(vault_root: Path, when_nz: datetime) -> Path:
    daily_dir = vault_root / "Daily" / when_nz.strftime("%Y") / when_nz.strftime("%m")
    daily_dir.mkdir(parents=True, exist_ok=True)
    return daily_dir / f"{when_nz.strftime('%d')}-daily-news.md"


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AEGIS morning brief.")
    parser.add_argument(
        "--vault-root",
        required=True,
        type=Path,
        help=(
            "Obsidian vault root; brief is written under "
            "<vault_root>/Daily/YYYY/MM/DD-daily-news.md."
        ),
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override date (YYYY-MM-DD, interpreted as NZDT). Defaults to today.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _parse_args(argv)

    if args.date:
        try:
            when_nz = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=NZDT)
        except ValueError as exc:
            log.error("invalid --date %r: %s", args.date, exc)
            return 2
    else:
        when_nz = datetime.now(NZDT)

    vault_root = args.vault_root.expanduser().resolve()
    if not vault_root.is_dir():
        log.error("vault-root does not exist or is not a directory: %s", vault_root)
        return 2

    with httpx.Client(timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        brief = build_brief(client, when_nz)

    dest = _destination(vault_root, when_nz)
    dest.write_text(brief, encoding="utf-8")
    log.info("wrote %s (%d bytes)", dest, len(brief))

    sys.stdout.write(brief)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
