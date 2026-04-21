"""Pure-stdlib time lookup. No network, no LLM.

Given a free-text query containing a city or IANA timezone hint, resolves to
the closest matching zone and returns the current local time. Unknown zones
return a structured error in the tool's return payload — never raise.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_CITY_TO_ZONE: dict[str, str] = {
    "tokyo": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "singapore": "Asia/Singapore",
    "hong kong": "Asia/Hong_Kong",
    "hongkong": "Asia/Hong_Kong",
    "shanghai": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "london": "Europe/London",
    "uk": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "new york": "America/New_York",
    "nyc": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "sf": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "utc": "UTC",
    "gmt": "UTC",
}

_TOKEN = re.compile(r"[a-z][a-z_\- ]{0,39}", re.IGNORECASE)


def time_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Resolve `args['query']` to a local time string.

    Returns:
        {"zone": "<IANA zone>", "now_iso": "<ISO8601>", "formatted": "<pretty>"}

    If no zone can be resolved, returns a payload with `"zone": null` and a
    helpful `"formatted"` message explaining the fallback.
    """
    query = args.get("query", "") or ""
    if not isinstance(query, str):
        raise TypeError("time: 'query' must be a string")

    zone_name = _resolve_zone(query)
    if zone_name is None:
        return {
            "zone": None,
            "now_iso": None,
            "formatted": (
                "I couldn't recognise a timezone in that request. "
                "Try a city like 'Tokyo' or a zone like 'America/New_York'."
            ),
        }

    try:
        tz = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError:
        return {
            "zone": None,
            "now_iso": None,
            "formatted": f"Unknown timezone {zone_name!r}.",
        }

    now = datetime.now(tz)
    return {
        "zone": zone_name,
        "now_iso": now.isoformat(timespec="seconds"),
        "formatted": now.strftime(f"It is %H:%M on %A, %d %B %Y in {zone_name}."),
    }


def _resolve_zone(query: str) -> str | None:
    """Pull a zone name out of the user's free-text query."""
    # 1. Exact IANA zone (case-sensitive) anywhere in query.
    for token in query.split():
        if "/" in token:
            try:
                ZoneInfo(token)
            except ZoneInfoNotFoundError:
                continue
            return token

    # 2. Known city aliases, longest match first so "new york" beats "new".
    lowered = query.lower()
    for alias in sorted(_CITY_TO_ZONE, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return _CITY_TO_ZONE[alias]

    # 3. Single-word tokens that happen to be IANA regions (e.g. "UTC").
    for match in _TOKEN.finditer(query):
        word = match.group(0).strip().lower()
        if word in _CITY_TO_ZONE:
            return _CITY_TO_ZONE[word]

    return None
