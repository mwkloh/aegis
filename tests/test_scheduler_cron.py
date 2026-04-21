"""Phase 10 Track A1 — cron expression validation + next-run math.

Pins:

* `validate(expr)` raises `ValueError` with a clear message on any
  expression croniter can't parse. No silent acceptance.
* `next_run(expr, after)` returns a tz-aware UTC datetime strictly
  greater than `after` (never equal — a tick that just fired must
  schedule forward, not loop).
* Naive input to `after` is rejected (same posture as the rest of
  AEGIS — timezones must be explicit at every boundary).
* `describe(expr)` renders a short operator-facing label for
  `/cron list` — "daily 07:00 UTC" etc. Best-effort; falls back to
  the raw expression for patterns the renderer doesn't recognise.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from runtime.scheduler.cron import describe, next_run, validate

pytestmark = pytest.mark.unit


# --- validate ------------------------------------------------------------


def test_validate_accepts_standard_five_field() -> None:
    validate("0 7 * * *")
    validate("*/5 * * * *")
    validate("0 0 1 * *")
    validate("0 */6 * * *")


def test_validate_accepts_lists_and_ranges() -> None:
    validate("0 9,12,17 * * *")
    validate("0 9-17 * * 1-5")


def test_validate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate("")


def test_validate_rejects_garbage() -> None:
    # 5 tokens so it clears the field-count guard and is rejected by croniter.
    with pytest.raises(ValueError, match="invalid cron"):
        validate("not a cron expression here")


def test_validate_rejects_wrong_field_count() -> None:
    with pytest.raises(ValueError, match="5 whitespace-separated fields"):
        validate("0 7 * *")  # four fields
    with pytest.raises(ValueError, match="5 whitespace-separated fields"):
        validate("0 7 * * * *")  # six fields


def test_validate_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="invalid cron"):
        validate("0 25 * * *")  # hour > 23
    with pytest.raises(ValueError, match="invalid cron"):
        validate("0 0 32 * *")  # day > 31


# --- next_run ------------------------------------------------------------


def test_next_run_daily_seven_am() -> None:
    after = datetime(2026, 4, 20, 6, 0, tzinfo=UTC)
    nxt = next_run("0 7 * * *", after=after)
    assert nxt == datetime(2026, 4, 20, 7, 0, tzinfo=UTC)


def test_next_run_rolls_to_tomorrow_when_past() -> None:
    after = datetime(2026, 4, 20, 8, 0, tzinfo=UTC)
    nxt = next_run("0 7 * * *", after=after)
    assert nxt == datetime(2026, 4, 21, 7, 0, tzinfo=UTC)


def test_next_run_strictly_after_input() -> None:
    # A tick at exactly the trigger time must schedule forward,
    # not return the same instant (would cause a tight loop).
    after = datetime(2026, 4, 20, 7, 0, tzinfo=UTC)
    nxt = next_run("0 7 * * *", after=after)
    assert nxt > after
    assert nxt == datetime(2026, 4, 21, 7, 0, tzinfo=UTC)


def test_next_run_every_5_minutes() -> None:
    after = datetime(2026, 4, 20, 12, 2, 30, tzinfo=UTC)
    nxt = next_run("*/5 * * * *", after=after)
    assert nxt == datetime(2026, 4, 20, 12, 5, tzinfo=UTC)


def test_next_run_returns_utc_aware() -> None:
    after = datetime(2026, 4, 20, 6, 0, tzinfo=UTC)
    nxt = next_run("0 7 * * *", after=after)
    assert nxt.tzinfo is not None
    assert nxt.utcoffset() == timedelta(0)


def test_next_run_rejects_naive_datetime() -> None:
    naive = datetime(2026, 4, 20, 6, 0)
    with pytest.raises(ValueError, match="tz-aware"):
        next_run("0 7 * * *", after=naive)


def test_next_run_normalises_non_utc_input_to_utc() -> None:
    # Same absolute instant, expressed in a non-UTC zone — must
    # produce the same trigger time in UTC terms.
    plus_twelve = timezone(timedelta(hours=12))
    after = datetime(2026, 4, 20, 18, 0, tzinfo=plus_twelve)  # 06:00 UTC
    nxt = next_run("0 7 * * *", after=after)
    assert nxt == datetime(2026, 4, 20, 7, 0, tzinfo=UTC)


def test_next_run_raises_on_bad_expr() -> None:
    after = datetime(2026, 4, 20, 6, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match=r"invalid cron|whitespace-separated"):
        next_run("garbage", after=after)


# --- describe ------------------------------------------------------------


def test_describe_daily_at_fixed_hour() -> None:
    assert describe("0 7 * * *") == "daily 07:00 UTC"
    assert describe("30 22 * * *") == "daily 22:30 UTC"


def test_describe_every_n_hours() -> None:
    assert describe("0 */6 * * *") == "every 6h"
    assert describe("0 */2 * * *") == "every 2h"


def test_describe_every_n_minutes() -> None:
    assert describe("*/5 * * * *") == "every 5m"
    assert describe("*/15 * * * *") == "every 15m"


def test_describe_every_minute() -> None:
    assert describe("* * * * *") == "every minute"


def test_describe_falls_back_to_raw_expr_when_unknown() -> None:
    # Complex expressions we don't handle — return raw expr with (UTC) suffix.
    assert describe("0 9-17 * * 1-5") == "0 9-17 * * 1-5 (UTC)"
