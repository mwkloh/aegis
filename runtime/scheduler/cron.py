"""Cron expression validation + next-run math.

Thin wrapper over `croniter` so the rest of AEGIS never imports
the library directly — if we ever swap implementations, the blast
radius is one module.

**Subset of standard Unix cron** (§6 of the plan):

* Five fields: `m h dom mon dow`.
* `*`, `*/N`, lists (`1,15,30`), ranges (`1-5`).
* **Not** accepted: `@reboot`, `@daily`, month/day names, `L`/`W`/`#`.
  `croniter` happens to support some of these — we do *not* document
  them, and the `describe()` renderer doesn't know them. Don't rely
  on accidental support.

Timezone posture matches the rest of AEGIS: datetimes crossing
module boundaries must be tz-aware. Internally everything is
normalised to UTC before handing to croniter; the returned
`next_run` is always tz-aware UTC.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

from croniter import CroniterBadCronError, croniter

_FIVE_FIELD = re.compile(r"^\s*\S+(\s+\S+){4}\s*$")


def validate(expr: str) -> None:
    """Raise `ValueError` if `expr` isn't a valid 5-field cron expression."""
    if not expr or not expr.strip():
        raise ValueError("cron expression is empty")
    if not _FIVE_FIELD.match(expr):
        raise ValueError(
            f"cron expression must have 5 whitespace-separated fields, got {expr!r}"
        )
    try:
        croniter(expr)
    except (CroniterBadCronError, ValueError) as exc:
        raise ValueError(f"invalid cron expression {expr!r}: {exc}") from exc


def next_run(expr: str, *, after: datetime) -> datetime:
    """Return the next firing time of `expr` strictly after `after`.

    `after` must be tz-aware. The returned datetime is always
    tz-aware UTC. Strict-after semantics (> not >=) prevent a tick
    that just fired from re-scheduling at the same instant.
    """
    if after.tzinfo is None:
        raise ValueError("`after` must be tz-aware")
    validate(expr)
    after_utc = after.astimezone(UTC)
    it = croniter(expr, after_utc)
    # croniter returns a float epoch; passing `ret_type=datetime`
    # hands back a naive datetime in the input's wall-clock zone,
    # so we reconstruct from the epoch to guarantee a UTC result.
    nxt_epoch: float = it.get_next(float)
    return datetime.fromtimestamp(nxt_epoch, tz=UTC)


# --- describe ------------------------------------------------------------


_RE_DAILY_FIXED = re.compile(r"^(\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+\*$")
_RE_EVERY_N_HOURS = re.compile(r"^0\s+\*/(\d{1,2})\s+\*\s+\*\s+\*$")
_RE_EVERY_N_MINUTES = re.compile(r"^\*/(\d{1,2})\s+\*\s+\*\s+\*\s+\*$")


def describe(expr: str) -> str:
    """Short operator-facing label for `/cron list`. Falls back to raw expr.

    Pure-function renderer — no croniter round-trip. We only label
    patterns the operator is likely to set via `/cron add`; anything
    exotic returns the raw expression so `/cron list` stays honest.
    """
    expr = expr.strip()
    if expr == "* * * * *":
        return "every minute"
    m = _RE_DAILY_FIXED.match(expr)
    if m:
        minute, hour = int(m.group(1)), int(m.group(2))
        return f"daily {hour:02d}:{minute:02d} UTC"
    m = _RE_EVERY_N_HOURS.match(expr)
    if m:
        return f"every {int(m.group(1))}h"
    m = _RE_EVERY_N_MINUTES.match(expr)
    if m:
        return f"every {int(m.group(1))}m"
    return f"{expr} (UTC)"


__all__ = [
    "describe",
    "next_run",
    "validate",
]
