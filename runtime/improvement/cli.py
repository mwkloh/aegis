"""Human approval loop for Plane 3.

```text
python -m runtime.improvement.cli                 # interactive review
python -m runtime.improvement.cli --list          # show pending only
python -m runtime.improvement.cli --decide IMP-7f3a1c2b \
    --verdict approve --rationale "safe; matches demo"
```

Pending = proposals whose latest decision is not `approve` or `reject`
(i.e. never decided, or last verdict was `defer`). Approval also queues
a `CT-NNN` row in `CODING_TASKS.md` (idempotent on the `imp_id`).
Returns 0 on clean exit; 1 only on I/O failure or unknown `IMP-id`.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from typing import Literal, get_args

from runtime.config import AegisConfig, get_config
from runtime.events import EventStream

from .coding_tasks import queue_task
from .decisions import (
    Decision,
    HumanVerdict,
    latest_by_imp,
    load_decisions,
    record_decision,
)
from .proposal_loader import LoadedProposal, load_proposals

InputFn = Callable[[str], str]

_VERDICT_KEYS: dict[str, HumanVerdict] = {
    "a": "approve",
    "r": "reject",
    "d": "defer",
}
_VALID_VERDICTS: tuple[HumanVerdict, ...] = get_args(HumanVerdict)
Action = Literal["approve", "reject", "defer", "skip", "quit"]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg = get_config()
    proposals = load_proposals(cfg.aegis_home)

    if args.list:
        return _do_list(cfg, proposals)
    if args.decide:
        return _do_scriptable(
            cfg,
            proposals,
            imp_id=args.decide,
            verdict=args.verdict,
            rationale=args.rationale or "",
        )
    return _do_interactive(cfg, proposals)


def _do_list(cfg: AegisConfig, proposals: list[LoadedProposal]) -> int:
    pending = _pending(proposals, load_decisions(cfg.aegis_home))
    print(f"[improvement] {len(pending)} pending of {len(proposals)} loaded")
    for p in pending:
        print(_format_proposal(p))
    return 0


def _do_scriptable(
    cfg: AegisConfig,
    proposals: list[LoadedProposal],
    *,
    imp_id: str,
    verdict: HumanVerdict | None,
    rationale: str,
) -> int:
    target = next((p for p in proposals if p.imp_id == imp_id), None)
    if target is None:
        print(f"[improvement] error: unknown IMP id {imp_id!r}", file=sys.stderr)
        return 1
    if verdict is None:
        print("[improvement] error: --verdict required with --decide", file=sys.stderr)
        return 1
    events = EventStream(cfg.storage.sessions_dir)
    decision = record_decision(
        cfg.aegis_home,
        imp_id=target.imp_id,
        verdict=verdict,
        rationale=rationale,
        events=events,
    )
    print(_decision_message(decision, target))
    if verdict == "approve":
        task = queue_task(cfg.aegis_home, target)
        if task is None:
            print(f"[improvement] {target.imp_id} already queued — no new CT")
        else:
            print(f"[improvement] queued {task.ct_id} → {target.imp_id}")
    return 0


def _do_interactive(
    cfg: AegisConfig,
    proposals: list[LoadedProposal],
    *,
    input_fn: InputFn = input,
) -> int:
    decisions = load_decisions(cfg.aegis_home)
    pending = _pending(proposals, decisions)
    summary = _summary(decisions)
    print(
        f"[improvement] loaded {len(proposals)} proposals from "
        f"reflection/PROPOSALS.md"
    )
    print(
        f"[improvement] {len(pending)} pending  "
        f"({summary['approve']} approved, {summary['reject']} rejected, "
        f"{summary['defer']} deferred)"
    )
    if not pending:
        print("[improvement] nothing to review.")
        return 0

    events = EventStream(cfg.storage.sessions_dir)
    for proposal in pending:
        print(_format_proposal(proposal))
        action = _prompt_action(input_fn)
        if action == "quit":
            print("[improvement] quit — remaining proposals untouched.")
            return 0
        if action == "skip":
            print(f"[improvement] skipped {proposal.imp_id}")
            continue
        rationale = input_fn("rationale> ").strip()
        decision = record_decision(
            cfg.aegis_home,
            imp_id=proposal.imp_id,
            verdict=action,
            rationale=rationale,
            events=events,
        )
        print(_decision_message(decision, proposal))
        if action == "approve":
            task = queue_task(cfg.aegis_home, proposal)
            if task is None:
                print(f"[improvement] {proposal.imp_id} already queued — no new CT")
            else:
                print(f"[improvement] queued {task.ct_id} → {proposal.imp_id}")
    return 0


def _pending(
    proposals: Iterable[LoadedProposal], decisions: list[Decision]
) -> list[LoadedProposal]:
    latest = latest_by_imp(decisions)
    out: list[LoadedProposal] = []
    for p in proposals:
        prior = latest.get(p.imp_id)
        if prior is None or prior.verdict == "defer":
            out.append(p)
    return out


def _summary(decisions: list[Decision]) -> dict[HumanVerdict, int]:
    """Count latest-per-IMP verdicts. Applier verdicts are not summarised
    here — that is the apply CLI's concern."""
    counts: dict[HumanVerdict, int] = {"approve": 0, "reject": 0, "defer": 0}
    for decision in latest_by_imp(decisions).values():
        match decision.verdict:
            case "approve" | "reject" | "defer" as v:
                counts[v] += 1
            case _:
                pass
    return counts


def _prompt_action(input_fn: InputFn) -> Action:
    while True:
        raw = input_fn("[a]pprove / [r]eject / [d]efer / [s]kip / [q]uit > ").strip().lower()
        if raw in ("q", "quit"):
            return "quit"
        if raw in ("s", "skip"):
            return "skip"
        if raw in _VERDICT_KEYS:
            return _VERDICT_KEYS[raw]
        if raw in _VALID_VERDICTS:
            return raw
        print("[improvement] unrecognized — type a/r/d/s/q")


def _format_proposal(p: LoadedProposal) -> str:
    affected = ", ".join(p.affected) or "—"
    rationale = p.rationale.strip() or "—"
    return (
        "\n──────────────────────────────────────────────\n"
        f"{p.imp_id} — {p.pattern_detector} (risk: {p.risk})\n"
        f"Affected:  {affected}\n"
        f"Change:    {p.change}\n"
        f"Rationale: {rationale}\n"
        "──────────────────────────────────────────────"
    )


def _decision_message(decision: Decision, proposal: LoadedProposal) -> str:
    sup = (
        f" (supersedes {decision.supersedes})" if decision.supersedes else ""
    )
    return f"[improvement] recorded {decision.verdict} on {proposal.imp_id}{sup}"


def _verdict_arg(value: str) -> HumanVerdict:
    if value not in _VALID_VERDICTS:
        raise argparse.ArgumentTypeError(
            f"verdict must be one of {_VALID_VERDICTS}, got {value!r}"
        )
    return value


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="improvement")
    p.add_argument(
        "--list",
        action="store_true",
        help="Print pending proposals and exit (no decisions recorded).",
    )
    p.add_argument(
        "--decide",
        metavar="IMP-id",
        default=None,
        help="Scriptable single-decision mode. Requires --verdict.",
    )
    p.add_argument(
        "--verdict",
        type=_verdict_arg,
        default=None,
        help="approve | reject | defer (used with --decide).",
    )
    p.add_argument(
        "--rationale",
        default="",
        help="Free-text justification (used with --decide).",
    )
    return p.parse_args(argv)


# Test hook — wraps `_do_interactive` so test code can inject input().
def run_interactive(
    cfg: AegisConfig,
    proposals: list[LoadedProposal],
    *,
    input_fn: InputFn,
) -> int:
    return _do_interactive(cfg, proposals, input_fn=input_fn)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
