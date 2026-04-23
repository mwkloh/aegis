"""Phase 7 §4.2 — `long_running.py` coordinator.

Pins:

* `InFlightRegistry` is single-writer safe: a second `try_acquire`
  for the same chat fails until `release`.
* `LongRunningRunner` rejects a second `/apply` while one is in
  flight (same chat); a different chat is unaffected.
* `/apply` validates `CT-NNN` format; no CT = usage; bad CT = typed
  rejection.
* `/apply` happy path: runner invoked with `python -m
  runtime.coding_harness.apply_cli CT-001 [extras]`, initial
  "Running ..." reply, final `edit_text` carries the subprocess
  output and status.
* `/apply` failure path: non-zero exit renders "failed (exit=N)".
* `/harness` passes args through untouched (flag-based CT
  selection); header echoes the args.
* Output longer than `MAX_LONG_RUNNING_CHARS` is tail-clipped with
  a leading `"… "`.
* `/apply` is released on exception in the runner, so a follow-up
  command in the same chat can still acquire.

No network, no real subprocess — `FakeSubprocessRunner` records
every call and replays canned `(exit_code, output)` tuples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from runtime.chat.telegram.dispatch import ParsedCommand
from runtime.chat.telegram.long_running import (
    MAX_LONG_RUNNING_CHARS,
    InFlightRegistry,
    LongRunningRunner,
)

pytestmark = pytest.mark.unit


# --- Fakes -------------------------------------------------------------


@dataclass
class _FakeEditableMessage:
    """One initial reply + later edit_text calls captured in-order."""

    initial_text: str
    edits: list[str] = field(default_factory=list)

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)


@dataclass
class _FakeReplyable:
    """reply_text returns a fresh `_FakeEditableMessage` each call."""

    replies: list[_FakeEditableMessage] = field(default_factory=list)

    async def reply_text(self, text: str) -> _FakeEditableMessage:
        m = _FakeEditableMessage(initial_text=text)
        self.replies.append(m)
        return m


@dataclass
class _FakeSubprocessRunner:
    """Injectable runner: records argv/cwd and replays canned tuples."""

    canned: list[tuple[int, str]] = field(default_factory=list)
    calls: list[tuple[list[str], Path]] = field(default_factory=list)
    raises: Exception | None = None

    async def run(self, argv: list[str], *, cwd: Path) -> tuple[int, str]:
        self.calls.append((list(argv), cwd))
        if self.raises is not None:
            raise self.raises
        if not self.canned:
            return (0, "")
        return self.canned.pop(0)


def _cmd(name: str, *args: str) -> ParsedCommand:
    return ParsedCommand(name=name, args=tuple(args))


# --- InFlightRegistry -------------------------------------------------


def test_registry_acquire_and_release() -> None:
    reg = InFlightRegistry()
    assert reg.try_acquire(1, "/apply") is True
    assert reg.try_acquire(1, "/apply") is False
    assert reg.current(1) == "/apply"
    reg.release(1)
    assert reg.current(1) is None
    assert reg.try_acquire(1, "/harness") is True


def test_registry_per_chat_isolation() -> None:
    reg = InFlightRegistry()
    assert reg.try_acquire(1, "/apply") is True
    assert reg.try_acquire(2, "/apply") is True  # different chat — allowed
    assert reg.current(1) == "/apply"
    assert reg.current(2) == "/apply"


def test_registry_release_is_noop_when_absent() -> None:
    reg = InFlightRegistry()
    reg.release(999)  # must not raise
    assert reg.current(999) is None


# --- /apply validation -------------------------------------------------


async def test_apply_no_args_renders_usage(tmp_path: Path) -> None:
    runner = LongRunningRunner(tmp_path, runner=_FakeSubprocessRunner())
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply"), message=msg)
    assert len(msg.replies) == 1
    assert "Usage:" in msg.replies[0].initial_text
    assert msg.replies[0].edits == []  # no edit on pure usage


async def test_apply_bad_ct_format_rejected(tmp_path: Path) -> None:
    runner = LongRunningRunner(tmp_path, runner=_FakeSubprocessRunner())
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply", "NOTACT"), message=msg)
    assert "expects a CT-NNN id" in msg.replies[0].initial_text


# --- /apply happy + failure paths -------------------------------------


async def test_apply_happy_path_invokes_runner(tmp_path: Path) -> None:
    fake = _FakeSubprocessRunner(canned=[(0, "applied_clean: CT-001\n")])
    runner = LongRunningRunner(
        tmp_path, runner=fake, python_executable="/usr/bin/python3"
    )
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply", "CT-001"), message=msg)
    # Exactly one argv call, shape is correct.
    assert len(fake.calls) == 1
    argv, cwd = fake.calls[0]
    assert argv == [
        "/usr/bin/python3",
        "-m",
        "runtime.coding_harness.apply_cli",
        "CT-001",
    ]
    assert cwd == tmp_path
    # Initial "Running..." reply then final edit.
    assert len(msg.replies) == 1
    sent = msg.replies[0]
    assert sent.initial_text == "Running /apply CT-001..."
    assert len(sent.edits) == 1
    assert "/apply CT-001 succeeded" in sent.edits[0]
    assert "applied_clean: CT-001" in sent.edits[0]


async def test_apply_normalizes_ct_prefix_casing(tmp_path: Path) -> None:
    fake = _FakeSubprocessRunner(canned=[(0, "")])
    runner = LongRunningRunner(tmp_path, runner=fake)
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply", "ct-007"), message=msg)
    # argv carries canonical uppercase CT-
    assert fake.calls[0][0][3] == "CT-007"
    assert msg.replies[0].initial_text == "Running /apply CT-007..."


async def test_apply_passes_extra_flags(tmp_path: Path) -> None:
    fake = _FakeSubprocessRunner(canned=[(0, "")])
    runner = LongRunningRunner(tmp_path, runner=fake)
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1, cmd=_cmd("/apply", "CT-001", "--dry-run"), message=msg
    )
    argv = fake.calls[0][0]
    assert argv[-2:] == ["CT-001", "--dry-run"]


async def test_apply_failure_renders_exit_code(tmp_path: Path) -> None:
    fake = _FakeSubprocessRunner(canned=[(1, "traceback: something broke")])
    runner = LongRunningRunner(tmp_path, runner=fake)
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply", "CT-001"), message=msg)
    final = msg.replies[0].edits[0]
    assert "failed (exit=1)" in final
    assert "traceback: something broke" in final


# --- /harness pass-through --------------------------------------------


async def test_harness_no_args_invokes_cli(tmp_path: Path) -> None:
    fake = _FakeSubprocessRunner(canned=[(0, "drafted 2 patches\n")])
    runner = LongRunningRunner(
        tmp_path, runner=fake, python_executable="/usr/bin/python3"
    )
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/harness"), message=msg)
    argv, cwd = fake.calls[0]
    assert argv == ["/usr/bin/python3", "-m", "runtime.coding_harness.cli"]
    assert cwd == tmp_path
    assert msg.replies[0].initial_text == "Running /harness..."
    assert "succeeded" in msg.replies[0].edits[0]


async def test_harness_passes_flags_through(tmp_path: Path) -> None:
    fake = _FakeSubprocessRunner(canned=[(0, "")])
    runner = LongRunningRunner(tmp_path, runner=fake)
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1,
        cmd=_cmd("/harness", "--task", "CT-001", "--with-context"),
        message=msg,
    )
    argv = fake.calls[0][0]
    assert argv[-3:] == ["--task", "CT-001", "--with-context"]
    assert msg.replies[0].initial_text.startswith(
        "Running /harness --task CT-001 --with-context..."
    )


# --- in-flight guard ---------------------------------------------------


async def test_in_flight_rejects_concurrent_same_chat(tmp_path: Path) -> None:
    # Pre-occupy the slot so a second /apply fails fast without touching
    # the runner.
    reg = InFlightRegistry()
    reg.try_acquire(1, "/apply")
    fake = _FakeSubprocessRunner(canned=[(0, "")])
    runner = LongRunningRunner(tmp_path, runner=fake, registry=reg)
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply", "CT-001"), message=msg)
    assert fake.calls == []  # runner never invoked
    assert "Already running /apply" in msg.replies[0].initial_text


async def test_in_flight_permits_different_chat(tmp_path: Path) -> None:
    reg = InFlightRegistry()
    reg.try_acquire(1, "/apply")  # chat 1 holds the slot
    fake = _FakeSubprocessRunner(canned=[(0, "")])
    runner = LongRunningRunner(tmp_path, runner=fake, registry=reg)
    msg = _FakeReplyable()
    await runner.run(chat_id=2, cmd=_cmd("/apply", "CT-002"), message=msg)
    assert len(fake.calls) == 1
    assert "succeeded" in msg.replies[0].edits[0]


async def test_slot_released_after_success(tmp_path: Path) -> None:
    reg = InFlightRegistry()
    fake = _FakeSubprocessRunner(canned=[(0, ""), (0, "")])
    runner = LongRunningRunner(tmp_path, runner=fake, registry=reg)
    # First run
    msg1 = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply", "CT-001"), message=msg1)
    # Second run in same chat — slot should be free
    msg2 = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply", "CT-002"), message=msg2)
    assert len(fake.calls) == 2
    assert reg.current(1) is None


async def test_slot_released_after_runner_exception(tmp_path: Path) -> None:
    reg = InFlightRegistry()
    fake = _FakeSubprocessRunner(raises=RuntimeError("boom"))
    runner = LongRunningRunner(tmp_path, runner=fake, registry=reg)
    msg = _FakeReplyable()
    with pytest.raises(RuntimeError):
        await runner.run(
            chat_id=1, cmd=_cmd("/apply", "CT-001"), message=msg
        )
    # Exception propagated but slot released so a retry can acquire.
    assert reg.current(1) is None


# --- tail clipping -----------------------------------------------------


async def test_long_output_is_tail_clipped(tmp_path: Path) -> None:
    huge = "x" * (MAX_LONG_RUNNING_CHARS + 500)
    fake = _FakeSubprocessRunner(canned=[(0, huge)])
    runner = LongRunningRunner(tmp_path, runner=fake)
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply", "CT-001"), message=msg)
    final = msg.replies[0].edits[0]
    assert len(final) <= MAX_LONG_RUNNING_CHARS
    assert final.startswith("… ")  # leading ellipsis marks tail-preference


# --- unknown command (defensive) --------------------------------------


async def test_unknown_command_replies_gracefully(tmp_path: Path) -> None:
    runner = LongRunningRunner(tmp_path, runner=_FakeSubprocessRunner())
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/apply-not-really"), message=msg)
    assert "unknown long-running command" in msg.replies[0].initial_text


# --- registration -----------------------------------------------------


def test_commands_set(tmp_path: Path) -> None:
    runner = LongRunningRunner(tmp_path, runner=_FakeSubprocessRunner())
    assert runner.commands == frozenset({"/apply", "/harness", "/brief"})


# --- /brief -----------------------------------------------------------


async def test_brief_not_configured_without_vault_root(tmp_path: Path) -> None:
    # No vault_root threaded → clear operator-visible error, no subprocess.
    fake = _FakeSubprocessRunner()
    runner = LongRunningRunner(tmp_path, runner=fake)
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/brief"), message=msg)
    assert fake.calls == []
    assert "not configured" in msg.replies[0].initial_text


async def test_brief_happy_path_returns_full_markdown(tmp_path: Path) -> None:
    body = "# Daily brief\n\nmotivational quote here"
    fake = _FakeSubprocessRunner(canned=[(0, body)])
    runner = LongRunningRunner(
        tmp_path,
        runner=fake,
        python_executable="/usr/bin/python3",
        vault_root=tmp_path / "vault",
        brief_script=tmp_path / "brief.py",
    )
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/brief"), message=msg)
    assert len(fake.calls) == 1
    argv, cwd = fake.calls[0]
    assert argv == [
        "/usr/bin/python3",
        str(tmp_path / "brief.py"),
        "--vault-root",
        str(tmp_path / "vault"),
    ]
    assert cwd == tmp_path
    assert msg.replies[0].initial_text == "Running /brief..."
    # Full stdout is echoed back (no "succeeded" banner — operators want
    # the markdown, not a status line).
    assert msg.replies[0].edits[0] == body


async def test_brief_failure_renders_exit_code(tmp_path: Path) -> None:
    fake = _FakeSubprocessRunner(canned=[(2, "Traceback: MetOcean down")])
    runner = LongRunningRunner(
        tmp_path,
        runner=fake,
        vault_root=tmp_path / "vault",
        brief_script=tmp_path / "brief.py",
    )
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/brief"), message=msg)
    final = msg.replies[0].edits[0]
    assert "/brief failed (exit=2)" in final
    assert "Traceback: MetOcean down" in final


async def test_brief_in_flight_guard_rejects_concurrent(tmp_path: Path) -> None:
    reg = InFlightRegistry()
    reg.try_acquire(1, "/brief")
    fake = _FakeSubprocessRunner()
    runner = LongRunningRunner(
        tmp_path,
        runner=fake,
        registry=reg,
        vault_root=tmp_path / "vault",
        brief_script=tmp_path / "brief.py",
    )
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/brief"), message=msg)
    assert fake.calls == []
    assert "Already running /brief" in msg.replies[0].initial_text


async def test_brief_not_configured_without_brief_script(tmp_path: Path) -> None:
    # vault_root set but brief_script missing → operator-visible error,
    # no subprocess. Guards the `build_application` wiring: if the
    # morning_brief descriptor isn't in the catalog, `/brief` must
    # fail cleanly instead of trying to exec `None`.
    fake = _FakeSubprocessRunner()
    runner = LongRunningRunner(
        tmp_path, runner=fake, vault_root=tmp_path / "vault"
    )
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=_cmd("/brief"), message=msg)
    assert fake.calls == []
    assert "not configured" in msg.replies[0].initial_text


# --- run_skill (intent-dispatch path) ---------------------------------


async def test_run_skill_happy_path_echoes_output(tmp_path: Path) -> None:
    body = "# Brief\n\n- quote\n- weather\n"
    fake = _FakeSubprocessRunner(canned=[(0, body)])
    runner = LongRunningRunner(tmp_path, runner=fake)
    msg = _FakeReplyable()
    argv = ["/usr/bin/python3", str(tmp_path / "morning_brief.py")]
    await runner.run_skill(
        chat_id=1, skill_id="morning_brief", argv=argv, message=msg
    )
    assert len(fake.calls) == 1
    sent_argv, cwd = fake.calls[0]
    assert sent_argv == argv
    assert cwd == tmp_path
    assert msg.replies[0].initial_text == "Running morning_brief..."
    # echo_output=True is the default — subprocess stdout shown verbatim.
    assert msg.replies[0].edits[0] == body


async def test_run_skill_failure_renders_exit_code(tmp_path: Path) -> None:
    fake = _FakeSubprocessRunner(canned=[(3, "boom")])
    runner = LongRunningRunner(tmp_path, runner=fake)
    msg = _FakeReplyable()
    await runner.run_skill(
        chat_id=1, skill_id="morning_brief", argv=["/bin/false"], message=msg
    )
    final = msg.replies[0].edits[0]
    assert "morning_brief failed (exit=3)" in final
    assert "boom" in final


async def test_run_skill_in_flight_guard_rejects_concurrent(tmp_path: Path) -> None:
    reg = InFlightRegistry()
    reg.try_acquire(1, "morning_brief")
    fake = _FakeSubprocessRunner()
    runner = LongRunningRunner(tmp_path, runner=fake, registry=reg)
    msg = _FakeReplyable()
    await runner.run_skill(
        chat_id=1, skill_id="morning_brief", argv=["/bin/true"], message=msg
    )
    assert fake.calls == []
    assert "Already running morning_brief" in msg.replies[0].initial_text


async def test_run_skill_echo_output_false_adds_banner(tmp_path: Path) -> None:
    fake = _FakeSubprocessRunner(canned=[(0, "details")])
    runner = LongRunningRunner(tmp_path, runner=fake)
    msg = _FakeReplyable()
    await runner.run_skill(
        chat_id=1,
        skill_id="example",
        argv=["/bin/true"],
        message=msg,
        echo_output=False,
    )
    final = msg.replies[0].edits[0]
    assert "example succeeded" in final
    assert "details" in final


async def test_run_skill_releases_slot_after_success(tmp_path: Path) -> None:
    reg = InFlightRegistry()
    fake = _FakeSubprocessRunner(canned=[(0, "ok")])
    runner = LongRunningRunner(tmp_path, runner=fake, registry=reg)
    msg = _FakeReplyable()
    await runner.run_skill(
        chat_id=1, skill_id="morning_brief", argv=["/bin/true"], message=msg
    )
    # Slot released so a follow-up /apply can acquire.
    assert reg.current(1) is None
