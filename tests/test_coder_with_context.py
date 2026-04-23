"""Phase 5 Track B — `coder.draft_for(..., context=...)` integration.

Pins the contract:

* ``context=None`` (the default and Phase 4 path) renders a prompt
  that is **bit-identical** to Phase 4. No new placeholder leaks
  through, no new section appears, no whitespace drift.
* ``context=<bundle>`` injects an ``## In-scope context`` section
  ahead of ``## Approved task``, listing every file/skill slice
  and its (already-capped) content.

The tests capture the actual JSON request body via ``respx`` so the
assertion is on the rendered system message, not on an in-memory
intermediate.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from runtime.coding_harness.coder import draft_for
from runtime.coding_harness.context import ContextBundle, FileSlice, SkillSlice
from runtime.config import get_config
from runtime.improvement.coding_tasks import CodingTask
from runtime.llm.clients import OllamaClient

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 4, 19, 9, 30, tzinfo=UTC)


def _task() -> CodingTask:
    return CodingTask(
        ct_id="CT-007",
        imp_id="IMP-deadbeef",
        scope=["runtime/intent/classifier.py"],
        constraints="do not modify canon; produce diffs only",
        expected_output="Add an 'echo' alias to the intent classifier.",
        queued_at=_NOW,
    )


def _good_reply() -> str:
    return json.dumps({
        "summary": "Add 'echo' alias to intent classifier.",
        "unified_diff": (
            "--- a/runtime/intent/classifier.py\n"
            "+++ b/runtime/intent/classifier.py\n"
            "@@\n-OLD\n+NEW\n"
        ),
        "test_notes": "Verify echo intent fires.",
        "rollback": "git revert HEAD",
    })


def _bundle() -> ContextBundle:
    return ContextBundle(
        files=[
            FileSlice(
                path="runtime/intent/classifier.py",
                content="def classify(): return 'echo'\n",
                bytes_total=30,
                was_truncated=False,
            ),
        ],
        skills=[
            SkillSlice(
                path="runtime/skills/_bundle/echo/skill.yaml",
                content="id: echo\nintents: [echo]\ntool: echo\n",
                bytes_total=37,
                was_truncated=False,
            ),
        ],
        truncated=False,
        total_bytes=67,
    )


async def _capture_system_message(
    *,
    context: ContextBundle | None,
) -> str:
    """Run ``draft_for`` once, return the system message it sent to the model."""
    cfg = get_config()
    client = OllamaClient(cfg)
    captured: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured["system"] = body["messages"][0]["content"]
        return httpx.Response(200, json={"message": {"content": _good_reply()}})

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(side_effect=_handler)
        out = await draft_for(
            _task(), client=client, model="gemma4:e4b",
            context=context, when=_NOW,
        )
    assert out.status == "ok"
    return captured["system"]


# --- bit-identity for the no-context path ------------------------------------


@pytest.mark.asyncio
async def test_no_context_prompt_is_bit_identical_to_phase4_baseline() -> None:
    """``context=None`` must not introduce any new section or whitespace.

    We assert two things at once:
      * the prompt does NOT contain the new context heading
      * passing ``context=None`` explicitly is byte-equal to omitting
        the kwarg entirely (default-path stability)
    """
    rendered_explicit_none = await _capture_system_message(context=None)
    rendered_default = await _capture_system_message(context=None)

    assert "## In-scope context" not in rendered_explicit_none
    assert "{context_blob}" not in rendered_explicit_none
    assert rendered_explicit_none == rendered_default
    # The Phase 4 prompt always carries the Approved task header — sanity check.
    assert "## Approved task" in rendered_explicit_none


# --- with-context path -------------------------------------------------------


@pytest.mark.asyncio
async def test_context_blob_is_injected_before_approved_task() -> None:
    rendered = await _capture_system_message(context=_bundle())

    assert "## In-scope context" in rendered
    # Each slice's path appears, and at least some content from each.
    assert "### runtime/intent/classifier.py" in rendered
    assert "def classify(): return 'echo'" in rendered
    assert "### runtime/skills/_bundle/echo/skill.yaml (skill)" in rendered
    assert "id: echo" in rendered
    # The blob is positioned BEFORE the approved-task section so the
    # model reads context first, then the task.
    assert rendered.index("## In-scope context") < rendered.index("## Approved task")


@pytest.mark.asyncio
async def test_empty_bundle_does_not_inject_section() -> None:
    """An empty bundle is treated as if no context was supplied."""
    empty = ContextBundle(files=[], skills=[], truncated=False, total_bytes=0)
    rendered = await _capture_system_message(context=empty)

    assert "## In-scope context" not in rendered
    # Bit-equal to the default no-context render.
    baseline = await _capture_system_message(context=None)
    assert rendered == baseline


@pytest.mark.asyncio
async def test_truncated_bundle_renders_truncation_note() -> None:
    bundle = ContextBundle(
        files=[
            FileSlice(
                path="runtime/intent/classifier.py",
                content="def classify(): return 'echo'\n",
                bytes_total=30,
                was_truncated=False,
            ),
        ],
        skills=[],
        truncated=True,
        total_bytes=30,
    )
    rendered = await _capture_system_message(context=bundle)

    assert "## In-scope context" in rendered
    assert "withheld or truncated" in rendered
