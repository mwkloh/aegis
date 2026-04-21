"""BoardWriter — deterministic filename + Markdown rendering."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.board.engine import BoardResult, PanelistResponse
from runtime.board.writer import BoardWriter

pytestmark = pytest.mark.unit


def _resp(
    name: str,
    *,
    response: str = "text",
    error: str | None = None,
    model: str = "m",
    provider: str = "ollama",
    latency_ms: int = 1000,
) -> PanelistResponse:
    return PanelistResponse(
        name=name,
        model=model,
        provider=provider,
        response=response,
        latency_ms=latency_ms,
        error=error,
    )


def _result(
    question: str = "Should we migrate to Postgres?",
    *,
    synthesis: str | None = "Bottom line.",
    responses: tuple[PanelistResponse, ...] = (),
) -> BoardResult:
    return BoardResult(
        board_id="BOARD-a3f2",
        question=question,
        created_at=datetime(2026, 4, 21, 12, 34, tzinfo=UTC),
        panelist_responses=responses or (_resp("Analyst"),),
        synthesis=synthesis,
    )


def test_write_creates_output_dir_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "does" / "not" / "exist"
    writer = BoardWriter(output_dir=target)
    path = writer.write(_result())
    assert target.is_dir()
    assert path.parent == target
    assert path.is_file()


def test_filename_has_date_board_id_and_slug(tmp_path: Path) -> None:
    writer = BoardWriter(output_dir=tmp_path)
    path = writer.write(_result())
    name = path.name
    assert name.startswith("2026-04-21-BOARD-a3f2-")
    assert name.endswith(".md")
    assert "should-we-migrate-to-postgres" in name


def test_slug_truncates_to_first_six_words_and_strips_punctuation(tmp_path: Path) -> None:
    writer = BoardWriter(output_dir=tmp_path)
    long_q = "Will the new launch plan, priced at 40%, beat Q2 targets in Europe next year?"
    path = writer.write(_result(question=long_q))
    # first 6 words: "will-the-new-launch-plan-priced"
    assert "will-the-new-launch-plan-priced" in path.name
    assert "%" not in path.name
    assert "," not in path.name


def test_markdown_has_frontmatter_with_board_id_and_question(tmp_path: Path) -> None:
    writer = BoardWriter(output_dir=tmp_path)
    path = writer.write(_result())
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "board_id: BOARD-a3f2" in text
    assert 'question: "Should we migrate to Postgres?"' in text


def test_markdown_renders_synthesis_section(tmp_path: Path) -> None:
    writer = BoardWriter(output_dir=tmp_path)
    path = writer.write(_result(synthesis="All roads lead to Postgres."))
    text = path.read_text(encoding="utf-8")
    assert "## Synthesis" in text
    assert "All roads lead to Postgres." in text


def test_markdown_renders_synthesis_absent_placeholder_when_none(tmp_path: Path) -> None:
    writer = BoardWriter(output_dir=tmp_path)
    path = writer.write(_result(synthesis=None))
    text = path.read_text(encoding="utf-8")
    assert "## Synthesis" in text
    assert "*(synthesis not configured)*" in text


def test_markdown_renders_error_panelist_with_error_marker(tmp_path: Path) -> None:
    writer = BoardWriter(output_dir=tmp_path)
    result = _result(
        responses=(
            _resp("Analyst", response="All good."),
            _resp("Strategist", response="", error="timeout", latency_ms=60000),
        )
    )
    path = writer.write(result)
    text = path.read_text(encoding="utf-8")
    assert "## Analyst" in text
    assert "All good." in text
    assert "## Strategist" in text
    assert "[Error: timeout]" in text
