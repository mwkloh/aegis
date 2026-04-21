"""`BoardWriter` — serialises `BoardResult` to a Markdown file.

Filename: `YYYY-MM-DD-BOARD-<id>-<slug>.md`. `<slug>` is lowercase, the
first six words of the question, non-alphanumerics replaced with `-`,
collapsed, max 60 chars. `output_dir` is created on first write.
"""
from __future__ import annotations

import re
from pathlib import Path

from runtime.board.engine import BoardResult

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_CHARS = 60
_SLUG_WORDS = 6
_SYNTH_ABSENT = "*(synthesis not configured)*"


class BoardWriter:
    """Render a `BoardResult` to a Markdown file in `output_dir`."""

    def __init__(self, *, output_dir: Path) -> None:
        self._output_dir = output_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def write(self, result: BoardResult) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = self._filename(result)
        path = self._output_dir / filename
        path.write_text(self._render(result), encoding="utf-8")
        return path

    def _filename(self, result: BoardResult) -> str:
        date = result.created_at.strftime("%Y-%m-%d")
        slug = self._slug(result.question)
        return f"{date}-{result.board_id}-{slug}.md"

    def _slug(self, question: str) -> str:
        words = question.strip().split()[:_SLUG_WORDS]
        joined = " ".join(words).lower()
        slug = _SLUG_PATTERN.sub("-", joined).strip("-")
        if len(slug) > _MAX_SLUG_CHARS:
            slug = slug[:_MAX_SLUG_CHARS].rstrip("-")
        return slug or "board"

    def _render(self, result: BoardResult) -> str:
        parts: list[str] = []
        parts.append(self._frontmatter(result))
        parts.append("")
        parts.append(f"# Board: {result.question}")
        parts.append("")
        stamp = result.created_at.strftime("%Y-%m-%d")
        n = len(result.panelist_responses)
        parts.append(f"*{stamp} · {n} panelists · {result.board_id}*")
        parts.append("")
        parts.append("## Synthesis")
        parts.append("")
        parts.append(result.synthesis if result.synthesis is not None else _SYNTH_ABSENT)
        parts.append("")
        for r in result.panelist_responses:
            parts.append("---")
            parts.append("")
            parts.append(f"## {r.name}")
            parts.append(f"*{r.model} via {r.provider} · {r.latency_ms}ms*")
            parts.append("")
            if r.error is not None:
                parts.append(f"[Error: {r.error}]")
            else:
                parts.append(r.response)
            parts.append("")
        return "\n".join(parts)

    def _frontmatter(self, result: BoardResult) -> str:
        lines = ["---"]
        lines.append(f"board_id: {result.board_id}")
        lines.append(f'question: "{self._escape_yaml(result.question)}"')
        lines.append(f"date: {result.created_at.strftime('%Y-%m-%d')}")
        lines.append("panelists:")
        for r in result.panelist_responses:
            lines.append(f"  - name: {r.name}")
            lines.append(f"    model: {r.model}")
            lines.append(f"    provider: {r.provider}")
        if result.synthesis is not None:
            lines.append("synthesis: present")
        lines.append("---")
        return "\n".join(lines)

    @staticmethod
    def _escape_yaml(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')
