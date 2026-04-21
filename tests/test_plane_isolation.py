"""Plane isolation guards.

Runtime MUST NOT:
  1. Import from `improvement/` or `coding_harness/`.
  2. Open canonical .md files in write mode.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).parent.parent
RUNTIME = REPO_ROOT / "runtime"

FORBIDDEN_IMPORTS = ("improvement", "coding_harness")
CANONICAL_FILES = (
    "USER.md",
    "IDENTITY.md",
    "SOUL.md",
    "AGENTS.md",
    "MEMORY.md",
    "HEARTBEAT.md",
    "SKILLS_INDEX.md",
)
WRITE_MODE_RE = re.compile(r"""['"][rwax+btU]*[wax+]['"]""")


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def test_runtime_does_not_import_other_planes() -> None:
    offenders: list[tuple[Path, str]] = []
    for path in _python_files(RUNTIME):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    head = alias.name.split(".")[0]
                    if head in FORBIDDEN_IMPORTS:
                        offenders.append((path, alias.name))
            elif isinstance(node, ast.ImportFrom):
                head = (node.module or "").split(".")[0]
                if head in FORBIDDEN_IMPORTS:
                    offenders.append((path, node.module or ""))
    assert not offenders, f"runtime/ leaks into other planes: {offenders}"


def test_runtime_does_not_open_canon_for_write() -> None:
    """Heuristic: forbid `open(..., '<write-mode>')` near a canonical filename in runtime/."""
    offenders: list[tuple[Path, int]] = []
    for path in _python_files(RUNTIME):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "open(" not in line:
                continue
            if not WRITE_MODE_RE.search(line):
                continue
            if any(canon in line for canon in CANONICAL_FILES):
                offenders.append((path, lineno))
    assert not offenders, f"runtime/ writes canonical .md: {offenders}"


def test_runtime_does_not_use_yaml_load() -> None:
    """Only `yaml.safe_load` is allowed."""
    bad: list[tuple[Path, int]] = []
    pattern = re.compile(r"yaml\.load\s*\(")
    for path in _python_files(RUNTIME):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                bad.append((path, lineno))
    assert not bad, f"runtime/ uses yaml.load — must use yaml.safe_load: {bad}"
