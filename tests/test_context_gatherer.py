"""Phase 5 Track B — `runtime/coding_harness/context.py`.

Pins the read-only contract: caps (4 KB per file / 15 KB total),
``..`` rejection, missing files skipped silently, skill YAMLs
matched by id-substring against scope basenames.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.coding_harness.context import (
    DEFAULT_PER_FILE_BYTES,
    DEFAULT_TOTAL_BYTES,
    ContextBundle,
    FileSlice,
    SkillSlice,
    _safe_resolve,
    gather_context,
)

pytestmark = pytest.mark.unit


# --- repo factory -----------------------------------------------------------


def _make_repo(tmp_path: Path) -> Path:
    """Stand up a minimal repo with a couple of source files + skill YAMLs."""
    repo = tmp_path / "repo"
    (repo / "runtime/intent").mkdir(parents=True)
    (repo / "runtime/skills/catalog").mkdir(parents=True)
    (repo / "runtime/tools").mkdir(parents=True)

    (repo / "runtime/intent/classifier.py").write_text(
        "def classify(): return 'echo'\n", encoding="utf-8",
    )
    (repo / "runtime/tools/echo_tool.py").write_text(
        "def echo(msg): return msg\n", encoding="utf-8",
    )
    (repo / "runtime/skills/catalog/echo.yaml").write_text(
        "id: echo\nintents: [echo]\ntool: echo\n", encoding="utf-8",
    )
    (repo / "runtime/skills/catalog/time_query.yaml").write_text(
        "id: time_query\nintents: [time]\ntool: time\n", encoding="utf-8",
    )
    return repo


# --- defaults ---------------------------------------------------------------


def test_default_caps_match_phase5_decision() -> None:
    """Decision #5 (sign-off 2026-04-18): 15 KB total / 4 KB per file."""
    assert DEFAULT_TOTAL_BYTES == 15360
    assert DEFAULT_PER_FILE_BYTES == 4096


# --- happy path -------------------------------------------------------------


def test_gather_returns_in_scope_files(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    bundle = gather_context(repo, ["runtime/intent/classifier.py"])

    assert isinstance(bundle, ContextBundle)
    assert [f.path for f in bundle.files] == ["runtime/intent/classifier.py"]
    assert "classify" in bundle.files[0].content
    assert bundle.files[0].was_truncated is False
    assert bundle.truncated is False
    assert bundle.total_bytes > 0


def test_gather_pulls_skill_yaml_when_scope_basename_matches_id(
    tmp_path: Path,
) -> None:
    """Scope ``runtime/tools/echo_tool.py`` must pull in ``echo.yaml``."""
    repo = _make_repo(tmp_path)
    bundle = gather_context(repo, ["runtime/tools/echo_tool.py"])

    skill_paths = [s.path for s in bundle.skills]
    assert "runtime/skills/catalog/echo.yaml" in skill_paths
    # An unrelated skill must NOT be pulled in.
    assert "runtime/skills/catalog/time_query.yaml" not in skill_paths


def test_gather_includes_yaml_directly_listed_in_scope(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    bundle = gather_context(repo, ["runtime/skills/catalog/echo.yaml"])

    # Direct scope hit is recorded as a file slice (not a skill slice) and
    # is NOT also added as a skill (no double-counting).
    file_paths = [f.path for f in bundle.files]
    skill_paths = [s.path for s in bundle.skills]
    assert "runtime/skills/catalog/echo.yaml" in file_paths
    assert skill_paths == []


# --- caps -------------------------------------------------------------------


def test_per_file_cap_head_truncates_with_marker(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "runtime/skills/catalog").mkdir(parents=True)
    (repo / "runtime").mkdir(exist_ok=True)
    big = repo / "runtime/big.py"
    big.write_text("x" * 10_000, encoding="utf-8")  # > 4 KB

    bundle = gather_context(repo, ["runtime/big.py"])

    assert len(bundle.files) == 1
    slice_ = bundle.files[0]
    assert slice_.was_truncated is True
    assert slice_.bytes_total == 10_000
    assert "[truncated" in slice_.content
    assert slice_.content.startswith("x" * 100)  # head kept
    assert bundle.truncated is True


def test_total_cap_stops_further_entries(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "runtime").mkdir(parents=True)
    (repo / "runtime/skills/catalog").mkdir(parents=True)
    for i in range(5):
        (repo / f"runtime/f{i}.py").write_text("y" * 4096, encoding="utf-8")

    # 5 * 4 KB = 20 KB > 15 KB cap. The 4th gets a head-truncated slice;
    # the 5th's remaining budget falls below the useful floor and is skipped.
    bundle = gather_context(
        repo, [f"runtime/f{i}.py" for i in range(5)],
    )
    assert len(bundle.files) == 4
    assert bundle.files[-1].was_truncated is True
    assert bundle.total_bytes <= DEFAULT_TOTAL_BYTES
    assert bundle.truncated is True


def test_custom_caps_respected(tmp_path: Path) -> None:
    """A second file whose budget falls below the useful-bytes floor is
    skipped (with ``truncated=True`` to signal data was withheld) — we
    don't emit marker-only stub slices."""
    repo = tmp_path / "repo"
    (repo / "runtime").mkdir(parents=True)
    (repo / "runtime/skills/catalog").mkdir(parents=True)
    (repo / "runtime/a.py").write_text("a" * 200, encoding="utf-8")
    (repo / "runtime/b.py").write_text("b" * 200, encoding="utf-8")

    bundle = gather_context(
        repo, ["runtime/a.py", "runtime/b.py"],
        max_total_bytes=250, max_per_file_bytes=200,
    )
    assert [f.path for f in bundle.files] == ["runtime/a.py"]
    assert bundle.files[0].was_truncated is False
    assert bundle.truncated is True  # b.py was withheld
    assert bundle.total_bytes <= 250


def test_total_cap_strict_for_each_slice(tmp_path: Path) -> None:
    """Each slice's encoded length is ≤ the per-slice budget at its turn,
    so ``total_bytes`` always stays under ``max_total_bytes``."""
    repo = tmp_path / "repo"
    (repo / "runtime").mkdir(parents=True)
    (repo / "runtime/skills/catalog").mkdir(parents=True)
    for i in range(5):
        (repo / f"runtime/g{i}.py").write_text("z" * 4096, encoding="utf-8")

    bundle = gather_context(repo, [f"runtime/g{i}.py" for i in range(5)])

    assert bundle.total_bytes <= DEFAULT_TOTAL_BYTES
    for slice_ in bundle.files:
        assert len(slice_.content.encode("utf-8")) <= DEFAULT_PER_FILE_BYTES


# --- safety -----------------------------------------------------------------


def test_dotdot_escape_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "runtime/skills/catalog").mkdir(parents=True)
    (tmp_path / "outside.py").write_text("secret\n", encoding="utf-8")

    bundle = gather_context(repo, ["../outside.py"])

    assert bundle.files == []
    assert bundle.skills == []
    assert bundle.truncated is True  # rejection is signalled


def test_safe_resolve_rejects_absolute_outside_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "outside.py").write_text("x", encoding="utf-8")

    # Absolute paths under the repo resolve fine; outside is rejected.
    inside = repo / "in.py"
    inside.write_text("y", encoding="utf-8")
    assert _safe_resolve(repo, "in.py") == inside.resolve()
    assert _safe_resolve(repo, str(tmp_path / "outside.py")) is None


def test_missing_files_skipped_silently(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    bundle = gather_context(
        repo,
        ["runtime/intent/classifier.py", "runtime/does_not_exist.py"],
    )

    paths = [f.path for f in bundle.files]
    assert paths == ["runtime/intent/classifier.py"]
    assert bundle.truncated is False  # missing file is not a truncation


def test_directory_path_is_ignored(tmp_path: Path) -> None:
    """A scope path pointing at a directory must not blow up."""
    repo = _make_repo(tmp_path)
    bundle = gather_context(repo, ["runtime/intent"])

    assert bundle.files == []
    assert bundle.truncated is False


# --- empty / degenerate -----------------------------------------------------


def test_empty_scope_returns_empty_bundle(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    bundle = gather_context(repo, [])

    assert bundle.files == []
    assert bundle.skills == []
    assert bundle.total_bytes == 0
    assert bundle.truncated is False


def test_no_skills_dir_does_not_crash(tmp_path: Path) -> None:
    """A repo without ``runtime/skills/catalog/`` returns no skill slices."""
    repo = tmp_path / "repo"
    (repo / "runtime").mkdir(parents=True)
    (repo / "runtime/foo.py").write_text("z\n", encoding="utf-8")

    bundle = gather_context(repo, ["runtime/foo.py"])

    assert [f.path for f in bundle.files] == ["runtime/foo.py"]
    assert bundle.skills == []


def test_models_are_frozen(tmp_path: Path) -> None:
    """Pydantic invariant: every slice is immutable at the boundary."""
    fs = FileSlice(path="a.py", content="x", bytes_total=1, was_truncated=False)
    ss = SkillSlice(
        path="s.yaml", content="id: s\n", bytes_total=8, was_truncated=False,
    )
    bundle = ContextBundle(
        files=[fs], skills=[ss], truncated=False, total_bytes=9,
    )
    with pytest.raises((TypeError, ValueError)):
        bundle.total_bytes = 0  # type: ignore[misc]
