# tests/test_files_harness.py
"""Unit tests for make_files_tools harness callables."""
from __future__ import annotations

from typing import Any

import pytest

from runtime.files.client import DirEntry, FileInfo, FileTooBig, PathDenied
from runtime.harness.tools.files_tool import make_files_tools

pytestmark = pytest.mark.unit

_NOW = "2026-04-22T00:00:00+00:00"


class _StubClient:
    def __init__(self) -> None:
        self.entries: list[DirEntry] = []
        self.content: str = "content"
        self.file_info = FileInfo(
            path="/x/doc.txt", type="file", size=7,
            created=_NOW, modified=_NOW, accessed=_NOW, mode="100644",
        )
        self.search_results: list[str] = []
        self.write_result: dict[str, Any] = {
            "path": "/x/written.txt", "bytes_written": 7, "overwrote": False,
        }
        self.raise_on_next: Exception | None = None

    def _maybe_raise(self) -> None:
        if self.raise_on_next is not None:
            exc = self.raise_on_next
            self.raise_on_next = None
            raise exc

    def list_dir(self, path: str, *, recursive: bool = False) -> list[DirEntry]:
        self._maybe_raise()
        return self.entries

    def read_file(self, path: str) -> str:
        self._maybe_raise()
        return self.content

    def stat(self, path: str) -> FileInfo:
        self._maybe_raise()
        return self.file_info

    def search(self, directory: str, pattern: str, *, kind: str = "any") -> list[str]:
        self._maybe_raise()
        return self.search_results

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        self._maybe_raise()
        return self.write_result


@pytest.fixture
def stub() -> _StubClient:
    return _StubClient()


@pytest.fixture
def tools(stub: _StubClient) -> dict[str, Any]:
    return make_files_tools(stub)  # type: ignore[arg-type]


def test_files_list_returns_formatted_entries(tools: dict, stub: _StubClient) -> None:
    stub.entries = [DirEntry(name="a.txt", path="/x/a.txt", type="file", size=10, modified=_NOW)]
    result = tools["files_list"]({"path": "/x"})
    assert "[f]" in result["result"]
    assert "a.txt" in result["result"]


def test_files_list_empty_directory(tools: dict) -> None:
    result = tools["files_list"]({"path": "/x"})
    assert "empty" in result["result"].lower()


def test_files_read_returns_content(tools: dict, stub: _StubClient) -> None:
    stub.content = "hello"
    result = tools["files_read"]({"path": "/x/doc.txt"})
    assert result["result"] == "hello"


# --- files_read pages a long file instead of silently cutting it -----------------
# The old behaviour cut at 3,500 characters with a bare "… (truncated)", so a model
# could not tell a file slightly too long from one 80 KB long, and had no way to see
# the rest. A bigger cap alone cannot fix that: no cap makes an 81 KB file fit. Paging
# lets the model fetch the next chunk on demand, paying for it only then.
# See docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md section 14.


def test_files_read_pages_a_long_file_and_says_how_to_continue(
    tools: dict, stub: _StubClient
) -> None:
    stub.content = "x" * 5000
    out = tools["files_read"]({"path": "/x/doc.txt"})["result"]

    assert out.startswith("x" * 3000)
    assert "x" * 3001 not in out
    assert "showing chars 1-3,000 of 5,000" in out
    assert "offset=3000" in out


def test_files_read_offset_continues_where_the_notice_said(tools: dict, stub: _StubClient) -> None:
    stub.content = "".join(str(i % 10) for i in range(5000))
    out = tools["files_read"]({"path": "/x/doc.txt", "offset": 3000})["result"]

    assert out.startswith(stub.content[3000:5000])
    assert "showing chars 3,001-5,000 of 5,000" in out
    assert "end of file" in out
    assert "offset=" not in out


def test_paging_reassembles_the_whole_file(tools: dict, stub: _StubClient) -> None:
    stub.content = "".join(chr(97 + i % 26) for i in range(8123))
    pieces: list[str] = []
    offset = 0
    for _ in range(20):
        out = tools["files_read"]({"path": "/x/doc.txt", "offset": offset})["result"]
        body, _, notice = out.partition("\n…[")
        pieces.append(body)
        if "offset=" not in notice:
            break
        offset = int(notice.split("offset=")[1].split(" ")[0].rstrip("]"))

    assert "".join(pieces) == stub.content


def test_a_file_exactly_one_page_long_has_no_notice(tools: dict, stub: _StubClient) -> None:
    stub.content = "y" * 3000
    out = tools["files_read"]({"path": "/x/doc.txt"})["result"]

    assert out == "y" * 3000


def test_files_read_offset_past_the_end_says_so(tools: dict, stub: _StubClient) -> None:
    stub.content = "z" * 5000
    out = tools["files_read"]({"path": "/x/doc.txt", "offset": 9000})["result"]

    assert "past the end" in out
    assert "5,000" in out


@pytest.mark.parametrize("bad", [-1, "abc", 2.5, True])
def test_files_read_rejects_a_bad_offset(tools: dict, stub: _StubClient, bad: object) -> None:
    with pytest.raises(ValueError, match="offset"):
        tools["files_read"]({"path": "/x/doc.txt", "offset": bad})


def test_a_null_offset_means_from_the_start(tools: dict, stub: _StubClient) -> None:
    """Optional arguments are often emitted as null; that is not an error."""
    stub.content = "q" * 100
    assert tools["files_read"]({"path": "/x/doc.txt", "offset": None})["result"] == "q" * 100


def test_files_read_accepts_a_numeric_string_offset(tools: dict, stub: _StubClient) -> None:
    """Models sometimes emit integers as strings."""
    stub.content = "a" * 4000
    out = tools["files_read"]({"path": "/x/doc.txt", "offset": "3000"})["result"]

    assert out.startswith("a" * 1000)


def test_the_bundled_read_file_skill_advertises_offset() -> None:
    """A model only pages if the argument schema it is shown includes `offset`; a skill
    that lacks it silently disables the feature however well the tool works."""
    from pathlib import Path

    from runtime.skills.registry import SkillRegistry

    bundle = Path(__file__).resolve().parent.parent / "runtime" / "skills" / "_bundle"
    (read_file,) = [d for d in SkillRegistry.from_directory(bundle).all() if d.tool == "files_read"]

    offset = read_file.args_schema["properties"]["offset"]
    assert offset["type"] == "integer"
    assert offset["minimum"] == 0
    assert "offset" not in read_file.args_schema["required"]
    assert "offset" in read_file.description


def test_a_page_and_its_notice_fit_inside_the_default_prompt_limit(
    tools: dict, stub: _StubClient
) -> None:
    """If the planner clipped the page, it would cut off the very notice that tells
    the model how to continue."""
    from runtime.harness.limits import DEFAULT_TOOL_RESULT_CHARS

    stub.content = "x" * 200_000
    out = tools["files_read"]({"path": "/x/doc.txt", "offset": 123_456})["result"]

    assert len(out) <= DEFAULT_TOOL_RESULT_CHARS


def test_files_stat_returns_key_value(tools: dict) -> None:
    result = tools["files_stat"]({"path": "/x/doc.txt"})
    assert "type" in result["result"]
    assert "size" in result["result"]


def test_files_search_returns_paths(tools: dict, stub: _StubClient) -> None:
    stub.search_results = ["/x/report.pdf"]
    result = tools["files_search"]({"directory": "/x", "pattern": "*.pdf"})
    assert "/x/report.pdf" in result["result"]


def test_files_search_no_matches(tools: dict) -> None:
    result = tools["files_search"]({"directory": "/x", "pattern": "*.pdf"})
    assert "No matches" in result["result"]


def test_files_write_returns_confirmation(tools: dict, stub: _StubClient) -> None:
    stub.write_result = {"path": "/x/notes.txt", "bytes_written": 42, "overwrote": False}
    result = tools["files_write"]({"path": "/x/notes.txt", "content": "hello"})
    assert "42" in result["result"]
    assert "/x/notes.txt" in result["result"]


def test_files_write_signals_overwrite(tools: dict, stub: _StubClient) -> None:
    stub.write_result = {"path": "/x/notes.txt", "bytes_written": 42, "overwrote": True}
    result = tools["files_write"]({"path": "/x/notes.txt", "content": "hello"})
    assert "overwrote existing file" in result["result"]


def test_files_write_no_overwrite_signal_on_new_file(tools: dict, stub: _StubClient) -> None:
    stub.write_result = {"path": "/x/notes.txt", "bytes_written": 42, "overwrote": False}
    result = tools["files_write"]({"path": "/x/notes.txt", "content": "hello"})
    assert "overwrote" not in result["result"]


def test_files_write_path_denied_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = PathDenied("outside roots")
    with pytest.raises(RuntimeError, match="outside roots"):
        tools["files_write"]({"path": "/etc/passwd", "content": "pwned"})


def test_files_write_file_too_big_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = FileTooBig("too big")
    with pytest.raises(RuntimeError, match="too big"):
        tools["files_write"]({"path": "/x/huge.bin", "content": "x" * 10})


def test_path_denied_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = PathDenied("outside roots")
    with pytest.raises(RuntimeError, match="outside roots"):
        tools["files_read"]({"path": "/etc/passwd"})


def test_file_too_big_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = FileTooBig("too big")
    with pytest.raises(RuntimeError, match="too big"):
        tools["files_read"]({"path": "/x/huge.bin"})


def test_oserror_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = OSError("disk error")
    with pytest.raises(RuntimeError, match="disk error"):
        tools["files_list"]({"path": "/x"})


def test_make_files_tools_returns_six_callables(tools: dict) -> None:
    assert set(tools.keys()) == {
        "files_list",
        "files_read",
        "files_stat",
        "files_search",
        "files_open",
        "files_write",
    }
