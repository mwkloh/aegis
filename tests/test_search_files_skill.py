"""The search_files skill must tell a model that `pattern` is a filename glob.

`files_search` matches `pattern` against file NAMES, so a bare word such as
`CT-001` finds only a file called exactly that. Measured: with the same skill text,
a model sent `*CT-001*` 9/9 times through native tools but `CT-001` 7/9 times through
the planner prompt. The description is the one place both arms read, so it has to
spell the wildcard rule out.
"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_SKILL = Path(__file__).resolve().parents[1] / "runtime/skills/_bundle/search_files/skill.yaml"


def _pattern_description() -> str:
    spec = yaml.safe_load(_SKILL.read_text(encoding="utf-8"))
    return spec["args_schema"]["properties"]["pattern"].get("description", "")


def test_pattern_argument_explains_wildcards() -> None:
    text = _pattern_description()
    assert "*" in text
    assert "name" in text.lower()


def test_pattern_argument_warns_that_a_bare_word_is_not_a_substring() -> None:
    assert "exactly" in _pattern_description().lower()
