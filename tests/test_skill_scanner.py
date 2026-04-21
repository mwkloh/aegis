from __future__ import annotations

from typing import Any

import pytest

from runtime.skills.registry import SkillDescriptor
from runtime.skills.scanner import scan_descriptor

pytestmark = pytest.mark.unit


def _descriptor(*, tools: list[dict[str, Any]] | None = None, **overrides: Any) -> SkillDescriptor:
    data: dict[str, Any] = {
        "id": "skill_x",
        "version": "0.1.0",
        "description": "A test skill.",
        "intents": ["skill_x"],
        "tool": "skill_x",
        "args_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
        "tools": tools or [],
    }
    data.update(overrides)
    return SkillDescriptor(**data)  # type: ignore[arg-type]


# --- clean descriptors --------------------------------------------------------


def test_clean_descriptor_has_no_findings() -> None:
    desc = _descriptor(
        tools=[
            {
                "name": "search",
                "argv_template": ["aegis", "vault", "search", "--query", "{query}"],
                "timeout_ms": 5_000,
                "allow_net": False,
            }
        ]
    )
    assert scan_descriptor(desc) == []


def test_empty_tools_list_has_no_findings() -> None:
    desc = _descriptor(tools=[])
    assert scan_descriptor(desc) == []


# --- shell-binary blocks ------------------------------------------------------


def test_shell_binary_as_argv0_blocks() -> None:
    desc = _descriptor(
        tools=[
            {
                "name": "shelly",
                "argv_template": ["sh", "-c", "echo hi"],
            }
        ]
    )
    findings = scan_descriptor(desc)
    codes = {f.code for f in findings}
    assert "shell_binary" in codes
    shell_block = next(f for f in findings if f.code == "shell_binary")
    assert shell_block.severity == "block"
    assert shell_block.tool == "shelly"


def test_shell_binary_with_full_path_also_blocks() -> None:
    desc = _descriptor(
        tools=[
            {
                "name": "shelly",
                "argv_template": ["/bin/bash", "-c", "echo hi"],
            }
        ]
    )
    findings = [f for f in scan_descriptor(desc) if f.code == "shell_binary"]
    assert len(findings) == 1
    assert findings[0].severity == "block"


# --- sensitive binary warnings ------------------------------------------------


def test_sensitive_binary_warns() -> None:
    desc = _descriptor(
        tools=[
            {"name": "nuke", "argv_template": ["rm", "-rf", "/tmp/scratch"]}
        ]
    )
    sensitive = [f for f in scan_descriptor(desc) if f.code == "sensitive_binary"]
    assert len(sensitive) == 1
    assert sensitive[0].severity == "warn"


def test_curl_warns_even_without_urls() -> None:
    desc = _descriptor(
        tools=[
            {"name": "fetch", "argv_template": ["curl", "--version"], "allow_net": True}
        ]
    )
    codes = {f.code for f in scan_descriptor(desc)}
    assert "sensitive_binary" in codes


# --- absolute-path warning ----------------------------------------------------


def test_absolute_path_outside_system_dirs_warns() -> None:
    desc = _descriptor(
        tools=[
            {"name": "local", "argv_template": ["/Users/op/bin/tool", "--go"]}
        ]
    )
    codes = {f.code for f in scan_descriptor(desc)}
    assert "absolute_path" in codes


def test_absolute_path_in_usr_bin_is_fine() -> None:
    desc = _descriptor(
        tools=[
            {"name": "ok", "argv_template": ["/usr/bin/jq", "."]}
        ]
    )
    codes = {f.code for f in scan_descriptor(desc)}
    assert "absolute_path" not in codes


# --- shell metacharacters -----------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        "a; rm -rf /",
        "a && b",
        "a || b",
        "a | b",
        "$(whoami)",
        "`id`",
        "> out.txt",
        "a & b",
    ],
)
def test_shell_metacharacters_in_token_warn(token: str) -> None:
    desc = _descriptor(
        tools=[
            {"name": "meta", "argv_template": ["aegis", token]}
        ]
    )
    findings = [f for f in scan_descriptor(desc) if f.code == "shell_metachar"]
    assert findings, f"no shell_metachar finding for {token!r}"
    assert findings[0].severity == "warn"


def test_safe_tokens_dont_trigger_shell_metachar() -> None:
    desc = _descriptor(
        tools=[
            {
                "name": "search",
                "argv_template": ["aegis", "vault", "search", "--query", "{query}"],
            }
        ]
    )
    codes = {f.code for f in scan_descriptor(desc)}
    assert "shell_metachar" not in codes


# --- URL findings -------------------------------------------------------------


def test_url_without_allow_net_blocks() -> None:
    desc = _descriptor(
        tools=[
            {
                "name": "fetch",
                "argv_template": ["aegis", "fetch", "https://example.com/data"],
                "allow_net": False,
            }
        ]
    )
    blocking = [f for f in scan_descriptor(desc) if f.code == "url_without_allow_net"]
    assert len(blocking) == 1
    assert blocking[0].severity == "block"


def test_url_with_allow_net_is_info() -> None:
    desc = _descriptor(
        tools=[
            {
                "name": "fetch",
                "argv_template": ["aegis", "fetch", "https://example.com/data"],
                "allow_net": True,
            }
        ]
    )
    url_findings = [f for f in scan_descriptor(desc) if f.code == "url_present"]
    assert len(url_findings) == 1
    assert url_findings[0].severity == "info"


def test_localhost_url_is_exempt() -> None:
    desc = _descriptor(
        tools=[
            {
                "name": "fetch",
                "argv_template": ["aegis", "fetch", "http://localhost:8080/healthz"],
                "allow_net": False,
            }
        ]
    )
    codes = {f.code for f in scan_descriptor(desc)}
    assert "url_without_allow_net" not in codes


# --- timing + net declarations ------------------------------------------------


def test_long_timeout_is_info() -> None:
    desc = _descriptor(
        tools=[
            {
                "name": "slow",
                "argv_template": ["aegis", "slow"],
                "timeout_ms": 300_000,
            }
        ]
    )
    timing = [f for f in scan_descriptor(desc) if f.code == "long_timeout"]
    assert len(timing) == 1
    assert timing[0].severity == "info"


def test_allow_net_true_emits_info() -> None:
    desc = _descriptor(
        tools=[
            {"name": "net", "argv_template": ["aegis", "net"], "allow_net": True}
        ]
    )
    net_findings = [f for f in scan_descriptor(desc) if f.code == "allow_net"]
    assert len(net_findings) == 1
    assert net_findings[0].severity == "info"


# --- aggregation --------------------------------------------------------------


def test_findings_tag_the_originating_tool() -> None:
    desc = _descriptor(
        tools=[
            {"name": "safe", "argv_template": ["aegis", "safe"]},
            {"name": "bad", "argv_template": ["sh", "-c", "echo"]},
        ]
    )
    findings = scan_descriptor(desc)
    bad_findings = [f for f in findings if f.tool == "bad"]
    safe_findings = [f for f in findings if f.tool == "safe"]
    assert bad_findings
    assert not safe_findings


def test_is_blocking_helper() -> None:
    desc = _descriptor(
        tools=[{"name": "shelly", "argv_template": ["sh", "-c", "x"]}]
    )
    findings = scan_descriptor(desc)
    assert any(f.is_blocking() for f in findings)
