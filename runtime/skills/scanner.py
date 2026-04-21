"""Descriptor safety scanner.

Phase 8 §C5. Before an operator-authored skill lands in the active
catalog, we run a deterministic scan against its declared tools.

The goal isn't to detect *every* malicious pattern — the tool harness
(§C3) is the real defence: argv-only spawn, bounded runtime, host
denial for network. The scanner exists to flag descriptors that
*look* like they're trying to bypass those defences, so an operator
doesn't confirm-install something they'll regret.

Design:
* **Deterministic first.** No LLM call in this module. Every finding
  has a rule id, a severity, and a human-readable message. Future
  LLM augmentation can be layered on top without changing the
  output contract.
* **Advisory, not enforcing.** The scanner returns findings; the
  installer decides whether ``block``-severity findings halt
  confirmation. That split keeps the scanner reusable (CLI tool,
  CI, or pre-install review).
* **No side effects.** ``scan_descriptor`` reads a parsed
  ``SkillDescriptor`` and returns a list. The installer is
  responsible for copying files, running the scan, and reporting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from runtime.skills.registry import SkillDescriptor, ToolSpec

ScanSeverity = Literal["info", "warn", "block"]

# Shell metacharacters that have no meaning in argv-only execution but
# signal operator confusion — someone copy-pasted a shell command
# instead of splitting into tokens. Flagged as ``warn`` because the
# harness will refuse them at resolve time anyway; they just shouldn't
# land in a catalog.
_SHELL_METACHAR_RE = re.compile(r"[;&|`$<>]|\$\(|&&|\|\|")

# HTTP(S) / ws(s) URLs in argv are a strong signal the tool wants
# network access. Combined with ``allow_net=False``, that's a block.
_URL_RE = re.compile(r"\b(?:https?|wss?|ftp)://\S+", re.IGNORECASE)

# "Local" hosts the harness considers benign even under allow_net=False
# when a future egress shim gates by hostname. Keep conservative.
# Host strings used only for URL detection below — never bound to a socket.
_LOCAL_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "::1", "0.0.0.0"}  # noqa: S104  # nosec B104
)

# Binaries that, when declared as argv[0], are themselves shells —
# running them reintroduces the exact shell-injection surface argv
# tokens are meant to eliminate.
_SHELL_BINARIES = frozenset(
    {"sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh", "powershell", "pwsh"}
)

# Known-dangerous argv[0] names. Not exhaustive; the harness's
# sandboxing is the real guard. This list is for operator legibility.
_DANGEROUS_BINARIES = frozenset(
    {"rm", "dd", "mkfs", "shutdown", "reboot", "kill", "pkill", "curl", "wget"}
)

# Tools over this timeout block the dispatcher; surface as an "info"
# finding so operators move them to the long-running surface.
_LONG_TIMEOUT_MS = 120_000


@dataclass(frozen=True)
class ScanFinding:
    """One issue found in a descriptor. Ordering of findings is stable."""

    severity: ScanSeverity
    code: str
    message: str
    tool: str = ""  # empty when the finding is against the whole descriptor

    def is_blocking(self) -> bool:
        return self.severity == "block"


def scan_descriptor(descriptor: SkillDescriptor) -> list[ScanFinding]:
    """Run every rule against ``descriptor``. Returns findings in rule order.

    An empty list means "no concerns" — not "verified safe". The
    harness + operator confirmation remain the source of truth.
    """
    findings: list[ScanFinding] = []
    for tool in descriptor.tools:
        findings.extend(_scan_tool(tool))
    return findings


def _scan_tool(tool: ToolSpec) -> list[ScanFinding]:
    findings: list[ScanFinding] = []

    # --- argv[0] guards -----------------------------------------------------

    if not tool.argv_template:
        return findings  # pydantic prevents this, but be defensive.
    head = tool.argv_template[0]
    # Strip any directory prefix so `/bin/sh` and `sh` both match.
    head_basename = head.rsplit("/", 1)[-1].lower()
    if head_basename in _SHELL_BINARIES:
        findings.append(
            ScanFinding(
                severity="block",
                code="shell_binary",
                message=(
                    f"argv[0]={head!r} is a shell; argv-only execution "
                    "defeats the point if a shell re-interprets the rest."
                ),
                tool=tool.name,
            )
        )
    elif head_basename in _DANGEROUS_BINARIES:
        findings.append(
            ScanFinding(
                severity="warn",
                code="sensitive_binary",
                message=(
                    f"argv[0]={head!r} is frequently destructive or performs "
                    "network I/O. Confirm this skill really needs it."
                ),
                tool=tool.name,
            )
        )
    if head.startswith("/") and not head.startswith("/usr/") and not head.startswith("/bin/"):
        # Absolute paths into operator-writable locations (home dirs,
        # /tmp, etc.) are a common self-install footgun.
        findings.append(
            ScanFinding(
                severity="warn",
                code="absolute_path",
                message=(
                    f"argv[0]={head!r} is an absolute path outside system "
                    "binary directories. Prefer a PATH-resolved name so "
                    "deployment is reproducible."
                ),
                tool=tool.name,
            )
        )

    # --- per-token scans ----------------------------------------------------

    for idx, token in enumerate(tool.argv_template):
        if _SHELL_METACHAR_RE.search(token):
            findings.append(
                ScanFinding(
                    severity="warn",
                    code="shell_metachar",
                    message=(
                        f"argv[{idx}]={token!r} contains shell metacharacters. "
                        "argv tokens are literal — if you want piping, use a "
                        "wrapper script instead."
                    ),
                    tool=tool.name,
                )
            )
        for match in _URL_RE.finditer(token):
            url = match.group(0)
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            if host in _LOCAL_HOSTS:
                continue
            if not tool.allow_net:
                findings.append(
                    ScanFinding(
                        severity="block",
                        code="url_without_allow_net",
                        message=(
                            f"argv[{idx}] references {url!r} but "
                            f"allow_net=False. Set allow_net: true "
                            "explicitly if this tool genuinely needs network."
                        ),
                        tool=tool.name,
                    )
                )
            else:
                findings.append(
                    ScanFinding(
                        severity="info",
                        code="url_present",
                        message=(
                            f"argv[{idx}] references {url!r}; allow_net is "
                            "true so this is permitted. Confirm the host is "
                            "expected."
                        ),
                        tool=tool.name,
                    )
                )

    # --- timing sanity ------------------------------------------------------

    if tool.timeout_ms > _LONG_TIMEOUT_MS:
        findings.append(
            ScanFinding(
                severity="info",
                code="long_timeout",
                message=(
                    f"timeout_ms={tool.timeout_ms} exceeds 120s. Long-running "
                    "tools block the dispatcher; consider moving to the "
                    "long-running surface."
                ),
                tool=tool.name,
            )
        )

    # --- net declaration ----------------------------------------------------

    if tool.allow_net:
        findings.append(
            ScanFinding(
                severity="info",
                code="allow_net",
                message=(
                    "allow_net=True. This skill may perform network I/O; "
                    "confirm the operator explicitly wants that."
                ),
                tool=tool.name,
            )
        )

    return findings


__all__ = [
    "ScanFinding",
    "ScanSeverity",
    "scan_descriptor",
]
