"""Verify Ollama, models, ~/.aegis/ layout, and pinned digests.

Phase 1: adds presence checks for the Tier 0 fast model and the Reflection
model in Ollama, and an optional OpenRouter ping when a key is present.
Missing reflection model is a **warning**, not a failure (see Phase 1 plan).

Phase 5: adds two ``services:`` rows for git — ``git:available`` (error
if missing — apply impossible) and ``git:repo_clean`` (warn if dirty
or not a repo, since apply will refuse but doctor itself still passes).
"""
from __future__ import annotations

import os
import shutil
import subprocess  # nosec
import sys
from pathlib import Path
from typing import Final

import httpx
from dotenv import load_dotenv

Severity = str  # "ok" | "warn" | "error"
Row = tuple[str, bool, str, Severity]

AEGIS_ROOT = Path.home() / ".aegis"
AEGIS_HOME = AEGIS_ROOT / "workspace"
DEFAULT_FAST_MODEL = "gemma4:e2b"
DEFAULT_REFLECTION_MODEL = "gemma4:e4b"
DEFAULT_SMART_MODEL = "minimax/minimax-m2.7"

REQUIRED_FILES = (
    AEGIS_HOME / "AGENTS.md",
    AEGIS_HOME / "USER.md",
    AEGIS_HOME / "IDENTITY.md",
    AEGIS_HOME / "SOUL.md",
    AEGIS_HOME / "HEARTBEAT.md",
    AEGIS_ROOT / "config.json",
    AEGIS_ROOT / ".env",
)


def _check_layout() -> list[Row]:
    rows: list[Row] = []
    for p in REQUIRED_FILES:
        ok = p.exists()
        rows.append((str(p), ok, "ok" if ok else "missing", "ok" if ok else "error"))
    return rows


_REQUIRED_ENV_KEYS: Final[tuple[str, ...]] = ("OPENROUTER_API_KEY", "MODEL_FAST")
_OPTIONAL_ENV_KEYS: Final[tuple[str, ...]] = (
    "OLLAMA_BASE_URL", "MODEL_SMART", "MODEL_REFLECTION", "MODEL_CODING",
)


def _check_env() -> list[Row]:
    rows: list[Row] = []
    env_path = AEGIS_ROOT / ".env"
    found: set[str] = set()
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, _ = line.partition("=")
            found.add(k.strip())
    for k in _REQUIRED_ENV_KEYS:
        present = k in found or k in os.environ
        rows.append((f".env:{k}", present, "ok" if present else "missing",
                     "ok" if present else "error"))
    for k in _OPTIONAL_ENV_KEYS:
        present = k in found or k in os.environ
        rows.append((f".env:{k}", present, "ok" if present else "unset (optional)",
                     "ok" if present else "warn"))
    return rows


def _ollama_base() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def _check_ollama() -> Row:
    base = _ollama_base()
    if not base.startswith(("http://", "https://")):
        return ("ollama:reachable", False,
                f"refusing non-http(s) base url: {base!r}", "error")
    url = f"{base.rstrip('/')}/api/tags"
    try:
        resp = httpx.get(url, timeout=2.0)
        resp.raise_for_status()
        data = resp.json()
        names = [m.get("name", "") for m in data.get("models", [])]
        return ("ollama:reachable", True, f"{len(names)} model(s)", "ok")
    except (httpx.HTTPError, ValueError) as exc:
        return ("ollama:reachable", False, str(exc), "error")


def _list_ollama_models() -> list[str] | None:
    """Return installed model names, or None if Ollama is unreachable."""
    base = _ollama_base()
    if not base.startswith(("http://", "https://")):
        return None
    try:
        resp = httpx.get(f"{base.rstrip('/')}/api/tags", timeout=2.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    return [str(m.get("name", "")) for m in data.get("models", [])]


def _model_present(installed: list[str], wanted: str) -> bool:
    """A digestless `wanted` matches any tag of the same model family."""
    base = wanted.split(":", 1)[0]
    return any(name == wanted or name.split(":", 1)[0] == base for name in installed)


def _check_models() -> list[Row]:
    fast = os.environ.get("MODEL_FAST", DEFAULT_FAST_MODEL)
    reflection = os.environ.get("MODEL_REFLECTION", DEFAULT_REFLECTION_MODEL)
    installed = _list_ollama_models()
    if installed is None:
        return [
            (f"ollama:model:{fast}", False, "ollama unreachable", "error"),
            (f"ollama:model:{reflection}", False, "ollama unreachable", "warn"),
        ]
    rows: list[Row] = []
    for name, severity in ((fast, "error"), (reflection, "warn")):
        ok = _model_present(installed, name)
        detail = "installed" if ok else f"not installed ({severity})"
        rows.append((f"ollama:model:{name}", ok, detail, "ok" if ok else severity))
    return rows


def _check_reflection_writable() -> Row:
    """Plane 2 needs to be able to create its own output directory."""
    target = AEGIS_HOME / "reflection"
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return ("reflection:writable", False, str(exc), "warn")
    return ("reflection:writable", True, str(target), "ok")


def _check_improvement_writable() -> Row:
    """Plane 3 needs to be able to create its own governance directory."""
    target = AEGIS_HOME / "improvement"
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return ("improvement:writable", False, str(exc), "warn")
    return ("improvement:writable", True, str(target), "ok")


def _check_coding_harness_writable() -> Row:
    """Phase 4 — drafts land in `<workspace>/coding_harness/diffs/`."""
    target = AEGIS_HOME / "coding_harness" / "diffs"
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return ("coding_harness:writable", False, str(exc), "warn")
    return ("coding_harness:writable", True, str(target), "ok")


def _check_openrouter() -> Row | None:
    """Optional reachability ping. Returns None when no key configured (skip row)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not base.startswith("https://"):
        return ("openrouter:reachable", False,
                f"refusing non-https base url: {base!r}", "warn")
    try:
        resp = httpx.get(
            f"{base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=3.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return ("openrouter:reachable", False, str(exc), "warn")
    return ("openrouter:reachable", True, "ok", "ok")


def _resolved_coding_model() -> str | None:
    """`MODEL_CODING` if set, else `MODEL_SMART` (matches `_coerce` fallback)."""
    return os.environ.get("MODEL_CODING") or os.environ.get("MODEL_SMART")


def _check_openrouter_coding_model() -> Row | None:
    """Probe the OpenRouter catalog for the configured coding model id.

    Returns None when no key/model is configured. Catches a common foot-gun:
    a leading ``openrouter/`` prefix on the model id (which yields 400 from
    ``/chat/completions`` with no hint about why).
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    model = _resolved_coding_model()
    if not key or not model:
        return None
    label = f"openrouter:coding_model:{model}"
    if model.startswith("openrouter/"):
        return (label, False,
                "invalid 'openrouter/' prefix — use '<vendor>/<model>'", "warn")
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not base.startswith("https://"):
        return (label, False, f"refusing non-https base url: {base!r}", "warn")
    try:
        resp = httpx.get(
            f"{base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=3.0,
        )
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return (label, False, f"catalog fetch failed: {exc}", "warn")
    items = body.get("data", []) if isinstance(body, dict) else []
    ids = {str(item.get("id", "")) for item in items if isinstance(item, dict)}
    if model in ids:
        return (label, True, "in catalog", "ok")
    return (label, False, "not in OpenRouter catalog (will 400 on chat)", "warn")


def _check_git() -> list[Row]:
    """Two rows under ``services:`` — git availability and repo cleanliness.

    Phase 5 needs ``git`` on PATH for ``make apply``; missing → error.
    The cleanliness probe is informational: ``apply_patch`` will refuse
    on a dirty tree anyway, so a dirty cwd is a warning, not a failure.
    """
    git = shutil.which("git")
    if git is None:
        return [
            ("git:available", False, "not on PATH", "error"),
            ("git:repo_clean", False, "git missing — cannot probe", "warn"),
        ]
    rows: list[Row] = []
    try:
        ver = subprocess.run(  # nosec
            [git, "--version"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        rows.append(("git:available", False, f"probe failed: {exc}", "error"))
        rows.append(("git:repo_clean", False, "git probe failed", "warn"))
        return rows
    if ver.returncode != 0:
        rows.append(
            ("git:available", False,
             (ver.stderr or ver.stdout).strip() or f"exit {ver.returncode}",
             "error"),
        )
        rows.append(("git:repo_clean", False, "git probe failed", "warn"))
        return rows
    rows.append(("git:available", True, ver.stdout.strip().splitlines()[0], "ok"))
    rows.append(_check_git_repo_clean(git))
    return rows


def _check_git_repo_clean(git: str) -> Row:
    """``git status --porcelain`` — only meaningful inside a repo."""
    try:
        inside = subprocess.run(  # nosec
            [git, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return ("git:repo_clean", False, f"probe failed: {exc}", "warn")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return ("git:repo_clean", False, "cwd is not inside a git repo", "warn")
    try:
        status = subprocess.run(  # nosec
            [git, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return ("git:repo_clean", False, f"status probe failed: {exc}", "warn")
    if status.returncode != 0:
        return ("git:repo_clean", False,
                (status.stderr or status.stdout).strip() or
                f"exit {status.returncode}", "warn")
    if status.stdout.strip():
        n = len([ln for ln in status.stdout.splitlines() if ln.strip()])
        return ("git:repo_clean", False,
                f"working tree dirty ({n} change(s))", "warn")
    return ("git:repo_clean", True, "clean", "ok")


_GLYPH = {"ok": "\u2713", "warn": "\u26a0", "error": "\u2717"}


def _print(rows: list[Row]) -> None:
    for name, _ok, detail, severity in rows:
        mark = _GLYPH.get(severity, "?")
        print(f"  {mark} {name:<40} {detail}")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    # Mirror runtime/config.py: load ~/.aegis/.env into os.environ (without
    # clobbering anything already exported in the shell). Without this, the
    # openrouter ping rows skip because OPENROUTER_API_KEY only lives in .env.
    env_path = AEGIS_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    print("AEGIS doctor")
    print("layout:")
    layout = _check_layout()
    _print(layout)
    print("env:")
    env = _check_env()
    _print(env)
    print("services:")
    services: list[Row] = [_check_ollama()]
    or_row = _check_openrouter()
    if or_row is not None:
        services.append(or_row)
    or_model_row = _check_openrouter_coding_model()
    if or_model_row is not None:
        services.append(or_model_row)
    services.extend(_check_git())
    _print(services)
    print("models:")
    models = _check_models()
    _print(models)
    print("reflection:")
    reflection = [_check_reflection_writable()]
    _print(reflection)
    print("improvement:")
    improvement = [_check_improvement_writable()]
    _print(improvement)
    print("coding_harness:")
    coding_harness = [_check_coding_harness_writable()]
    _print(coding_harness)
    all_rows = (
        *layout, *env, *services, *models, *reflection, *improvement, *coding_harness,
    )
    errors = [r for r in all_rows if r[3] == "error"]
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
