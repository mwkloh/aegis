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

DEFAULT_FAST_MODEL = "gemma4:e2b"
DEFAULT_REFLECTION_MODEL = "gemma4:e4b"
DEFAULT_SMART_MODEL = "minimax/minimax-m2.7"
# Mirrors `ModelConfig.smart_local` in runtime/config.py.
DEFAULT_SMART_LOCAL_MODEL = "llama3.1:8b"

_CANON_FILES = ("AGENTS.md", "USER.md", "IDENTITY.md", "SOUL.md", "HEARTBEAT.md")


def _aegis_root() -> Path:
    """Same resolution as `runtime.config`: `AEGIS_ROOT`, else `~/.aegis`."""
    return Path(os.environ.get("AEGIS_ROOT", str(Path.home() / ".aegis"))).expanduser()


def _aegis_home() -> Path:
    return Path(os.environ.get("AEGIS_HOME", str(_aegis_root() / "workspace"))).expanduser()


def _check_layout() -> list[Row]:
    root, home = _aegis_root(), _aegis_home()
    required = (*(home / name for name in _CANON_FILES), root / "config.json", root / ".env")
    rows: list[Row] = []
    for p in required:
        ok = p.exists()
        rows.append((str(p), ok, "ok" if ok else "missing", "ok" if ok else "error"))
    return rows


# `OPENROUTER_API_KEY` is only required when the smart tier is pinned to OpenRouter; an
# all-local Ollama setup has no use for it.
REQUIRED_ENV_KEYS: Final[tuple[str, ...]] = ("MODEL_FAST",)
_OPTIONAL_ENV_KEYS: Final[tuple[str, ...]] = (
    "OLLAMA_BASE_URL", "MODEL_SMART_LOCAL", "MODEL_REFLECTION", "MODEL_CODING",
)


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        values[k.strip()] = v.split("#", 1)[0].strip()
    return values


def _smart_provider(lookup: dict[str, str]) -> str:
    return (lookup.get("MODEL_SMART_PROVIDER") or "ollama").strip().lower()


def _check_env() -> list[Row]:
    # The shell wins over the file, as in `runtime.config`.
    lookup = {**_read_env_file(_aegis_root() / ".env"), **os.environ}
    openrouter = _smart_provider(lookup) == "openrouter"
    rows: list[Row] = []
    for k in REQUIRED_ENV_KEYS:
        present = bool(lookup.get(k))
        rows.append((f".env:{k}", present, "ok" if present else "missing",
                     "ok" if present else "error"))
    if openrouter:
        present = bool(lookup.get("OPENROUTER_API_KEY"))
        rows.append((".env:OPENROUTER_API_KEY", present,
                     "ok" if present else "empty (required when MODEL_SMART_PROVIDER=openrouter)",
                     "ok" if present else "error"))
    optional = _OPTIONAL_ENV_KEYS if openrouter else (*_OPTIONAL_ENV_KEYS, "OPENROUTER_API_KEY")
    for k in optional:
        present = bool(lookup.get(k))
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
    """A tagged `wanted` must match exactly (`gemma4:e2b` is not `gemma4:e4b`).

    An untagged `wanted` (`llama3.2`) matches any installed tag of that model.
    """
    if ":" in wanted:
        return wanted in installed
    return any(name.split(":", 1)[0] == wanted for name in installed)


def _wanted_models() -> list[tuple[str, str]]:
    """(model, severity-if-missing) for every Ollama model this setup will call.

    The smart-tier local model is only needed when the smart tier is pinned to Ollama.
    A model named twice keeps its first (stricter) severity.
    """
    wanted = [(os.environ.get("MODEL_FAST", DEFAULT_FAST_MODEL), "error")]
    if _smart_provider(dict(os.environ)) == "ollama":
        wanted.append(
            (os.environ.get("MODEL_SMART_LOCAL", DEFAULT_SMART_LOCAL_MODEL), "error")
        )
    wanted.append((os.environ.get("MODEL_REFLECTION", DEFAULT_REFLECTION_MODEL), "warn"))
    deduped: dict[str, str] = {}
    for name, severity in wanted:
        deduped.setdefault(name, severity)
    return list(deduped.items())


def _check_models() -> list[Row]:
    wanted = _wanted_models()
    installed = _list_ollama_models()
    if installed is None:
        return [
            (f"ollama:model:{name}", False, "ollama unreachable (start it: ollama serve)", sev)
            for name, sev in wanted
        ]
    rows: list[Row] = []
    for name, severity in wanted:
        ok = _model_present(installed, name)
        detail = "installed" if ok else f"not installed — run: ollama pull {name}"
        rows.append((f"ollama:model:{name}", ok, detail, "ok" if ok else severity))
    return rows


_USERNAME_PLACEHOLDER = "YOUR_USERNAME"


def _check_skill_placeholders() -> Row:
    """Warn while the bundled skills' example paths still say `/Users/YOUR_USERNAME`.

    The examples are model-visible prompt text; an unfilled placeholder can be copied
    into a tool call verbatim. A warning, not an error: nothing is broken until a model
    does that, and the README documents the one-line fix.
    """
    catalog = _aegis_home() / "skills"
    stale = sorted(
        p.parent.name
        for p in catalog.glob("*/skill.yaml")
        if _USERNAME_PLACEHOLDER in p.read_text(encoding="utf-8")
    )
    if not stale:
        return ("skills:example_paths", True, "ok", "ok")
    fix = f"sed -i '' \"s#{_USERNAME_PLACEHOLDER}#$(id -un)#g\" {catalog}/*/skill.yaml"
    return (
        "skills:example_paths",
        False,
        f"{len(stale)} skill(s) still show {_USERNAME_PLACEHOLDER} in example paths "
        f"(Linux: drop the '' after -i). Fix: {fix}",
        "warn",
    )


def _check_reflection_writable() -> Row:
    """Plane 2 needs to be able to create its own output directory."""
    target = _aegis_home() / "reflection"
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
    target = _aegis_home() / "improvement"
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
    target = _aegis_home() / "coding_harness" / "diffs"
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
    env_path = _aegis_root() / ".env"
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
    print("skills:")
    skills = [_check_skill_placeholders()]
    _print(skills)
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
        *layout, *env, *services, *models, *skills, *reflection, *improvement,
        *coding_harness,
    )
    errors = [r for r in all_rows if r[3] == "error"]
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
