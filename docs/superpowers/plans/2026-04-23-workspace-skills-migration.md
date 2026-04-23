# Workspace Skills Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the AEGIS skill catalog from the in-repo `runtime/skills/catalog/` path to the sovereign workspace path `~/.aegis/workspace/skills/`, using a directory-per-skill layout (`<skill_id>/skill.yaml` + optional bundled script), so that built-in skills like `morning_brief` live as canonical state outside the repo — matching the Phase 0 design pin and the atamai convention.

**Architecture:** Introduce a `SkillsConfig` block on `AegisConfig` that points at `~/.aegis/workspace/skills/` by default. Upgrade `SkillRegistry.from_directory` to scan `*/skill.yaml` (directory-per-skill) in addition to today's flat `*.yaml`. Add a new `{skill_dir}` placeholder to `argv_template` so a skill can invoke its co-located script by absolute path. Ship built-in descriptors as a seed bundle under `runtime/skills/_bundle/` that is copied into the workspace on first boot (idempotent; workspace wins on conflict). Skills whose implementation is a standalone script (`morning_brief`, `echo`) move their script into the skill directory; skills backed by repo subsystems (`tier2_compress`, `vault_reindex`, `reflection_sweep`, `macos-files`) keep `python -m runtime.<module>` argv and only the descriptor moves.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, pytest, existing `runtime.skills` + `runtime.config` modules. No new third-party dependencies.

---

## File Structure

**Create (new files):**
- `runtime/skills/bootstrap.py` — seed-bundle copier, invoked at startup.
- `runtime/skills/_bundle/` — directory holding source-of-truth copies of built-in skills. Each subdir is one skill: `skill.yaml` + optional `*.py`.
- `runtime/skills/_bundle/morning_brief/skill.yaml`
- `runtime/skills/_bundle/morning_brief/morning_brief.py`
- `runtime/skills/_bundle/echo/skill.yaml`
- `runtime/skills/_bundle/echo/echo.py`
- `runtime/skills/_bundle/tier2_compress/skill.yaml`
- `runtime/skills/_bundle/vault_reindex/skill.yaml`
- `runtime/skills/_bundle/reflection_sweep/skill.yaml`
- `runtime/skills/_bundle/list_files/skill.yaml`
- `runtime/skills/_bundle/read_file/skill.yaml`
- `runtime/skills/_bundle/search_files/skill.yaml`
- `runtime/skills/_bundle/file_info/skill.yaml`
- `runtime/skills/_bundle/ask_question/skill.yaml`
- `runtime/skills/_bundle/time_query/skill.yaml`
- `tests/test_skills_config.py` — config surface tests.
- `tests/test_skills_bootstrap.py` — seed-copy tests.
- `tests/test_skills_dir_loader.py` — directory-per-skill registry tests.

**Modify:**
- `runtime/config.py` — add `SkillsConfig` class + `skills: SkillsConfig` field on `AegisConfig`.
- `runtime/skills/registry.py` — `from_directory` scans `*/skill.yaml` plus legacy `*.yaml`; registry tracks `source_dir_of(id)`; descriptor validator exempts `{skill_dir}` placeholder.
- `runtime/skills/loader.py` — lazy loader understands both layouts.
- `runtime/chat/telegram/bot.py` — `build_intent_router`/`build_scheduler`/`build_skill_arg_resolver` read `cfg.skills.catalog_dir`; resolver injects `{skill_dir}` from registry; startup calls `seed_builtin_skills`.
- `runtime/chat/cli.py` — uses `cfg.skills.catalog_dir` instead of `CATALOG_DIR` constant.
- `runtime/reflection/cli.py` — same.
- `runtime/skills/installer.py` — `confirm_skill` promotes to workspace path, writes `<staging_dir>/<id>.yaml` into `<catalog_dir>/<id>/skill.yaml` (directory-per-skill shape).
- `runtime/scheduler/seed.py` — unchanged skill ids, but add docstring note about workspace location.
- `README.md` — update canonical-state section to mention `skills/`.
- `docs/PLAN_PHASE_0_AND_WALKING_SKELETON.md` — mark `skills/` no-longer-empty.

**Delete (at end, after migration verified):**
- `runtime/skills/catalog/` (entire directory — 11 YAML files).
- `runtime/skills/scripts/` (entire directory — `morning_brief.py`, `echo.py`, `tier2_compress.py`, `vault_reindex.py`).

**Existing tests that will need updates (expect churn here):**
- `tests/test_skill_registry.py`, `tests/test_skill_loader.py`, `tests/test_skill_installer.py`, `tests/test_intent_router.py`, `tests/test_scheduler_seed.py`, `tests/test_scheduler_engine.py`, `tests/test_scheduler_runner.py`, `tests/test_telegram_bot.py`, `tests/test_telegram_long_running.py`, `tests/test_morning_brief_script.py`, `tests/test_skill_catalog_recurring.py`, `tests/test_cron_run.py`, `tests/test_health_handler.py`.

---

## Task 1: Add `SkillsConfig` to `AegisConfig`

**Files:**
- Modify: `runtime/config.py`
- Test: `tests/test_skills_config.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_skills_config.py`:

```python
"""SkillsConfig — catalog directory + bundle-source knobs."""
from __future__ import annotations

import os
from pathlib import Path

from runtime.config import AegisConfig, SkillsConfig


def test_default_catalog_dir_is_under_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGIS_ROOT", str(tmp_path))
    monkeypatch.delenv("AEGIS_HOME", raising=False)
    cfg = AegisConfig()
    assert cfg.skills.catalog_dir == tmp_path / "workspace" / "skills"


def test_catalog_dir_respects_aegis_home(monkeypatch, tmp_path):
    custom = tmp_path / "custom-workspace"
    monkeypatch.setenv("AEGIS_HOME", str(custom))
    cfg = AegisConfig()
    assert cfg.skills.catalog_dir == custom / "skills"


def test_bundle_dir_points_at_repo():
    cfg = SkillsConfig()
    # Default bundle dir resolves relative to runtime/skills/_bundle
    assert cfg.bundle_dir.name == "_bundle"
    assert cfg.bundle_dir.parent.name == "skills"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skills_config.py -v`
Expected: FAIL — `SkillsConfig` / `cfg.skills` does not exist.

- [ ] **Step 3: Implement `SkillsConfig`**

Edit `runtime/config.py`. Add after `FilesConfig` class (before `AegisConfig`):

```python
def _bundle_dir() -> Path:
    """Absolute path to the built-in skill seed bundle inside the repo."""
    return Path(__file__).resolve().parent / "skills" / "_bundle"


class SkillsConfig(BaseModel):
    """Skill catalog location + seed-bundle source.

    ``catalog_dir`` is the active catalog — the directory the loader scans
    at runtime and the installer writes to on ``/skills confirm``. Defaults
    under the sovereign workspace so canonical state lives outside the repo.

    ``bundle_dir`` is the repo-shipped seed source. First-boot logic copies
    missing entries from here into ``catalog_dir``; operator edits to
    ``catalog_dir`` always win on subsequent boots (no silent overwrite).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_dir: Path = Field(default_factory=lambda: _aegis_home() / "skills")
    bundle_dir: Path = Field(default_factory=_bundle_dir)

    @field_validator("catalog_dir", "bundle_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser()
```

Then modify `AegisConfig` — add one line after the `files` field:

```python
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skills_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run full suite for regressions**

Run: `pytest -x -q`
Expected: PASS. No test should rely on `AegisConfig` rejecting a `skills` field.

- [ ] **Step 6: Commit**

```bash
git add runtime/config.py tests/test_skills_config.py
git commit -m "feat(config): add SkillsConfig with sovereign catalog_dir default"
```

---

## Task 2: `SkillRegistry` loads directory-per-skill layout and tracks source dirs

**Files:**
- Modify: `runtime/skills/registry.py:91-143`
- Test: `tests/test_skills_dir_loader.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_skills_dir_loader.py`:

```python
"""SkillRegistry.from_directory — directory-per-skill layout + source_dir tracking."""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.skills.registry import SkillRegistry


def _write_skill_yaml(path: Path, skill_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
id: {skill_id}
version: 0.1.0
description: Test skill {skill_id}.
intents: [{skill_id}]
tool: t
args_schema:
  type: object
  additionalProperties: false
requires_tier1: false
""".strip()
    )


def test_loads_directory_per_skill(tmp_path: Path) -> None:
    _write_skill_yaml(tmp_path / "alpha" / "skill.yaml", "alpha")
    _write_skill_yaml(tmp_path / "beta" / "skill.yaml", "beta")

    registry = SkillRegistry.from_directory(tmp_path)

    assert {d.id for d in registry.all()} == {"alpha", "beta"}


def test_source_dir_of_returns_skill_directory(tmp_path: Path) -> None:
    _write_skill_yaml(tmp_path / "alpha" / "skill.yaml", "alpha")

    registry = SkillRegistry.from_directory(tmp_path)

    assert registry.source_dir_of("alpha") == tmp_path / "alpha"
    assert registry.source_dir_of("missing") is None


def test_still_loads_flat_layout_for_back_compat(tmp_path: Path) -> None:
    # Legacy: <catalog>/<id>.yaml with no subdirectory.
    flat = tmp_path / "gamma.yaml"
    _write_skill_yaml(flat, "gamma")

    registry = SkillRegistry.from_directory(tmp_path)

    assert {d.id for d in registry.all()} == {"gamma"}
    # Flat-layout source_dir is the catalog dir itself (no per-skill home).
    assert registry.source_dir_of("gamma") == tmp_path


def test_directory_wins_on_duplicate_id(tmp_path: Path) -> None:
    _write_skill_yaml(tmp_path / "dup" / "skill.yaml", "dup")
    _write_skill_yaml(tmp_path / "dup.yaml", "dup")

    registry = SkillRegistry.from_directory(tmp_path)

    # Directory layout is the authoritative form; flat legacy file is ignored.
    assert registry.source_dir_of("dup") == tmp_path / "dup"


def test_missing_catalog_returns_empty_registry(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    registry = SkillRegistry.from_directory(missing)
    assert registry.all() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skills_dir_loader.py -v`
Expected: FAIL — `source_dir_of` does not exist; directory layout is not recognised.

- [ ] **Step 3: Update `SkillRegistry`**

Edit `runtime/skills/registry.py`. Replace the `SkillRegistry` class from line 111 to the end:

```python
class SkillRegistry:
    """In-memory index. Build at startup, never mutate at runtime."""

    def __init__(
        self,
        descriptors: list[SkillDescriptor],
        *,
        source_dirs: dict[str, Path] | None = None,
    ) -> None:
        self._by_id: dict[str, SkillDescriptor] = {d.id: d for d in descriptors}
        self._by_intent: dict[str, SkillDescriptor] = {}
        for d in descriptors:
            for intent in d.intents:
                # Deterministic: first descriptor to claim an intent wins.
                self._by_intent.setdefault(intent, d)
        self._source_dirs: dict[str, Path] = dict(source_dirs or {})

    @classmethod
    def from_directory(cls, catalog_dir: Path) -> SkillRegistry:
        """Load every skill under ``catalog_dir``.

        Recognised layouts, in priority order:

        1. **Directory-per-skill** — ``<catalog_dir>/<id>/skill.yaml``.
           This is the preferred sovereign layout. ``source_dir_of(id)``
           returns the per-skill directory so ``{skill_dir}`` placeholders
           in ``argv_template`` resolve to a co-located script.
        2. **Flat** — ``<catalog_dir>/<id>.yaml``. Legacy in-repo layout.
           Kept for back-compat during migration; ``source_dir_of`` returns
           the catalog directory itself.

        When both forms declare the same id, the directory form wins.
        """
        catalog_dir = Path(catalog_dir)
        if not catalog_dir.is_dir():
            return cls([])
        descriptors: list[SkillDescriptor] = []
        source_dirs: dict[str, Path] = {}
        seen: set[str] = set()

        for entry in sorted(catalog_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_yaml = entry / "skill.yaml"
            if not skill_yaml.is_file():
                continue
            descriptor = cls._load_one(skill_yaml)
            if descriptor.id in seen:
                continue
            descriptors.append(descriptor)
            source_dirs[descriptor.id] = entry
            seen.add(descriptor.id)

        for path in sorted(catalog_dir.glob("*.yaml")):
            descriptor = cls._load_one(path)
            if descriptor.id in seen:
                continue
            descriptors.append(descriptor)
            source_dirs[descriptor.id] = catalog_dir
            seen.add(descriptor.id)

        return cls(descriptors, source_dirs=source_dirs)

    @staticmethod
    def _load_one(path: Path) -> SkillDescriptor:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: skill descriptor must be a YAML mapping")
        return SkillDescriptor(**raw)

    def get(self, skill_id: str) -> SkillDescriptor | None:
        return self._by_id.get(skill_id)

    def for_intent(self, intent: str) -> SkillDescriptor | None:
        return self._by_intent.get(intent)

    def all(self) -> list[SkillDescriptor]:
        return list(self._by_id.values())

    def source_dir_of(self, skill_id: str) -> Path | None:
        """Directory that holds the descriptor — or ``None`` if unknown."""
        return self._source_dirs.get(skill_id)
```

- [ ] **Step 4: Run new test to verify it passes**

Run: `pytest tests/test_skills_dir_loader.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run existing registry tests**

Run: `pytest tests/test_skill_registry.py -v`
Expected: PASS — flat-layout fallback keeps existing fixtures green.

- [ ] **Step 6: Commit**

```bash
git add runtime/skills/registry.py tests/test_skills_dir_loader.py
git commit -m "feat(skills): registry understands directory-per-skill + tracks source_dir"
```

---

## Task 3: `{skill_dir}` placeholder support

**Files:**
- Modify: `runtime/skills/registry.py` — `_validate_tool_placeholders`
- Modify: `runtime/chat/telegram/bot.py:620-656` — `build_skill_arg_resolver`
- Test: add to `tests/test_skills_dir_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_skills_dir_loader.py`:

```python
from runtime.chat.telegram.bot import build_skill_arg_resolver
from runtime.config import AegisConfig


def test_skill_dir_placeholder_is_allowed_on_descriptor(tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        """
id: alpha
description: uses skill_dir
tool: t
args_schema:
  type: object
  additionalProperties: false
tools:
  - name: t
    argv_template: [python, "{skill_dir}/run.py"]
    timeout_ms: 1000
    allow_net: false
""".strip()
    )

    # Loading must succeed — {skill_dir} is not declared in args_schema
    # but it's an infrastructure placeholder, not a user arg.
    registry = SkillRegistry.from_directory(tmp_path)
    assert registry.get("alpha") is not None


def test_resolver_injects_skill_dir(tmp_path: Path, monkeypatch) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        """
id: alpha
description: uses skill_dir
tool: t
args_schema:
  type: object
  additionalProperties: false
tools:
  - name: t
    argv_template: [python, "{skill_dir}/run.py"]
    timeout_ms: 1000
    allow_net: false
""".strip()
    )

    registry = SkillRegistry.from_directory(tmp_path)
    resolver = build_skill_arg_resolver(
        AegisConfig(),
        registry=registry,
        python_executable="/usr/bin/python3",
    )
    descriptor = registry.get("alpha")
    assert descriptor is not None

    argv = resolver(descriptor)

    assert argv == ["/usr/bin/python3", f"{skill_dir}/run.py"]


def test_resolver_rejects_unknown_placeholder(tmp_path: Path) -> None:
    skill_dir = tmp_path / "beta"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        """
id: beta
description: references an undeclared placeholder
tool: t
args_schema:
  type: object
  properties:
    known:
      type: string
  required: [known]
  additionalProperties: false
tools:
  - name: t
    argv_template: [python, "{known}"]
    timeout_ms: 1000
    allow_net: false
""".strip()
    )

    registry = SkillRegistry.from_directory(tmp_path)
    resolver = build_skill_arg_resolver(AegisConfig(), registry=registry)
    descriptor = registry.get("beta")
    assert descriptor is not None

    # `known` is declared in args_schema but the resolver has no value for
    # it (not vault_root / skill_dir) → unresolvable → None.
    assert resolver(descriptor) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_skills_dir_loader.py -v`
Expected: FAIL — descriptor validator rejects `{skill_dir}` OR resolver signature does not accept `registry=`.

- [ ] **Step 3: Exempt `skill_dir` in the descriptor validator**

Edit `runtime/skills/registry.py`. Near the top, add:

```python
# Placeholders the harness injects automatically from the registry, not from
# the user-supplied args. Exempted from the "must appear in args_schema" check.
_INFRASTRUCTURE_PLACEHOLDERS: frozenset[str] = frozenset({"skill_dir"})
```

Then in `SkillDescriptor._validate_tool_placeholders` replace:

```python
            missing = spec.placeholders() - allowed
```

with:

```python
            missing = spec.placeholders() - allowed - _INFRASTRUCTURE_PLACEHOLDERS
```

- [ ] **Step 4: Update the resolver in `bot.py`**

Edit `runtime/chat/telegram/bot.py`. Replace `build_skill_arg_resolver` (lines 620–656):

```python
def build_skill_arg_resolver(
    cfg: AegisConfig,
    *,
    registry: SkillRegistry | None = None,
    python_executable: str | None = None,
) -> SkillArgResolver:
    """Return a resolver that turns a ``SkillDescriptor`` into a runnable argv.

    Known placeholders:

    * ``{vault_root}`` — from ``cfg.vault_indexing.vault_root``; returns
      ``None`` if the vault is not configured so the caller can report
      "skill not configured in this deployment" rather than crash.
    * ``{skill_dir}`` — injected from ``registry.source_dir_of(descriptor.id)``
      when a registry is supplied; skills can co-locate a script with their
      ``skill.yaml`` and reference it by absolute path.

    Any other placeholder returns ``None``.

    The leading ``python`` token in ``argv_template`` is swapped for the
    current interpreter (``sys.executable`` by default) so dispatches land
    on the same venv as ``/brief``.
    """
    python = python_executable if python_executable is not None else sys.executable

    def resolve(descriptor: SkillDescriptor) -> list[str] | None:
        if not descriptor.tools:
            return None
        spec = descriptor.tools[0]
        values: dict[str, str] = {}
        for name in spec.placeholders():
            if name == "vault_root":
                vr = cfg.vault_indexing.vault_root
                if vr is None:
                    return None
                values[name] = str(vr)
            elif name == "skill_dir":
                if registry is None:
                    return None
                source = registry.source_dir_of(descriptor.id)
                if source is None:
                    return None
                values[name] = str(source)
            else:
                return None
        resolved = [token.format_map(values) for token in spec.argv_template]
        if resolved and resolved[0] == "python":
            resolved[0] = python
        return resolved

    return resolve
```

- [ ] **Step 5: Run the new tests**

Run: `pytest tests/test_skills_dir_loader.py -v`
Expected: PASS (8 tests total including the earlier 5).

- [ ] **Step 6: Update every existing call site of `build_skill_arg_resolver`**

Find them: `grep -rn "build_skill_arg_resolver" runtime/ tests/`. Each call site now needs to pass `registry=`. Most existing tests build a `SkillRegistry` already; they just need to thread it through.

For `runtime/chat/telegram/bot.py:~1150-1250` (the `route_chat` / `build_application` flow), find where `skill_arg_resolver` is created and pass the registry that `build_intent_router` builds. If the registry isn't already in scope, create it once near the top of `build_application` and reuse.

Concretely inside `build_application` (grep for `build_skill_arg_resolver(` in bot.py), change:

```python
skill_arg_resolver = build_skill_arg_resolver(cfg)
```

to:

```python
registry = SkillRegistry.from_directory(cfg.skills.catalog_dir)
skill_arg_resolver = build_skill_arg_resolver(cfg, registry=registry)
```

(and thread the same `registry` into `build_intent_router` / `build_scheduler` in later tasks — for this commit, only thread it into the resolver.)

- [ ] **Step 7: Run the telegram bot tests**

Run: `pytest tests/test_telegram_bot.py tests/test_telegram_long_running.py -v`
Expected: PASS.

- [ ] **Step 8: Run full suite**

Run: `pytest -x -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add runtime/skills/registry.py runtime/chat/telegram/bot.py tests/test_skills_dir_loader.py
git commit -m "feat(skills): resolver injects {skill_dir} from registry"
```

---

## Task 4: Seed-bundle bootstrap — copy `_bundle/` → `catalog_dir` on first boot

**Files:**
- Create: `runtime/skills/bootstrap.py`
- Test: `tests/test_skills_bootstrap.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_skills_bootstrap.py`:

```python
"""Seed-bundle bootstrap — copies built-in skills into the workspace on first boot."""
from __future__ import annotations

from pathlib import Path

from runtime.skills.bootstrap import seed_builtin_skills


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_copies_missing_skills_from_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    catalog = tmp_path / "workspace" / "skills"
    _write(bundle / "alpha" / "skill.yaml", "id: alpha\ndescription: x\ntool: t\n")
    _write(bundle / "alpha" / "alpha.py", "print('alpha')\n")

    inserted = seed_builtin_skills(bundle_dir=bundle, catalog_dir=catalog)

    assert inserted == 1
    assert (catalog / "alpha" / "skill.yaml").read_text().startswith("id: alpha")
    assert (catalog / "alpha" / "alpha.py").read_text() == "print('alpha')\n"


def test_workspace_copy_wins_and_is_not_overwritten(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    catalog = tmp_path / "workspace" / "skills"
    _write(bundle / "alpha" / "skill.yaml", "id: alpha\ndescription: bundle\ntool: t\n")
    _write(catalog / "alpha" / "skill.yaml", "id: alpha\ndescription: operator\ntool: t\n")

    inserted = seed_builtin_skills(bundle_dir=bundle, catalog_dir=catalog)

    assert inserted == 0
    assert "operator" in (catalog / "alpha" / "skill.yaml").read_text()


def test_missing_bundle_is_a_noop(tmp_path: Path) -> None:
    catalog = tmp_path / "workspace" / "skills"
    inserted = seed_builtin_skills(
        bundle_dir=tmp_path / "does-not-exist", catalog_dir=catalog
    )
    assert inserted == 0
    assert not catalog.exists()


def test_catalog_dir_is_created_if_absent(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    catalog = tmp_path / "workspace" / "skills"
    _write(bundle / "alpha" / "skill.yaml", "id: alpha\ndescription: x\ntool: t\n")

    seed_builtin_skills(bundle_dir=bundle, catalog_dir=catalog)

    assert catalog.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skills_bootstrap.py -v`
Expected: FAIL — `runtime.skills.bootstrap` does not exist.

- [ ] **Step 3: Implement bootstrap**

Create `runtime/skills/bootstrap.py`:

```python
"""Seed the built-in skill bundle into the sovereign workspace on first boot.

Every skill shipped with this repo lives under ``runtime/skills/_bundle/`` as a
directory-per-skill tree (``<id>/skill.yaml`` plus any co-located scripts).
At startup we copy each bundle entry into ``catalog_dir`` (default
``~/.aegis/workspace/skills/``) unless the operator already has a directory
for that skill — in which case the workspace copy wins and we never touch it.

Design pins:

* **Idempotent.** Re-seeding is a no-op for any skill already present in
  the catalog. Safe to call on every bot start.
* **Workspace always wins.** An operator editing a descriptor in the
  workspace must survive every subsequent boot. This is the same contract
  as ``~/.aegis/workspace/*.md`` — repo code never overwrites it.
* **Never raises.** A copy failure for one skill is logged and skipped so
  the bot still boots with a partially-seeded catalog.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def seed_builtin_skills(*, bundle_dir: Path, catalog_dir: Path) -> int:
    """Copy missing built-in skills into ``catalog_dir``.

    Returns the count of skill directories actually created. Already-present
    workspace directories are untouched regardless of their content.
    """
    bundle_dir = Path(bundle_dir)
    catalog_dir = Path(catalog_dir)
    if not bundle_dir.is_dir():
        return 0

    catalog_dir.mkdir(parents=True, exist_ok=True)
    inserted = 0
    for entry in sorted(bundle_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        target = catalog_dir / entry.name
        if target.exists():
            continue
        try:
            shutil.copytree(entry, target)
            inserted += 1
        except OSError:
            logger.exception(
                "skills.seed_failed",
                extra={"skill_id": entry.name, "target": str(target)},
            )
    return inserted


__all__ = ["seed_builtin_skills"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skills_bootstrap.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add runtime/skills/bootstrap.py tests/test_skills_bootstrap.py
git commit -m "feat(skills): seed built-in skills into workspace on first boot"
```

---

## Task 5: Wire `cfg.skills.catalog_dir` through every catalog consumer

**Files:**
- Modify: `runtime/chat/telegram/bot.py:90`, `bot.py:659-671`, `bot.py:915-950` (scheduler builder), `bot.py:~1235` (build_application), and the startup path.
- Modify: `runtime/chat/cli.py:32`
- Modify: `runtime/reflection/cli.py:36`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_skills_bootstrap.py`:

```python
from runtime.chat.telegram.bot import build_intent_router
from runtime.config import AegisConfig


def test_build_intent_router_reads_cfg_catalog_dir(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "skills"
    (catalog / "alpha").mkdir(parents=True)
    (catalog / "alpha" / "skill.yaml").write_text(
        "id: alpha\nversion: 0.1.0\ndescription: x\nintents: [alpha]\n"
        "tool: t\nargs_schema: {type: object, additionalProperties: false}\n"
        "requires_tier1: false\n"
    )
    monkeypatch.setenv("AEGIS_HOME", str(tmp_path))

    cfg = AegisConfig()
    router = build_intent_router(cfg.skills.catalog_dir)

    assert router is not None
    # Sanity — the intent is live
    descriptor = router.match("alpha")
    assert descriptor is not None
    assert descriptor.id == "alpha"
```

- [ ] **Step 2: Run test to verify it fails (or passes trivially)**

Run: `pytest tests/test_skills_bootstrap.py::test_build_intent_router_reads_cfg_catalog_dir -v`

If it already passes because `build_intent_router` accepts an arg, great — move on. The real change is at the _callers_ of `build_intent_router` / `build_scheduler` / etc., which today use the module-level constant.

- [ ] **Step 3: Replace the hardcoded constant in `bot.py`**

Edit `runtime/chat/telegram/bot.py`. Delete line 90:

```python
_CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "catalog"
```

Grep for every remaining use of `_CATALOG_DIR` inside `bot.py`:

```
grep -n "_CATALOG_DIR" runtime/chat/telegram/bot.py
```

For each hit, rewrite so the caller passes `cfg.skills.catalog_dir` (or accepts it from an arg already in scope). Concretely:

- `build_intent_router(catalog_dir=...)` — existing signature already accepts it; just change callers from `build_intent_router()` to `build_intent_router(cfg.skills.catalog_dir)`.
- `build_scheduler(...)` — line 947 uses `catalog = _CATALOG_DIR`. Change to `catalog = cfg.skills.catalog_dir`.
- `build_application(...)` — find where the registry + intent router + scheduler are wired; replace any remaining `_CATALOG_DIR` reference with `cfg.skills.catalog_dir`.

- [ ] **Step 4: Call `seed_builtin_skills` at startup**

Still in `bot.py`, find `build_application` (grep for `def build_application`). Early in that function — before `build_intent_router`, before `SkillRegistry.from_directory`, before anything else touches the catalog — insert:

```python
from runtime.skills.bootstrap import seed_builtin_skills

seed_builtin_skills(
    bundle_dir=cfg.skills.bundle_dir,
    catalog_dir=cfg.skills.catalog_dir,
)
```

(Move the import to the top of the file alongside the other `runtime.skills.*` imports if that matches the file's style.)

- [ ] **Step 5: Update `runtime/chat/cli.py`**

Edit line 32. Replace:

```python
CATALOG_DIR = Path(__file__).parent.parent / "skills" / "catalog"
```

with:

```python
def _catalog_dir() -> Path:
    from runtime.config import get_config  # noqa: PLC0415
    return get_config().skills.catalog_dir
```

Grep for `CATALOG_DIR` uses inside the file; replace each with `_catalog_dir()`.

- [ ] **Step 6: Update `runtime/reflection/cli.py`**

Edit line 36 the same way.

- [ ] **Step 7: Run the full suite**

Run: `pytest -x -q`
Expected: PASS. Fixtures that used to rely on in-repo `runtime/skills/catalog/` may need to point at a tmp dir — fix them as they fail. Don't delete in-repo `runtime/skills/catalog/` yet; it's still valid fallback content until Task 11.

- [ ] **Step 8: Commit**

```bash
git add runtime/chat/telegram/bot.py runtime/chat/cli.py runtime/reflection/cli.py tests/test_skills_bootstrap.py
git commit -m "feat(skills): route every catalog consumer through cfg.skills.catalog_dir"
```

---

## Task 6: Migrate `morning_brief` to the seed bundle (end-to-end proof)

**Files:**
- Create: `runtime/skills/_bundle/morning_brief/skill.yaml`
- Create: `runtime/skills/_bundle/morning_brief/morning_brief.py`
- Delete: `runtime/skills/catalog/morning_brief.yaml`
- Delete: `runtime/skills/scripts/morning_brief.py`
- Modify: `tests/test_morning_brief_script.py` — update the module import path (the script is no longer a module).

- [ ] **Step 1: Copy the script body into the bundle**

Copy the full contents of `runtime/skills/scripts/morning_brief.py` to `runtime/skills/_bundle/morning_brief/morning_brief.py` — byte-for-byte. Do not rewrite logic.

- [ ] **Step 2: Write the new descriptor with `{skill_dir}`**

Create `runtime/skills/_bundle/morning_brief/skill.yaml`:

```yaml
id: morning_brief
version: 0.1.0
description: >-
  Daily brief — motivational quote, NZ weather (MetOcean), GitHub Trending,
  Hacker News top 5, TechCrunch AI, and NZ news. Writes Markdown to
  <vault_root>/Daily/YYYY/MM/DD-daily-news.md and returns the full Markdown.
  Requires METSERVICE_API_KEY in environment for the weather section; other
  sections degrade to inline warnings on failure.
intents:
  - morning_brief
  - daily_brief
tool: morning_brief
args_schema:
  type: object
  properties:
    vault_root:
      type: string
      minLength: 1
      maxLength: 512
      description: Absolute path to the Obsidian vault root.
  required:
    - vault_root
  additionalProperties: false
requires_tier1: false
tools:
  - name: morning_brief
    argv_template:
      - python
      - "{skill_dir}/morning_brief.py"
      - --vault-root
      - "{vault_root}"
    timeout_ms: 90000
    allow_net: true
```

- [ ] **Step 3: Delete the legacy in-repo copies**

```bash
rm runtime/skills/catalog/morning_brief.yaml
rm runtime/skills/scripts/morning_brief.py
```

- [ ] **Step 4: Update the script's test**

`tests/test_morning_brief_script.py` almost certainly imports the module as `runtime.skills.scripts.morning_brief`. Change to a `subprocess`-based smoke that invokes the copied file by absolute path, or import via `importlib.util.spec_from_file_location`. Concrete code:

```python
# top of tests/test_morning_brief_script.py, replace the import block
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "runtime" / "skills" / "_bundle" / "morning_brief" / "morning_brief.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("morning_brief_bundle", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


morning_brief = _load_module()
```

Replace every subsequent reference to `runtime.skills.scripts.morning_brief` in that file with `morning_brief` (the local alias).

- [ ] **Step 5: Run the morning_brief test**

Run: `pytest tests/test_morning_brief_script.py -v`
Expected: PASS.

- [ ] **Step 6: Run the scheduler + intent-router tests**

Run: `pytest tests/test_scheduler_seed.py tests/test_scheduler_engine.py tests/test_scheduler_runner.py tests/test_intent_router.py tests/test_telegram_bot.py tests/test_skill_catalog_recurring.py tests/test_cron_run.py -v`

Expected: PASS. These tests build a registry that now picks up `morning_brief` from the bundle path (via `seed_builtin_skills`) instead of the deleted catalog file.

Fix any test fixture that hardcodes `runtime/skills/catalog/morning_brief.yaml`: switch to seeding a `tmp_path` catalog directory first (use the `seed_builtin_skills` helper if convenient).

- [ ] **Step 7: End-to-end smoke**

Run:
```bash
rm -rf /tmp/aegis-smoke && AEGIS_HOME=/tmp/aegis-smoke python -c "
from runtime.config import AegisConfig
from runtime.skills.bootstrap import seed_builtin_skills
from runtime.skills.registry import SkillRegistry
cfg = AegisConfig()
seed_builtin_skills(bundle_dir=cfg.skills.bundle_dir, catalog_dir=cfg.skills.catalog_dir)
r = SkillRegistry.from_directory(cfg.skills.catalog_dir)
print(r.get('morning_brief'))
print(r.source_dir_of('morning_brief'))
"
```

Expected: prints the descriptor and `/tmp/aegis-smoke/skills/morning_brief`.

- [ ] **Step 8: Commit**

```bash
git add runtime/skills/_bundle/morning_brief runtime/skills/catalog/morning_brief.yaml runtime/skills/scripts/morning_brief.py tests/test_morning_brief_script.py
git commit -m "refactor(skills): migrate morning_brief to workspace bundle with {skill_dir}"
```

---

## Task 7: Migrate `echo` to the bundle

**Files:**
- Create: `runtime/skills/_bundle/echo/skill.yaml`
- Create: `runtime/skills/_bundle/echo/echo.py`
- Delete: `runtime/skills/catalog/echo.yaml`
- Delete: `runtime/skills/scripts/echo.py`

- [ ] **Step 1: Copy script to bundle**

Copy `runtime/skills/scripts/echo.py` → `runtime/skills/_bundle/echo/echo.py` byte-for-byte.

- [ ] **Step 2: Write the bundle descriptor**

Create `runtime/skills/_bundle/echo/skill.yaml`:

```yaml
id: echo
version: 0.1.0
description: >-
  Trivial echo skill — proves the runtime → intent → skill → harness → tool
  pipeline. Returns the user's message verbatim.
intents:
  - echo
  - ping
tool: echo
args_schema:
  type: object
  properties:
    message:
      type: string
  required:
    - message
requires_tier1: false
tools:
  - name: echo
    argv_template:
      - python
      - "{skill_dir}/echo.py"
    timeout_ms: 5000
    allow_net: false
```

- [ ] **Step 3: Delete legacy copies**

```bash
rm runtime/skills/catalog/echo.yaml
rm runtime/skills/scripts/echo.py
```

- [ ] **Step 4: Run affected tests**

Run: `pytest tests/test_intent_router.py tests/test_skill_catalog_recurring.py tests/test_telegram_bot.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add runtime/skills/_bundle/echo runtime/skills/catalog/echo.yaml runtime/skills/scripts/echo.py
git commit -m "refactor(skills): migrate echo to workspace bundle"
```

---

## Task 8: Migrate repo-coupled maintenance skills (YAML-only move)

Skills whose argv invokes a repo module directly (`python -m runtime.<mod>`) keep that argv unchanged — only the descriptor moves. No `{skill_dir}` needed.

**Files:**
- Create: `runtime/skills/_bundle/tier2_compress/skill.yaml`
- Create: `runtime/skills/_bundle/vault_reindex/skill.yaml`
- Create: `runtime/skills/_bundle/reflection_sweep/skill.yaml`
- Delete: `runtime/skills/catalog/tier2_compress.yaml`
- Delete: `runtime/skills/catalog/vault_reindex.yaml`
- Delete: `runtime/skills/catalog/reflection_sweep.yaml`
- Keep: `runtime/skills/scripts/tier2_compress.py` and `runtime/skills/scripts/vault_reindex.py` (they import `runtime.*` and must remain importable as repo modules).

- [ ] **Step 1: Copy each descriptor byte-for-byte into its bundle directory**

For each of the three skills, the descriptor content in `runtime/skills/_bundle/<id>/skill.yaml` is **identical** to the current `runtime/skills/catalog/<id>.yaml`. No argv changes — they still reference `runtime.skills.scripts.*` / `runtime.reflection.cli` via `python -m`.

Example — `runtime/skills/_bundle/tier2_compress/skill.yaml` — copy the existing file byte-for-byte using `cp`:

```bash
mkdir -p runtime/skills/_bundle/tier2_compress
cp runtime/skills/catalog/tier2_compress.yaml runtime/skills/_bundle/tier2_compress/skill.yaml

mkdir -p runtime/skills/_bundle/vault_reindex
cp runtime/skills/catalog/vault_reindex.yaml runtime/skills/_bundle/vault_reindex/skill.yaml

mkdir -p runtime/skills/_bundle/reflection_sweep
cp runtime/skills/catalog/reflection_sweep.yaml runtime/skills/_bundle/reflection_sweep/skill.yaml
```

- [ ] **Step 2: Delete the legacy catalog files**

```bash
rm runtime/skills/catalog/tier2_compress.yaml
rm runtime/skills/catalog/vault_reindex.yaml
rm runtime/skills/catalog/reflection_sweep.yaml
```

- [ ] **Step 3: Run affected tests**

Run: `pytest tests/test_scheduler_seed.py tests/test_scheduler_engine.py tests/test_scheduler_runner.py tests/test_health_handler.py tests/test_cron_run.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add runtime/skills/_bundle runtime/skills/catalog
git commit -m "refactor(skills): migrate maintenance skill descriptors to workspace bundle"
```

---

## Task 9: Migrate macos-files cluster + LLM-only descriptors

Same shape as Task 8 — descriptors move to the bundle, no argv changes. These skills either call repo modules (macos-files) or have no `tools:` block at all (LLM-only `ask_question`, `time_query`).

**Files:**
- Create: `runtime/skills/_bundle/list_files/skill.yaml`
- Create: `runtime/skills/_bundle/read_file/skill.yaml`
- Create: `runtime/skills/_bundle/search_files/skill.yaml`
- Create: `runtime/skills/_bundle/file_info/skill.yaml`
- Create: `runtime/skills/_bundle/ask_question/skill.yaml`
- Create: `runtime/skills/_bundle/time_query/skill.yaml`
- Delete: six corresponding files under `runtime/skills/catalog/`.

- [ ] **Step 1: Copy descriptors byte-for-byte**

```bash
for id in list_files read_file search_files file_info ask_question time_query; do
  mkdir -p "runtime/skills/_bundle/$id"
  cp "runtime/skills/catalog/${id}.yaml" "runtime/skills/_bundle/$id/skill.yaml" 2>/dev/null || \
    cp "runtime/skills/catalog/general_question.yaml" "runtime/skills/_bundle/$id/skill.yaml"
done
```

(Note: the existing file is `runtime/skills/catalog/general_question.yaml` but its internal `id` is `ask_question` — keep the bundle directory matching the internal id, i.e. `ask_question`, so the loader's directory-per-skill contract holds.)

Confirm each `skill.yaml` landed correctly:
```bash
ls runtime/skills/_bundle/*/skill.yaml
```

- [ ] **Step 2: Delete the legacy catalog files**

```bash
rm runtime/skills/catalog/list_files.yaml
rm runtime/skills/catalog/read_file.yaml
rm runtime/skills/catalog/search_files.yaml
rm runtime/skills/catalog/file_info.yaml
rm runtime/skills/catalog/general_question.yaml
rm runtime/skills/catalog/time_query.yaml
```

- [ ] **Step 3: Run the full suite**

Run: `pytest -x -q`
Expected: PASS. Any test still referencing `runtime/skills/catalog/<id>.yaml` by path needs to switch to `runtime/skills/_bundle/<id>/skill.yaml` or build a tmp catalog via `seed_builtin_skills`. Fix as they fail.

- [ ] **Step 4: Commit**

```bash
git add runtime/skills/_bundle runtime/skills/catalog
git commit -m "refactor(skills): migrate file + LLM-only descriptors to workspace bundle"
```

---

## Task 10: Installer promotes confirmed skills into the new directory layout

**Files:**
- Modify: `runtime/skills/installer.py` — `confirm_skill`
- Modify: `runtime/chat/telegram/skills_slash.py:138` and `scripts/skill_add.py:132-135` — pass `cfg.skills.catalog_dir` instead of the old in-repo path.
- Test: `tests/test_skill_installer.py` (update)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_skill_installer.py`:

```python
def test_confirm_writes_into_directory_per_skill_layout(tmp_path: Path) -> None:
    from runtime.skills.installer import confirm_skill, stage_skill

    src = tmp_path / "alpha.yaml"
    src.write_text(
        "id: alpha\nversion: 0.1.0\ndescription: x\nintents: [alpha]\n"
        "tool: t\nargs_schema: {type: object, additionalProperties: false}\n"
        "requires_tier1: false\n"
    )
    staging = tmp_path / "staging"
    catalog = tmp_path / "workspace" / "skills"

    stage_skill(src, staging_dir=staging)
    outcome = confirm_skill("alpha", staging_dir=staging, catalog_dir=catalog)

    assert outcome.is_success()
    assert outcome.final_path == catalog / "alpha" / "skill.yaml"
    assert (catalog / "alpha" / "skill.yaml").is_file()
    assert not (staging / "alpha.yaml").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_installer.py::test_confirm_writes_into_directory_per_skill_layout -v`
Expected: FAIL — current implementation writes `<catalog_dir>/<id>.yaml` (flat).

- [ ] **Step 3: Update `confirm_skill`**

Edit `runtime/skills/installer.py`. Replace the confirm path construction. Find the lines that compute `final_path = catalog_dir / f"{skill_id}.yaml"` (or equivalent) and replace with:

```python
final_dir = catalog_dir / skill_id
final_dir.mkdir(parents=True, exist_ok=True)
final_path = final_dir / "skill.yaml"
```

Keep the atomic `.tmp` rename pattern the existing function uses — just point it at the new `final_path`.

- [ ] **Step 4: Run the new test**

Run: `pytest tests/test_skill_installer.py -v`
Expected: PASS. Older tests that asserted flat-layout paths need updating — change the asserted final path to `<catalog_dir>/<id>/skill.yaml`.

- [ ] **Step 5: Update the two install callers**

`runtime/chat/telegram/skills_slash.py:138` and `scripts/skill_add.py:132-135` currently pass a `catalog_dir=` that resolves to the in-repo path. Grep + fix:

```
grep -n "catalog_dir" runtime/chat/telegram/skills_slash.py scripts/skill_add.py
```

For each call site, thread `cfg.skills.catalog_dir` through. In `scripts/skill_add.py`, load `AegisConfig` at the top and use `cfg.skills.catalog_dir`. In `skills_slash.py`, the command context should already carry `cfg`; if not, inject it via the handler builder.

- [ ] **Step 6: Run the full suite**

Run: `pytest -x -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add runtime/skills/installer.py runtime/chat/telegram/skills_slash.py scripts/skill_add.py tests/test_skill_installer.py
git commit -m "feat(skills): installer promotes confirmed skills into workspace dir layout"
```

---

## Task 11: Delete legacy paths + update docs

**Files:**
- Delete: `runtime/skills/catalog/` (must be empty now).
- Delete: `runtime/skills/scripts/` (must contain only tier2_compress.py + vault_reindex.py — these stay. The directory itself stays with those two files.).
- Modify: `README.md` — canonical-state section.
- Modify: `docs/PLAN_PHASE_0_AND_WALKING_SKELETON.md` — `skills/` is no longer empty.

- [ ] **Step 1: Confirm catalog is empty**

```bash
ls runtime/skills/catalog/ 2>/dev/null
```

Expected: empty listing or "No such file or directory" (some deletes in Tasks 6–9 may have already removed everything).

If anything remains, stop and audit — a descriptor was missed.

- [ ] **Step 2: Delete the legacy catalog directory**

```bash
rmdir runtime/skills/catalog
```

Also audit `runtime/skills/scripts/`:

```bash
ls runtime/skills/scripts/
```

Expected: `tier2_compress.py`, `vault_reindex.py`, `__init__.py`. These are imports from repo modules and stay in place.

- [ ] **Step 3: Grep for any remaining references to the legacy path**

```bash
grep -rn "skills/catalog\|runtime.skills.scripts.morning_brief\|runtime.skills.scripts.echo" --include="*.py" --include="*.md" --include="*.yaml"
```

Expected: only matches in docs referring to historical context (plans, commit messages, specs). Fix any runtime or test code still referencing them.

- [ ] **Step 4: Update `README.md`**

Find the "Canonical state lives outside this repo" bullet list in `README.md:39`. Add `skills/` to the list:

```markdown
- `~/.aegis/workspace/skills/` — AEGIS skill catalog (descriptors + co-located scripts)
```

- [ ] **Step 5: Update `docs/PLAN_PHASE_0_AND_WALKING_SKELETON.md`**

Find line 55: `mkdir -p ~/.aegis/workspace/skills        # empty for now` — remove the "# empty for now" comment and add a pointer to the workspace migration doc:

```markdown
mkdir -p ~/.aegis/workspace/skills        # seeded on first boot from runtime/skills/_bundle
```

- [ ] **Step 6: Run full suite one last time**

Run: `pytest -q`
Expected: PASS, every test green.

- [ ] **Step 7: Cold-boot smoke**

```bash
rm -rf /tmp/aegis-coldboot
AEGIS_ROOT=/tmp/aegis-coldboot python -c "
from runtime.config import AegisConfig
from runtime.skills.bootstrap import seed_builtin_skills
from runtime.skills.registry import SkillRegistry
cfg = AegisConfig()
seed_builtin_skills(bundle_dir=cfg.skills.bundle_dir, catalog_dir=cfg.skills.catalog_dir)
reg = SkillRegistry.from_directory(cfg.skills.catalog_dir)
ids = sorted(d.id for d in reg.all())
print(ids)
"
```

Expected output (order may vary — nine-to-eleven ids):
```
['ask_question', 'echo', 'file_info', 'list_files', 'morning_brief', 'read_file', 'reflection_sweep', 'search_files', 'tier2_compress', 'time_query', 'vault_reindex']
```

- [ ] **Step 8: Commit**

```bash
git add runtime/skills/catalog README.md docs/PLAN_PHASE_0_AND_WALKING_SKELETON.md
git commit -m "chore(skills): retire runtime/skills/catalog after workspace migration"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Workspace path `~/.aegis/workspace/skills/` is the canonical catalog (Task 1 config + Task 5 wiring).
- ✅ Directory-per-skill shape matching atamai (Task 2 loader + Task 10 installer).
- ✅ Bundled scripts co-located with descriptor via `{skill_dir}` (Task 3 + Task 6 + Task 7).
- ✅ Repo-coupled skills keep module-style argv (Task 8 + Task 9).
- ✅ First-boot seed from repo; operator edits win (Task 4).
- ✅ Installer writes into the new layout (Task 10).
- ✅ Legacy paths retired + docs updated (Task 11).

**Placeholder scan:** No "TBD" / "similar to above" / "handle edge cases" in any step. Every step shows the actual code or the exact command.

**Type consistency:** `SkillsConfig.catalog_dir` / `SkillsConfig.bundle_dir` are used identically in every task that mentions them. `seed_builtin_skills(bundle_dir=, catalog_dir=)` kwargs match between Task 4 definition and Task 5 + Task 11 calls. `registry.source_dir_of(id)` is defined in Task 2 and consumed by the resolver in Task 3 with the same signature.

**Trade-offs locked in:**

- Skill IDs keep underscores (`morning_brief`, not `morning-brief`) — matches existing AEGIS intent strings, scheduler `SYS-*` ids, and avoids a cascading rename. Atamai's hyphen convention is not preserved; parity on shape (one-dir-per-skill + `skill.yaml`) is what matters here.
- `tier2_compress` and `vault_reindex` scripts stay in `runtime/skills/scripts/` because they import repo subsystems (`runtime.chat.memory.tier2`, `runtime.config`). Moving them would break their imports. The descriptor moves to the workspace; the implementation remains a repo module invoked via `python -m`.
- Flat-layout loading (`<id>.yaml` at catalog root) is kept as a back-compat fallback in `SkillRegistry.from_directory` — useful in tests that don't want to build a full directory tree and as a one-revision safety net during migration.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-23-workspace-skills-migration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
