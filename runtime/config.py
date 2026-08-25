"""Single config gateway for Plane 1.

Loads `~/.aegis/.env` and `~/.aegis/config.json`. Never reads project-local .env.
Pydantic validates every field at the boundary.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from runtime.board.config import BoardConfig

logger = logging.getLogger(__name__)


def _aegis_root() -> Path:
    return Path(os.environ.get("AEGIS_ROOT", str(Path.home() / ".aegis"))).expanduser()


def _aegis_home() -> Path:
    return Path(
        os.environ.get("AEGIS_HOME", str(_aegis_root() / "workspace"))
    ).expanduser()


class ModelConfig(BaseModel):
    """Pinned model identifiers and routing preferences."""

    fast: str = Field(default="gemma4:e2b", description="Tier 0 intent classifier.")
    smart: str = Field(
        default="minimax/minimax-m2.7", description="Tier 1 reasoning (OpenRouter)."
    )
    smart_local: str = Field(
        default="llama3.1:8b",
        description="Tier 1 reasoning via Ollama — local weights or an Ollama Cloud tag.",
    )
    reflection: str = Field(default="gemma4:e4b", description="Reflection plane.")
    coding: str = Field(
        default="minimax/minimax-m2.7",
        description="Plane 3 coding harness (OpenRouter by default).",
    )
    smart_think: bool | None = Field(
        default=None,
        description=(
            "Ollama reasoning-channel override for the Tier-1 planner and "
            "reply-synthesis calls. None leaves the model's own default "
            "alone. Measured 2026-08-25: disabling it took qwen3-vl:4b from "
            "40% to 80% TGC but dropped qwen3.5:4b-mlx's in-budget score "
            "from 73.3% to 6.7% -- there is no safe global setting, so this "
            "is per-deployment. Ignored by OpenRouter."
        ),
    )
    smart_provider: Literal["ollama", "openrouter"] = Field(
        default="ollama",
        description="Explicit SMART-tier provider pin — no silent fallback between them.",
    )


class ModelProfile(BaseModel):
    """Per-model overrides, keyed by exact model id in `config.json`.

    Exists because a tier-wide setting cannot be right for every model.
    Measured 2026-08-25 across the six-task eval suite, `think=False`:

    * `qwen3-vl:4b`    TGC 40.0% -> 80.0%  (rescued -- it was spending its
      whole token budget on hidden reasoning and never emitting content)
    * `qwen3.5:4b-mlx` in-budget 73.3% -> 6.7%  (broken -- that reasoning was
      how it got its JSON right first time; without it, retries tripled)

    Only fields with a measured per-model effect belong here. `think` is
    currently the only one; adding knobs on reasoning alone is what this whole
    line of work exists to stop.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    think: bool | None = Field(
        default=None,
        description="Ollama reasoning channel for Tier-1. None leaves the model alone.",
    )


def _coerce_model_profiles(raw: Any) -> dict[str, ModelProfile]:
    """Build `{model_id: ModelProfile}` from `config.json` -> `modelProfiles`.

    Tolerant by design, matching `_coerce_files` / `_coerce_board`: a
    malformed entry is dropped with a warning rather than raising, so a
    half-edited config.json still boots and the operator can see which key
    was ignored instead of getting a dead process.
    """
    if not isinstance(raw, dict):
        return {}
    profiles: dict[str, ModelProfile] = {}
    for model_id, body in raw.items():
        if not isinstance(model_id, str) or not isinstance(body, dict):
            logger.warning("config.model_profiles.skipped", extra={"model": model_id})
            continue
        try:
            profiles[model_id] = ModelProfile(**body)
        except ValidationError:
            # extra="forbid" lands here too, so a typo'd key is reported
            # rather than silently doing nothing for the life of the config.
            logger.warning(
                "config.model_profiles.invalid", extra={"model": model_id}
            )
    return profiles


class ProviderConfig(BaseModel):
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    ollama_embedding_model: str = Field(default="bge-m3")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_api_key: str | None = Field(default=None, repr=False)


class TelegramConfig(BaseModel):
    enabled: bool = False
    bot_token: str | None = Field(default=None, repr=False)
    user_allowlist: list[int] = Field(default_factory=list)


class VaultSource(BaseModel):
    """One folder under `vault_root` eligible for tier-2 indexing.

    Mirrors `MemorySourceSchema` from atamai so an existing config can
    be ported with a key renamer. `priority` is the multiplier applied
    to cosine similarity in `Tier2Store.search_vault` — higher means
    "promote hits from this folder above generic matches".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, description="Relative to vault_root.")
    priority: float = Field(default=1.0, ge=0.5, le=4.0)
    glob: str = "**/*.md"
    exclude: tuple[str, ...] = ()
    label: str | None = None


class VaultIndexingConfig(BaseModel):
    """Selective Obsidian-vault indexing (Phase 7 Track C §5.1).

    `vault_root=None` disables vault indexing entirely — the loader
    and the `/vault` slashes both treat this as "not configured" and
    do not crash. Callers should check `is_enabled()` before wiring.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    vault_root: Path | None = None
    sources: tuple[VaultSource, ...] = ()
    reindex_interval_hours: int = Field(default=6, ge=1)

    @field_validator("vault_root")
    @classmethod
    def _expand_root(cls, v: Path | None) -> Path | None:
        return None if v is None else Path(v).expanduser()

    def is_enabled(self) -> bool:
        return self.vault_root is not None and len(self.sources) > 0


class StorageConfig(BaseModel):
    workspace: Path = Field(default_factory=_aegis_home)
    sessions_dir: Path = Field(default_factory=lambda: _aegis_home() / "sessions")
    memory_db: Path = Field(default_factory=lambda: _aegis_home() / "memory" / "aegis-index.db")

    @field_validator("workspace", "sessions_dir", "memory_db")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser()


class HarnessConfig(BaseModel):
    """Tool-use harness behaviour (HarnessDispatcher).

    `multi_step=False` keeps the legacy single-shot dispatch (classify → one
    tool → synthesise). When True, dispatch routes through `plan_next` so the
    operator can chain tool calls within one turn (see
    docs/PLAN_MULTI_STEP_AGENT_LOOP.md).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    multi_step: bool = Field(
        default=False,
        description="Opt-in flag for the multi-step agent loop.",
    )
    max_steps: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Hard cap on tool-call steps per turn when multi_step=True.",
    )


class FilesConfig(BaseModel):
    """Sandboxed filesystem access config for /files and file harness tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_roots: list[Path] = Field(
        default_factory=lambda: [
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            Path.home() / "Development",
            Path.home() / "data",
        ]
    )

    @field_validator("allowed_roots", mode="before")
    @classmethod
    def _expand_roots(cls, v: object) -> list[Path]:
        if not isinstance(v, list):
            return []
        return [
            Path(str(r)).expanduser()
            for r in v
            if isinstance(r, (str, Path)) and str(r).strip()
        ]


class CommandsConfig(BaseModel):
    """Argv-only command runner (run_command tool). No shell, ever."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_binaries: tuple[str, ...] = Field(
        default=("ls", "cat", "head", "tail", "wc", "grep", "file"),
        description="Binaries the model may invoke as argv[0]. Read-only "
        "inspection tools by default; operators extend deliberately. "
        "`find` is deliberately excluded — its -delete/-exec/-execdir/-ok "
        "flags allow arbitrary deletion/execution with no confirmation "
        "(run_command is not in DESTRUCTIVE_TOOLS); add it back only if "
        "you accept that risk.",
    )
    timeout_ms: int = Field(default=15_000, ge=100, le=120_000)
    max_output_bytes: int = Field(default=32_768, ge=1024, le=262_144)


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
    staging_dir: Path = Field(default_factory=lambda: _aegis_home() / "skills_staging")

    @field_validator("catalog_dir", "bundle_dir", "staging_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser()


class AegisConfig(BaseModel):
    """Top-level config object — the only thing runtime code should import."""

    aegis_home: Path = Field(default_factory=_aegis_home)
    aegis_root: Path = Field(default_factory=_aegis_root)
    models: ModelConfig = Field(default_factory=ModelConfig)
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    vault_indexing: VaultIndexingConfig = Field(default_factory=VaultIndexingConfig)
    board: BoardConfig = Field(default_factory=BoardConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    commands: CommandsConfig = Field(default_factory=CommandsConfig)
    model_profiles: dict[str, ModelProfile] = Field(default_factory=dict)

    @field_validator("aegis_home", "aegis_root")
    @classmethod
    def _expand_root(cls, v: Path) -> Path:
        return Path(v).expanduser()

    def think_for(self, model: str) -> bool | None:
        """Resolve the Tier-1 reasoning-channel setting for one model.

        Precedence: `MODEL_SMART_THINK` beats any profile, so an operator can
        force a setting for a debugging session without editing config.json --
        the same direction `MODEL_SMART_LOCAL` already overrides config. Note
        `False` is a real setting and wins like `True` does; only an unset env
        var falls through to the profile.

        Matching is on exact model id. No family prefixes: `qwen3-vl:4b` wants
        `think=False` and `qwen3.5:4b-mlx` wants `think=True`, so inheriting
        by vendor would hand one of them the other's measured-wrong value.
        """
        if self.models.smart_think is not None:
            return self.models.smart_think
        profile = self.model_profiles.get(model)
        return profile.think if profile is not None else None


def _load_env(env_path: Path) -> None:
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _load_json(json_path: Path) -> dict[str, Any]:
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} must contain a JSON object at top level")
    return data


def _env_bool(raw: str | None, *, default: bool) -> bool:
    """Permissive truthy parser for env flags."""
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_tristate_bool(raw: str | None) -> bool | None:
    """Unset -> None ("leave the model alone"), which is distinct from False."""
    if raw is None or not raw.strip():
        return None
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    logger.warning("config.smart_think.unrecognized_value", extra={"raw": raw})
    return None


def _parse_smart_provider(raw: str | None) -> Literal["ollama", "openrouter"]:
    """Explicit SMART-tier provider pin. Unset/invalid -> 'ollama' (local-first default)."""
    if raw is None:
        return "ollama"
    normalized = raw.strip().lower()
    if normalized == "openrouter":
        return "openrouter"
    if normalized and normalized != "ollama":
        logger.warning(
            "config.smart_provider.unrecognized_value",
            extra={"raw": raw},
        )
    return "ollama"


def _parse_env_allowlist(raw: str | None) -> list[int] | None:
    """Comma-separated ids → list[int]. `None` means "env var unset"."""
    if raw is None:
        return None
    out: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if p and p.lstrip("-").isdigit():
            out.append(int(p))
    return out


def _coerce(env: dict[str, str], cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the AegisConfig kwargs from env + json without leaking unknowns."""
    models = ModelConfig(
        fast=env.get("MODEL_FAST", ModelConfig.model_fields["fast"].default),
        smart=env.get("MODEL_SMART", ModelConfig.model_fields["smart"].default),
        smart_local=env.get(
            "MODEL_SMART_LOCAL", ModelConfig.model_fields["smart_local"].default
        ),
        reflection=env.get("MODEL_REFLECTION", ModelConfig.model_fields["reflection"].default),
        coding=env.get(
            "MODEL_CODING",
            env.get("MODEL_SMART", ModelConfig.model_fields["coding"].default),
        ),
        smart_provider=_parse_smart_provider(env.get("MODEL_SMART_PROVIDER")),
        smart_think=_parse_tristate_bool(env.get("MODEL_SMART_THINK")),
    )
    providers = ProviderConfig(
        ollama_base_url=env.get(
            "OLLAMA_BASE_URL", ProviderConfig.model_fields["ollama_base_url"].default
        ),
        ollama_embedding_model=env.get(
            "OLLAMA_EMBEDDING_MODEL",
            ProviderConfig.model_fields["ollama_embedding_model"].default,
        ),
        openrouter_base_url=env.get(
            "OPENROUTER_BASE_URL",
            ProviderConfig.model_fields["openrouter_base_url"].default,
        ),
        openrouter_api_key=env.get("OPENROUTER_API_KEY"),
    )
    tg_cfg = cfg.get("telegram", {}) if isinstance(cfg.get("telegram"), dict) else {}
    env_allow = _parse_env_allowlist(env.get("TELEGRAM_USER_ALLOWLIST"))
    if env_allow is not None:
        allow = env_allow
    else:
        raw = tg_cfg.get("userAllowlist") or tg_cfg.get("user_allowlist") or []
        # Config with `${VAR}` placeholders leaves a string here — iterating it
        # by character yields garbage, so require a list before parsing.
        if isinstance(raw, list):
            allow = [int(x) for x in raw if str(x).strip().lstrip("-").isdigit()]
        else:
            allow = []
    telegram = TelegramConfig(
        enabled=bool(tg_cfg.get("enabled", False)),
        bot_token=env.get("TELEGRAM_BOT_TOKEN"),
        user_allowlist=allow,
    )
    board = _coerce_board(cfg.get("board"), env)
    return {
        "models": models,
        "providers": providers,
        "telegram": telegram,
        "storage": StorageConfig(),
        "vault_indexing": _coerce_vault_indexing(cfg.get("vaultIndexing")),
        "board": board,
        "files": _coerce_files(cfg.get("files")),
        "harness": _coerce_harness(cfg.get("harness"), env),
        "commands": _coerce_commands(cfg.get("commands")),
        "model_profiles": _coerce_model_profiles(cfg.get("modelProfiles")),
    }


def _coerce_vault_indexing(raw: Any) -> VaultIndexingConfig:
    """Build a `VaultIndexingConfig` from `config.json` → `vaultIndexing`.

    Missing / non-dict / `${VAR}` placeholder → disabled default.
    Sources with missing `path` are silently dropped rather than raising,
    so a partially-edited config.json still boots (operators see the
    `/vault sources` output is empty and fix from there).
    """
    if not isinstance(raw, dict):
        return VaultIndexingConfig()
    vault_root_raw = raw.get("vaultRoot") or raw.get("vault_root")
    vault_root = Path(vault_root_raw).expanduser() if isinstance(vault_root_raw, str) else None
    sources_raw = raw.get("sources")
    sources: list[VaultSource] = []
    if isinstance(sources_raw, list):
        for entry in sources_raw:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            priority = entry.get("priority", 1.0)
            glob = entry.get("glob", "**/*.md")
            exclude_raw = entry.get("exclude", [])
            exclude = tuple(x for x in exclude_raw if isinstance(x, str)) \
                if isinstance(exclude_raw, list) else ()
            label = entry.get("label")
            try:
                sources.append(
                    VaultSource(
                        path=path.strip(),
                        priority=float(priority),
                        glob=str(glob),
                        exclude=exclude,
                        label=str(label) if isinstance(label, str) else None,
                    )
                )
            except (ValueError, TypeError):
                continue
    interval = raw.get("reindexIntervalHours") or raw.get("reindex_interval_hours") or 6
    try:
        interval_int = max(1, int(interval))
    except (ValueError, TypeError):
        interval_int = 6
    return VaultIndexingConfig(
        vault_root=vault_root,
        sources=tuple(sources),
        reindex_interval_hours=interval_int,
    )


def _coerce_board(raw: Any, env: dict[str, str]) -> BoardConfig:
    """Build a `BoardConfig` from `config.json` → `board` + env.

    Reads BRAVE_SEARCH_API_KEY from env and injects it into the research
    block (overriding the ${VAR} placeholder). Key absent → research=None.
    Missing / non-dict raw → default (empty panelists, no research).
    """
    if not isinstance(raw, dict):
        return BoardConfig()
    board_raw = dict(raw)
    research_raw = board_raw.get("research")
    brave_key = env.get("BRAVE_SEARCH_API_KEY")
    if brave_key:
        # Build research block from env key; preserve any top_k/timeout_s from config
        research_copy = dict(research_raw) if isinstance(research_raw, dict) else {}
        research_copy["brave_api_key"] = brave_key
        board_raw["research"] = research_copy
    elif isinstance(research_raw, dict):
        # research block present in config but no key in env → drop it
        board_raw.pop("research", None)
    try:
        return BoardConfig.model_validate(board_raw)
    except Exception:
        logger.warning("config.board.invalid", exc_info=True)
        return BoardConfig()


def _coerce_files(raw: Any) -> FilesConfig:
    """Build a `FilesConfig` from `config.json` → `files`.

    Missing / non-dict → default (5 standard macOS dirs).
    """
    if not isinstance(raw, dict):
        return FilesConfig()
    roots_raw = raw.get("allowed_roots") or raw.get("allowedRoots")
    if isinstance(roots_raw, list):
        try:
            return FilesConfig(allowed_roots=roots_raw)
        except (ValueError, TypeError):
            pass
    return FilesConfig()


def _coerce_commands(raw: Any) -> CommandsConfig:
    """Build a `CommandsConfig` from `config.json` → `commands`.

    Missing / non-dict → default (read-only inspection allowlist). Each field
    degrades independently: an invalid `allowed_binaries`, `timeout_ms`, or
    `max_output_bytes` falls back to its default rather than failing the whole
    load, so one bad key can't strand the operator's other overrides.
    """
    if not isinstance(raw, dict):
        return CommandsConfig()
    kwargs: dict[str, Any] = {}
    binaries_raw = raw.get("allowed_binaries") or raw.get("allowedBinaries")
    if isinstance(binaries_raw, list) and all(
        isinstance(b, str) for b in binaries_raw
    ):
        kwargs["allowed_binaries"] = tuple(binaries_raw)
    for key in ("timeout_ms", "max_output_bytes"):
        if key in raw:
            with contextlib.suppress(TypeError, ValueError):
                kwargs[key] = int(raw[key])
    try:
        return CommandsConfig(**kwargs)
    except Exception:
        logger.warning("config.commands.invalid", exc_info=True)
        return CommandsConfig()


def _coerce_harness(raw: Any, env: dict[str, str]) -> HarnessConfig:
    """Build a `HarnessConfig` from `config.json` → `harness` + env overrides.

    Missing / non-dict raw → defaults (single-shot dispatch). Env override
    `HARNESS_MULTI_STEP=1` is honoured so the operator can flip the flag from
    the launchd plist without editing config.json.
    """
    raw_dict = raw if isinstance(raw, dict) else {}
    multi_step_default = bool(raw_dict.get("multi_step", False))
    multi_step = _env_bool(env.get("HARNESS_MULTI_STEP"), default=multi_step_default)
    max_steps_raw = raw_dict.get("max_steps", HarnessConfig.model_fields["max_steps"].default)
    try:
        max_steps = int(max_steps_raw)
    except (TypeError, ValueError):
        max_steps = HarnessConfig.model_fields["max_steps"].default
    try:
        return HarnessConfig(multi_step=multi_step, max_steps=max_steps)
    except Exception:
        logger.warning("config.harness.invalid", exc_info=True)
        return HarnessConfig()


@lru_cache(maxsize=1)
def get_config() -> AegisConfig:
    """Load config once per process. Tests should call `reset_config()`."""
    root = _aegis_root()
    _load_env(root / ".env")
    cfg = _load_json(root / "config.json")
    kwargs = _coerce(dict(os.environ), cfg)
    return AegisConfig(**kwargs)


def reset_config() -> None:
    """Drop the cached config — for tests only."""
    get_config.cache_clear()
