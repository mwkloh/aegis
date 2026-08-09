"""FilesConfig — Pydantic model + _coerce_files integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.config import AegisConfig, FilesConfig, _coerce_files

pytestmark = pytest.mark.unit


def test_files_config_defaults_are_non_empty() -> None:
    cfg = FilesConfig()
    assert len(cfg.allowed_roots) > 0


def test_files_config_tilde_is_expanded() -> None:
    cfg = FilesConfig(allowed_roots=["~/Documents"])
    assert not str(cfg.allowed_roots[0]).startswith("~")
    assert cfg.allowed_roots[0].is_absolute()


def test_files_config_empty_roots_is_valid() -> None:
    # FilesConfig itself doesn't reject empty — FilesClient does at construction.
    cfg = FilesConfig(allowed_roots=[])
    assert cfg.allowed_roots == []


def test_coerce_files_returns_default_when_raw_is_none() -> None:
    cfg = _coerce_files(None)
    assert isinstance(cfg, FilesConfig)
    assert len(cfg.allowed_roots) > 0


def test_coerce_files_parses_allowed_roots() -> None:
    cfg = _coerce_files({"allowed_roots": ["~/Documents", "~/Downloads"]})
    assert len(cfg.allowed_roots) == 2
    assert all(p.is_absolute() for p in cfg.allowed_roots)


def test_coerce_files_falls_back_on_non_dict() -> None:
    cfg = _coerce_files("garbage")
    assert isinstance(cfg, FilesConfig)


def test_files_config_path_objects_are_accepted() -> None:
    cfg = FilesConfig(allowed_roots=[Path("~/Documents")])
    assert len(cfg.allowed_roots) == 1
    assert cfg.allowed_roots[0].is_absolute()


def test_coerce_files_accepts_camel_case_key() -> None:
    cfg = _coerce_files({"allowedRoots": ["~/Documents"]})
    assert len(cfg.allowed_roots) == 1


def test_aegis_config_has_files_field() -> None:
    cfg = AegisConfig()
    assert hasattr(cfg, "files")
    assert isinstance(cfg.files, FilesConfig)
