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
    descriptor = router.match("alpha")
    assert descriptor is not None
    assert descriptor.id == "alpha"
