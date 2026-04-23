"""Repo-coupled skill scripts invoked as ``python -m runtime.skills.scripts.<name>``.

Only ``tier2_compress`` and ``vault_reindex`` live here — both import repo
subsystems, so they stay as package modules. Standalone skills co-locate
their scripts with their descriptor in ``runtime/skills/_bundle/<id>/``.
The argv contract is declared in the matching
``runtime/skills/_bundle/<name>/skill.yaml`` ToolSpec.
"""
