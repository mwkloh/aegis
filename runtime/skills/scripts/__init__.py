"""Executable skill scripts invoked as ``python -m runtime.skills.scripts.<name>``.

Each module here is a self-contained CLI. The argv contract is declared in the
matching ``runtime/skills/catalog/<name>.yaml`` ToolSpec so the tool harness
can spawn it under argv-only discipline.
"""
