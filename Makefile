.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTHON_BIN ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3.13 || command -v python3)

.PHONY: help setup venv install bootstrap doctor lint type test test-unit test-e2e security run reflect review harness apply clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

venv:  ## Create the local virtualenv
	@test -d $(VENV) || $(PYTHON_BIN) -m venv $(VENV)

install: venv  ## Install runtime + dev deps
	$(PIP) install -U pip
	$(PIP) install -e '.[dev]'

setup: install bootstrap doctor  ## Full first-time setup

bootstrap:  ## Idempotently create ~/.aegis/ (canon, env, config)
	$(PY) scripts/bootstrap_aegis_home.py

doctor:  ## Verify Ollama, models, ~/.aegis/ layout, and pinned digests
	$(PY) scripts/doctor.py

lint:  ## Run ruff + bandit
	$(VENV)/bin/ruff check .
	$(VENV)/bin/bandit -q -c pyproject.toml -r runtime memory scripts -x tests

type:  ## Run mypy --strict
	$(VENV)/bin/mypy runtime memory scripts

test: lint type test-unit test-e2e  ## Full test gate

test-unit:  ## Fast unit tests
	$(VENV)/bin/pytest -m "unit or not e2e" tests/

test-e2e:  ## End-to-end walking-skeleton tests
	$(VENV)/bin/pytest -m e2e tests/

security:  ## Bandit + Semgrep (Semgrep already on system)
	$(VENV)/bin/bandit -q -c pyproject.toml -r runtime memory scripts -x tests
	semgrep --config auto runtime memory scripts || true

run:  ## Launch the CLI walking skeleton
	$(PY) -m runtime.chat.cli

reflect:  ## Run the Reflection plane over today's events (read-only)
	$(PY) -m runtime.reflection.cli

review:  ## Walk pending proposals and record human verdicts (Plane 3)
	$(PY) -m runtime.improvement.cli

harness:  ## Draft `.patch.md` files for approved coding tasks (Plane 3, draft-only)
	$(PY) -m runtime.coding_harness.cli

apply:  ## Apply ONE drafted CT onto a fresh branch and run the test suite (CT=CT-NNN)
	$(PY) -m runtime.coding_harness.apply_cli $(CT) $(ARGS)

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
