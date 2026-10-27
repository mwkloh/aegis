"""Where a benchmarked model actually ran: on this device, on Ollama's cloud, or hosted.

The benchmark's point is how well small models run on a local device. A "<=9B
SLM" result that quietly came from a cloud endpoint would be a different claim,
and two of the first run's configs (hosted qwen3.5-9b and llama-3.1-8b on
OpenRouter) were exactly that. `gemma4:cloud` is also worth labelling: it goes
through the local Ollama daemon (so `provider` says "ollama") but runs remotely.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.eval.cli import _locality_refusal, _parse_args
from runtime.eval.locality import classify_locality
from runtime.eval.report import EvalReport, TaskResult, VariantResult, render_console, write_json

pytestmark = pytest.mark.unit

_PULLED = {"gemma4:e2b-mlx", "llama3.1:8b", "qwen3.5:9b"}


def test_a_pulled_ollama_model_is_local() -> None:
    assert classify_locality("ollama", "llama3.1:8b", _PULLED) == "local"


def test_an_ollama_model_that_is_not_pulled_is_cloud() -> None:
    """`gemma4:cloud` is addressed through the local daemon but never appears in
    its model list -- the daemon proxies it to a remote host."""
    assert classify_locality("ollama", "gemma4:cloud", _PULLED) == "cloud"


def test_openrouter_is_hosted() -> None:
    assert classify_locality("openrouter", "qwen/qwen3.5-9b", _PULLED) == "hosted"


def test_a_cloud_suffixed_name_is_cloud_even_if_listed() -> None:
    """Belt and braces: a ':cloud' tag must never be counted as on-device."""
    assert classify_locality("ollama", "gemma4:cloud", _PULLED | {"gemma4:cloud"}) == "cloud"


@pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "qwen3-coder:480b-cloud"])
def test_a_dash_cloud_tag_is_cloud_even_if_listed(model: str) -> None:
    """Ollama's older cloud tags end in `-cloud`, not `:cloud`."""
    assert classify_locality("ollama", model, _PULLED | {model}) == "cloud"


def test_a_cloud_word_in_the_model_name_is_not_a_cloud_tag() -> None:
    """Only the tag after the last `:` marks a cloud model."""
    assert classify_locality("ollama", "cloud-helper:7b", {"cloud-helper:7b"}) == "local"


def test_matching_is_exact() -> None:
    assert classify_locality("ollama", "llama3.1", _PULLED) == "cloud"


def test_cli_require_local_defaults_off() -> None:
    assert _parse_args([]).require_local is False
    assert _parse_args(["--require-local"]).require_local is True


@pytest.mark.parametrize("locality", ["cloud", "hosted"])
def test_require_local_refuses_anything_off_device(locality: str) -> None:
    reason = _locality_refusal(locality, require_local=True)
    assert reason is not None
    assert locality in reason


def test_require_local_allows_a_local_model() -> None:
    assert _locality_refusal("local", require_local=True) is None


def test_without_require_local_nothing_is_refused() -> None:
    assert _locality_refusal("cloud", require_local=False) is None


def _report(locality: str | None) -> EvalReport:
    extra = {} if locality is None else {"locality": locality}
    return EvalReport(
        provider="ollama",
        model="m",
        started_at=datetime(2026, 10, 4, tzinfo=UTC),
        tasks=(
            TaskResult(
                task_id="t",
                description="d",
                variants=(
                    VariantResult(task_id="t", variant_text="x", passed=True, duration_s=1.0),
                ),
            ),
        ),
        **extra,
    )


def test_report_locality_defaults_to_unknown_so_old_files_still_parse() -> None:
    assert _report(None).locality == "unknown"


def test_report_locality_round_trips_through_json(tmp_path: Path) -> None:
    path = write_json(_report("local"), tmp_path)
    assert EvalReport.model_validate_json(path.read_text(encoding="utf-8")).locality == "local"


def test_console_header_names_where_the_model_ran() -> None:
    assert "[local]" in render_console(_report("local"))
    assert "[cloud]" in render_console(_report("cloud"))
    assert "[unknown]" not in render_console(_report(None))


async def test_require_local_refuses_a_model_ollama_serves_remotely(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end through `_run`: a pulled cloud model is listed in /api/tags with a
    `remote_host`. The name carries no cloud tag, so only that field can catch it."""
    from runtime.config import AegisConfig
    from runtime.eval import cli
    from runtime.llm.clients import OllamaClient

    model = "remote-only:7b"
    cfg = AegisConfig.model_validate({"models": {"smart_provider": "ollama", "smart_local": model}})

    async def _tags(_self: OllamaClient) -> list[dict[str, object]]:
        return [{"name": model, "remote_host": "https://ollama.com"}]

    monkeypatch.setattr(cli, "get_config", lambda: cfg)
    monkeypatch.setattr(OllamaClient, "_tags", _tags)
    repo = Path(__file__).resolve().parent.parent
    args = _parse_args(["--require-local", "--tasks-dir", str(repo / "eval" / "tasks")])

    assert await cli._run(args) == 2
    assert "this model is cloud" in capsys.readouterr().err
