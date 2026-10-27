"""Eval arms are presets of the production config (spec 2026-10-06, section 5)."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.config import AegisConfig, ModelProfile
from runtime.eval.cli import ARMS, _apply_arm, _arm_unsupported_reason, _variant_runner
from runtime.eval.report import EvalReport, write_json
from runtime.harness.guardrails import Guardrails

pytestmark = pytest.mark.unit

M = "gemma4:e4b-mlx"


def _cfg() -> AegisConfig:
    # A native profile must not leak into the `full` arm.
    return AegisConfig(model_profiles={M: ModelProfile(harness_mode="native")})


@pytest.mark.parametrize(
    ("arm", "mode", "guard"),
    [
        ("full", "planner", Guardrails(plan_lock=True)),
        ("full-nolock", "planner", Guardrails(plan_lock=False)),
        ("native", "native", Guardrails(False, False, False, False, False)),
        ("native-guard", "native", Guardrails(plan_lock=False)),
        ("native-guard-synth", "native", Guardrails(synthesis=True, plan_lock=False)),
    ],
)
def test_each_arm_is_a_config_preset(arm: str, mode: str, guard: Guardrails) -> None:
    cfg = _apply_arm(_cfg(), arm)
    assert cfg.harness_mode_for(M) == mode
    assert cfg.guardrails_for(M) == guard


@pytest.mark.parametrize("arm", ["native", "native-guard", "native-guard-synth"])
def test_measured_native_arms_keep_the_bare_prompt(arm: str) -> None:
    """Nights 1-2 measured these arms with the bare prompt; a rerun must too."""
    assert _apply_arm(_cfg(), arm).models.native_prompt == "bare"


def test_native_guard_id_is_native_guard_with_the_identity_prompt() -> None:
    cfg = _apply_arm(_cfg(), "native-guard-id")
    assert cfg.harness_mode_for(M) == "native"
    assert cfg.guardrails_for(M) == _apply_arm(_cfg(), "native-guard").guardrails_for(M)
    assert cfg.models.native_prompt == "identity"


def test_every_arm_reads_the_shipped_template_never_the_operator_workspace(
    tmp_path: Path,
) -> None:
    """The eval measures what a fresh install sends: the repo's Aegis template, pinned by
    the commit. It never reads this machine's IDENTITY.md (a private persona, different
    per machine) -- in the native prompt or in reply synthesis, for any arm."""
    from runtime.eval.cli import _TEMPLATE_WORKSPACE, _tier1_loader

    cfg = AegisConfig.model_validate({"storage": {"workspace": str(tmp_path / "private")}})
    loader = _tier1_loader()  # the loader the eval run builds
    assert loader.root == _TEMPLATE_WORKSPACE != cfg.storage.workspace
    assert "**Name:** Aegis" in loader.load("1").identity


def test_production_defaults_to_the_identity_prompt() -> None:
    cfg = AegisConfig()
    assert cfg.models.native_prompt is None  # unset: defer to the profile, then identity
    assert cfg.native_prompt_for(M) == "identity"


def test_an_arm_beats_a_models_profile_prompt() -> None:
    """Eval arms pin the prompt globally, so a production profile cannot leak into a rerun."""
    cfg = AegisConfig(
        model_profiles={M: ModelProfile(harness_mode="native", native_prompt="bare")}
    )
    assert _apply_arm(cfg, "native-guard-id").native_prompt_for(M) == "identity"
    assert _apply_arm(cfg, "native-guard").native_prompt_for(M) == "bare"


def _local_cfg() -> AegisConfig:
    return AegisConfig.model_validate({"models": {"smart_provider": "ollama", "smart_local": M}})


def _probe(monkeypatch: pytest.MonkeyPatch, *, answers: bool) -> list[str]:
    """Stub the classifier probe; returns the models it was called with."""
    from runtime.eval import cli

    probed: list[str] = []

    async def _error(_cfg: AegisConfig, model: str, _intents: list[str]) -> str | None:
        probed.append(model)
        return None if answers else "ClassifierUnavailableError: invalid_json"

    monkeypatch.setattr(cli, "_classifier_error", _error)
    return probed


@pytest.mark.parametrize("arm", ARMS)
def test_no_arm_preset_touches_the_classifier(arm: str) -> None:
    """Which model classifies is decided in `_run`, once locality is known."""
    assert _apply_arm(_local_cfg(), arm).models.fast == _local_cfg().models.fast


@pytest.mark.parametrize("arm", [a for a in ARMS if a != "bare"])
async def test_a_local_run_classifies_on_the_model_under_test(
    arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Night 4 (Ollama server.log): the default classifier, gemma4:e2b, evicted the model
    under test on every variant, so each variant paid two loads inside the timed window."""
    from runtime.eval.cli import _pin_classifier

    probed = _probe(monkeypatch, answers=True)
    cfg = await _pin_classifier(_apply_arm(_local_cfg(), arm), arm, "local", ["files"])
    assert cfg.models.fast == M
    assert probed == [M]


async def test_a_local_model_that_fails_the_probe_keeps_the_default_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted smoke, 2026-10-27: a model that reasons despite think=False overran the
    classifier's 64-token reply and failed every variant (glm-5.3)."""
    from runtime.eval.cli import _pin_classifier

    _probe(monkeypatch, answers=False)
    cfg = await _pin_classifier(_local_cfg(), "native-guard", "local", ["files"])
    assert cfg.models.fast == _local_cfg().models.fast


@pytest.mark.parametrize(
    ("arm", "provider", "model", "locality"),
    [
        ("native-guard", "ollama", "glm-5.3:cloud", "cloud"),  # holds no local memory
        ("full", "openrouter", M, "hosted"),  # the classifier client is always Ollama
        ("bare", "ollama", M, "local"),  # bare has no classifier
    ],
)
async def test_runs_that_keep_the_default_classifier_never_probe(
    arm: str, provider: str, model: str, locality: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from runtime.eval.cli import _pin_classifier

    probed = _probe(monkeypatch, answers=True)
    cfg = AegisConfig.model_validate({"models": {"smart_provider": provider, "smart_local": model}})
    assert (await _pin_classifier(cfg, arm, locality, ["files"])).models.fast == cfg.models.fast
    assert probed == []


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (None, None),
        ("unavailable", "ClassifierUnavailableError: invalid_json"),
        ("unexpected", "KeyError: 'intent'"),  # must not abort the run
    ],
)
async def test_the_probe_says_why_the_classifier_failed(
    raised: str | None, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from runtime.eval import cli
    from runtime.intent.classifier import ClassifierUnavailableError, IntentClassification

    class _Classifier:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["model"] == M

        async def classify(self, text: str) -> IntentClassification:
            assert text == cli._CLASSIFIER_PROBE
            if raised == "unavailable":
                raise ClassifierUnavailableError("invalid_json")
            if raised == "unexpected":
                raise KeyError("intent")
            return IntentClassification(intent="files", confidence=0.9)

    monkeypatch.setattr(cli, "OllamaClient", lambda _cfg: object())
    monkeypatch.setattr(cli, "ModelBackedClassifier", _Classifier)
    assert await cli._classifier_error(_local_cfg(), M, ["files"]) == expected


@pytest.mark.parametrize("no_prewarm", [False, True])
async def test_run_probes_after_the_warm_up_and_records_the_classifier(
    no_prewarm: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The probe must not pay the cold load (a 30 s read timeout would fail it), and
    --no-prewarm leaves that load to the first task, so it skips the probe."""
    import json

    from runtime.eval import cli
    from runtime.eval.report import TaskResult

    order: list[str] = []

    class _Ollama:
        def __init__(self, _cfg: AegisConfig) -> None: ...

        async def list_local_models(self) -> list[str]:
            return [M]

    async def _prewarm(_client: object, _model: str) -> None:
        order.append("prewarm")

    async def _error(_cfg: AegisConfig, _model: str, _intents: list[str]) -> str | None:
        order.append("probe")
        return None

    async def _run_task(cfg: AegisConfig, _r: object, _t: object, task: object, **_k: object):
        order.append(f"task:{cfg.models.fast}")
        return TaskResult(task_id="t", description="d", variants=())

    class _Registry:
        @staticmethod
        def from_directory(_path: object) -> _Registry:
            return _Registry()

        def all(self) -> list[object]:
            return []

    monkeypatch.setattr(cli, "get_config", _local_cfg)
    monkeypatch.setattr(cli, "OllamaClient", _Ollama)
    monkeypatch.setattr(cli, "prewarm", _prewarm)
    monkeypatch.setattr(cli, "_classifier_error", _error)
    monkeypatch.setattr(cli, "run_task", _run_task)
    monkeypatch.setattr(cli, "SkillRegistry", _Registry)
    repo = Path(__file__).resolve().parent.parent
    argv = ["--arm", "native-guard", "--yes", "--tasks-dir", str(repo / "eval" / "tasks")]
    argv += ["--out-dir", str(tmp_path)] + (["--no-prewarm"] if no_prewarm else [])

    assert await cli._run(cli._parse_args(argv)) == 0

    if no_prewarm:
        fast = _local_cfg().models.fast
        assert "prewarm" not in order and "probe" not in order
        assert order[0] == f"task:{fast}"
    else:
        fast = M
        assert order[:3] == ["prewarm", "probe", f"task:{M}"]
    (written,) = tmp_path.glob("*.json")
    assert json.loads(written.read_text())["classifier_model"] == fast


def test_the_report_records_the_classifier_and_old_files_still_load(tmp_path: Path) -> None:
    stamp = datetime(2026, 10, 27, tzinfo=UTC)
    report = EvalReport(
        provider="ollama", model=M, started_at=stamp, tasks=(), classifier_model=M
    )
    path = write_json(report, tmp_path)
    assert EvalReport.model_validate_json(path.read_text()).classifier_model == M
    old = EvalReport(provider="ollama", model=M, started_at=stamp, tasks=())
    legacy = old.model_dump(mode="json")
    legacy.pop("classifier_model")
    assert EvalReport.model_validate(legacy).classifier_model is None


def test_bare_stays_the_eval_only_control() -> None:
    from runtime.eval import bare

    assert "bare" in ARMS
    assert _apply_arm(_cfg(), "bare") == _cfg()
    assert _variant_runner("bare") is bare.run_variant_bare


@pytest.mark.parametrize(
    "arm", ["native", "native-guard", "native-guard-synth", "native-guard-id", "bare"]
)
def test_native_arms_need_ollama(arm: str) -> None:
    assert _arm_unsupported_reason(arm, "openrouter") is not None
    assert _arm_unsupported_reason(arm, "ollama") is None


def test_result_files_do_not_collide_across_arms(tmp_path: Path) -> None:
    stamp = datetime(2026, 10, 6, tzinfo=UTC)
    names = {
        write_json(
            EvalReport(provider="ollama", model=M, started_at=stamp, tasks=(), arm=arm), tmp_path
        ).name
        for arm in ARMS
    }
    assert len(names) == len(ARMS)
    assert "2026-10-06T00-00-00Z-ollama-gemma4-e4b-mlx.json" in names  # full: unchanged


async def test_eval_refuses_to_run_while_the_model_is_the_placeholder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuses before asking Ollama for its models or prompting to proceed."""
    from runtime.config import ModelConfig
    from runtime.eval import cli

    cfg = AegisConfig(models=ModelConfig(smart_provider="ollama", smart_local="YOUR_MODEL"))
    monkeypatch.setattr(cli, "get_config", lambda: cfg)

    def _no_ollama(*_a: object, **_k: object) -> None:
        raise AssertionError("contacted Ollama before refusing")

    monkeypatch.setattr(cli, "OllamaClient", _no_ollama)
    repo = Path(__file__).resolve().parent.parent
    args = cli._parse_args(["--arm", "full", "--tasks-dir", str(repo / "eval" / "tasks")])

    assert await cli._run(args) == 2
    assert "YOUR_MODEL" in capsys.readouterr().err
