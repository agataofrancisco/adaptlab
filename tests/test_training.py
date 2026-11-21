"""Testes de treino: formato conversacional, configuração, guarda de CUDA e dry-run em CPU."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from adaptlab.schema import Label
from adaptlab.training.config import (
    DEFAULT_TRAIN_CONFIG,
    ConfigError,
    load_train_config,
    parse_train_config,
)
from adaptlab.training.dataset import read_jsonl, spread_sample
from adaptlab.training.format import SYSTEM_PROMPT, build_prompt, to_prompt_completion
from adaptlab.training.tracking import EnvironmentRequirementError, require_quantization_support

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _record() -> dict:
    return {
        "message": "a net ta fraca no Kilamba desde ontem",
        "label": {
            "intent": "avaria_rede",
            "priority": "alta",
            "entities": {"numero": None, "valor": None, "localidade": "Kilamba", "data": "desde ontem"},
            "requires_human": False,
        },
    }


def test_prompt_completion_has_system_user_and_assistant_json() -> None:
    row = to_prompt_completion(_record())
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert row["prompt"][0]["content"] == SYSTEM_PROMPT
    assert row["prompt"][1]["content"] == "a net ta fraca no Kilamba desde ontem"
    assert row["completion"][0]["role"] == "assistant"
    parsed = Label.model_validate_json(row["completion"][0]["content"])
    assert parsed.intent == "avaria_rede"


def test_invalid_label_never_reaches_the_trainer() -> None:
    bad = _record()
    bad["label"]["intent"] = "inventado"
    with pytest.raises(ValueError, match="label inválido"):
        to_prompt_completion(bad)


@pytest.mark.parametrize("message", [None, 3, ["a"]])
def test_non_text_message_is_rejected(message: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_prompt(message)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        to_prompt_completion({"message": message, "label": _record()["label"]})


def test_default_config_loads_and_pins_revision() -> None:
    config = load_train_config(DEFAULT_TRAIN_CONFIG)
    assert config.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert len(config.model_revision) == 40
    assert config.training.completion_only_loss is True
    assert config.lora.r == 16 and config.lora.alpha == 32


def test_invalid_config_values_fail_explicitly() -> None:
    raw = yaml.safe_load(DEFAULT_TRAIN_CONFIG.read_text(encoding="utf-8"))
    broken = copy.deepcopy(raw)
    broken["quantization"] = "int2"
    with pytest.raises(ConfigError, match="quantization"):
        parse_train_config(broken)
    broken = copy.deepcopy(raw)
    broken["lora"]["r"] = 0
    with pytest.raises(ConfigError, match="lora.r|'r'"):
        parse_train_config(broken)


def test_qlora_without_cuda_fails_with_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = yaml.safe_load(DEFAULT_TRAIN_CONFIG.read_text(encoding="utf-8"))
    raw["quantization"] = "nf4"
    config = parse_train_config(raw)
    monkeypatch.setattr(
        "adaptlab.training.tracking.detect_device",
        lambda: {"device": "cpu", "cuda_available": False, "gpu_name": None},
    )
    with pytest.raises(EnvironmentRequirementError, match="notebook GPU"):
        require_quantization_support(config)


def test_no_quantization_needs_no_gpu() -> None:
    config = load_train_config(DEFAULT_TRAIN_CONFIG)
    require_quantization_support(config)  # não lança


def test_spread_sample_is_deterministic_and_covers_range() -> None:
    records = [{"i": i} for i in range(100)]
    picked = spread_sample(records, 8)
    assert len(picked) == 8
    assert picked == spread_sample(records, 8)
    assert picked[0]["i"] == 0 and picked[-1]["i"] >= 87


@pytest.mark.skipif(not (PROCESSED / "train.jsonl").exists(), reason="dataset não gerado")
def test_dry_run_masks_prompt_and_trains_two_steps(tmp_path: Path) -> None:
    """Dry-run real: tokenizer do modelo configurado, modelo tiny local, 1 passo LoRA."""
    from adaptlab.training.dryrun import run_dry_run

    raw = yaml.safe_load(DEFAULT_TRAIN_CONFIG.read_text(encoding="utf-8"))
    raw["dry_run"]["sample_size"] = 4
    raw["dry_run"]["max_steps"] = 1
    config = parse_train_config(raw)
    report = run_dry_run(config, tmp_path)
    assert report["status"] == "ok"
    assert report["masking"]["masked_prompt_tokens"] > 0
    assert report["masking"]["contiguous_labels"] is True
    assert report["masking"]["labeled_text"].lstrip().startswith("{")
    assert report["steps"] == 1


def test_read_jsonl_reports_bad_line(tmp_path: Path) -> None:
    from adaptlab.training.dataset import DatasetFormatError

    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\nnão é json\n', encoding="utf-8")
    with pytest.raises(DatasetFormatError, match=":2:"):
        read_jsonl(path)
