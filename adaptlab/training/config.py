"""Configuração de treino carregada de ``configs/train.yaml`` com validação explícita.

Os valores inválidos levantam ``ConfigError`` com a chave em causa; nunca são
corrigidos em silêncio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"
QUANTIZATION_MODES = ("none", "nf4")


class ConfigError(ValueError):
    """Configuração de treino inválida."""


@dataclass(frozen=True)
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    bias: str


@dataclass(frozen=True)
class TrainingSettings:
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    lr_scheduler_type: str
    logging_steps: int
    eval_strategy: str
    save_strategy: str
    seed: int
    completion_only_loss: bool


@dataclass(frozen=True)
class DryRunSettings:
    sample_size: int
    max_steps: int
    tiny_hidden_size: int
    tiny_layers: int
    tiny_heads: int
    tiny_kv_heads: int
    tiny_intermediate: int


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    model_revision: str
    max_length: int
    data_dir: Path
    train_file: str
    dev_file: str
    lora: LoraSettings
    quantization: str
    training: TrainingSettings
    dry_run: DryRunSettings
    runs_dir: Path
    raw: dict[str, Any]


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"secção '{key}' em falta ou inválida")
    return value


def _int(section: dict[str, Any], key: str, minimum: int = 0) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"'{key}' tem de ser um inteiro >= {minimum} (recebido {value!r})")
    return value


def _float(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"'{key}' tem de ser um número (recebido {value!r})")
    return float(value)


def _str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' tem de ser texto não vazio (recebido {value!r})")
    return value


def _bool(section: dict[str, Any], key: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' tem de ser true/false (recebido {value!r})")
    return value


def parse_train_config(data: dict[str, Any]) -> TrainConfig:
    model = _section(data, "model")
    lora = _section(data, "lora")
    training = _section(data, "training")
    dry = _section(data, "dry_run")
    data_section = _section(data, "data")
    tracking = _section(data, "tracking")

    quantization = data.get("quantization")
    if quantization not in QUANTIZATION_MODES:
        raise ConfigError(f"'quantization' tem de ser um de {QUANTIZATION_MODES} (recebido {quantization!r})")

    targets = lora.get("target_modules")
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
        raise ConfigError("'lora.target_modules' tem de ser uma lista de nomes")

    if _int(lora, "r", 1) > 256:
        raise ConfigError("'lora.r' acima de 256 não faz sentido para este projecto")

    return TrainConfig(
        model_name=_str(model, "name"),
        model_revision=_str(model, "revision"),
        max_length=_int(model, "max_length", 16),
        data_dir=PROJECT_ROOT / _str(data_section, "dir"),
        train_file=_str(data_section, "train_file"),
        dev_file=_str(data_section, "dev_file"),
        lora=LoraSettings(
            r=_int(lora, "r", 1),
            alpha=_int(lora, "alpha", 1),
            dropout=_float(lora, "dropout"),
            target_modules=tuple(targets),
            bias=_str(lora, "bias"),
        ),
        quantization=quantization,
        training=TrainingSettings(
            learning_rate=_float(training, "learning_rate"),
            num_train_epochs=_float(training, "num_train_epochs"),
            per_device_train_batch_size=_int(training, "per_device_train_batch_size", 1),
            gradient_accumulation_steps=_int(training, "gradient_accumulation_steps", 1),
            warmup_ratio=_float(training, "warmup_ratio"),
            lr_scheduler_type=_str(training, "lr_scheduler_type"),
            logging_steps=_int(training, "logging_steps", 1),
            eval_strategy=_str(training, "eval_strategy"),
            save_strategy=_str(training, "save_strategy"),
            seed=_int(training, "seed", 0),
            completion_only_loss=_bool(training, "completion_only_loss"),
        ),
        dry_run=DryRunSettings(
            sample_size=_int(dry, "sample_size", 1),
            max_steps=_int(dry, "max_steps", 1),
            tiny_hidden_size=_int(dry, "tiny_hidden_size", 8),
            tiny_layers=_int(dry, "tiny_layers", 1),
            tiny_heads=_int(dry, "tiny_heads", 1),
            tiny_kv_heads=_int(dry, "tiny_kv_heads", 1),
            tiny_intermediate=_int(dry, "tiny_intermediate", 8),
        ),
        runs_dir=PROJECT_ROOT / _str(tracking, "runs_dir"),
        raw=data,
    )


def load_train_config(path: Path = DEFAULT_TRAIN_CONFIG) -> TrainConfig:
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"não consigo ler {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} não contém um mapa YAML")
    return parse_train_config(data)
