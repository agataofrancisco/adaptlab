"""Tracking JSON por run (``runs/<id>/``) e verificação do ambiente.

Cada run guarda: configuração completa, versões das bibliotecas, GPU/CPU,
VRAM máxima medida, tempo e curvas. Nada aqui importa bibliotecas pesadas
no topo do módulo, para que a verificação de ambiente funcione sem torch.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptlab.training.config import TrainConfig

TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
)


class EnvironmentRequirementError(RuntimeError):
    """Ambiente incompatível com a configuração pedida (por exemplo, QLoRA sem CUDA)."""


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def detect_device() -> dict[str, Any]:
    """Dispositivo disponível. Importa torch só aqui; falha de import não é fatal."""
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
    except ImportError:
        info.update(device="none", cuda_available=False, gpu_name=None)
        return info
    cuda = bool(torch.cuda.is_available())
    info.update(
        device="cuda" if cuda else "cpu",
        cuda_available=cuda,
        gpu_name=torch.cuda.get_device_name(0) if cuda else None,
        torch_cuda=torch.version.cuda,
    )
    return info


def require_quantization_support(config: TrainConfig) -> None:
    """QLoRA 4-bit (nf4) exige CUDA e bitsandbytes. Falha com mensagem clara; não faz fallback silencioso."""
    if config.quantization != "nf4":
        return
    device = detect_device()
    if not device["cuda_available"]:
        raise EnvironmentRequirementError(
            "QLoRA (quantization: nf4) precisa de GPU CUDA. Este PC não tem CUDA. "
            "Corre o treino real no notebook GPU (notebooks/train_colab.ipynb, Colab/Kaggle) "
            "ou usa 'quantization: none' em configs/train.yaml."
        )
    try:
        importlib.import_module("bitsandbytes")
    except ImportError as exc:
        raise EnvironmentRequirementError(
            "QLoRA precisa de bitsandbytes, que só está instalado no notebook GPU."
        ) from exc


def new_run_dir(runs_dir: Path, prefix: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"{prefix}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def write_run_metadata(run_dir: Path, config: TrainConfig, mode: str) -> None:
    write_json(
        run_dir / "config.json",
        {"mode": mode, "config": config.raw, "model_revision": config.model_revision},
    )
    write_json(run_dir / "environment.json", {**detect_device(), "packages": package_versions()})
