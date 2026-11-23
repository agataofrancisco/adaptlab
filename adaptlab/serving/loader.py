"""Carrega o predictor real. Lê o caminho do adapter de ``ADAPTLAB_ADAPTER_DIR``.

Só é chamado com ``ADAPTLAB_MODE=real``. Não faz download silencioso de nada além do
que o ``ChatGenerator`` já faz (modelo base pelo commit fixado em ``configs/train.yaml``).
"""

from __future__ import annotations

import os
from pathlib import Path

from adaptlab.serving.predictor import RealPredictor
from adaptlab.training.config import load_train_config


def load_real_predictor() -> RealPredictor:
    from adaptlab.evaluation.generation import ChatGenerator

    adapter_env = os.environ.get("ADAPTLAB_ADAPTER_DIR", "").strip()
    if not adapter_env:
        raise RuntimeError("ADAPTLAB_ADAPTER_DIR não definido (pasta do adapter exportado)")
    adapter_dir = Path(adapter_env)
    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError(f"adapter_config.json não encontrado em {adapter_dir}")

    config = load_train_config()
    base = ChatGenerator(config.model_name, config.model_revision)
    adapted = ChatGenerator(config.model_name, config.model_revision, adapter_dir=adapter_dir)
    return RealPredictor({"base": base, "adapted": adapted}, device=base.device)
