"""Predictores da API: modo real (modelo + adapter) e modo demo (fixtures pré-calculadas).

Honestidade: no modo demo cada resposta leva ``mode: "precomputed_demo"``. Mensagens fora
do conjunto pré-calculado recebem 503 com a explicação. As fixtures vêm de uma run medida
(``scripts/build_demo_fixtures.py``) e são rotuladas com a sua origem.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from adaptlab.data.quality import normalize_text
from adaptlab.schema import parse_label

MODE_REAL = "real"
MODE_DEMO = "precomputed_demo"
MAX_MESSAGE_CHARS = 2000

# (messages) -> (texto, tokens gerados)
Generate = Callable[[list[dict[str, str]]], tuple[str, int]]


@dataclass(frozen=True)
class Outcome:
    status: int
    body: dict[str, Any]


class Predictor(Protocol):
    mode: str
    device: str
    model_loaded: bool
    model_names: tuple[str, ...]

    def predict(self, message: str, model: str) -> Outcome: ...


def _risk(label: dict[str, Any]) -> bool:
    return bool(label.get("requires_human")) or label.get("priority") == "alta"


class DemoPredictor:
    """Serve apenas as mensagens do ficheiro de fixtures. Nunca chama um modelo."""

    mode = MODE_DEMO
    device = "cpu"
    model_loaded = False
    model_names: tuple[str, ...] = ("regras_fixture",)

    def __init__(self, fixtures_path: Path) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self.fixtures_path = fixtures_path
        if fixtures_path.exists():
            for line in fixtures_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get("message"), str):
                    self._rows[normalize_text(row["message"])] = row

    @property
    def fixture_count(self) -> int:
        return len(self._rows)

    def predict(self, message: str, model: str) -> Outcome:
        row = self._rows.get(normalize_text(message))
        if row is None:
            return Outcome(
                503,
                {
                    "mode": MODE_DEMO,
                    "error": "demo_message_not_precomputed",
                    "explanation": (
                        "Modo demo sem GPU: esta mensagem não está no conjunto pré-calculado e a "
                        "inferência real não está disponível. Corre a API com ADAPTLAB_MODE=real "
                        "num ambiente com o modelo."
                    ),
                    "requires_human": True,
                },
            )
        label = row["label"]
        return Outcome(
            200,
            {
                "mode": MODE_DEMO,
                "model": row.get("model", "regras_fixture"),
                "model_requested": model,
                "label": label,
                "error": None,
                "requires_human": _risk(label),
                "latency_ms": None,
                "fixture_source": row.get("source"),
                "note": "Resposta pré-calculada (fixture). Não é uma inferência do modelo adaptado.",
            },
        )


class RealPredictor:
    """Modelo base e adaptado com decoding greedy. ``generators`` mapeia nome -> gerador."""

    mode = MODE_REAL

    def __init__(self, generators: dict[str, Generate], device: str) -> None:
        if not generators:
            raise ValueError("RealPredictor precisa de pelo menos um gerador")
        self._generators = generators
        self.device = device
        self.model_loaded = True
        self.model_names = tuple(generators)

    def predict(self, message: str, model: str) -> Outcome:
        from adaptlab.training.format import build_prompt

        if model not in self._generators:
            return Outcome(
                400,
                {"mode": MODE_REAL, "error": "unknown_model", "allowed": list(self._generators)},
            )
        started = time.perf_counter()
        raw, tokens = self._generators[model](build_prompt(message))
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        label, error = parse_label(raw)
        if label is None:
            return Outcome(
                200,
                {
                    "mode": MODE_REAL,
                    "model": model,
                    "label": None,
                    "error": "invalid_model_output",
                    "detail": error,
                    "requires_human": True,
                    "latency_ms": latency_ms,
                    "output_tokens": tokens,
                },
            )
        dumped = label.model_dump(mode="json")
        return Outcome(
            200,
            {
                "mode": MODE_REAL,
                "model": model,
                "label": dumped,
                "error": None,
                "requires_human": _risk(dumped),
                "latency_ms": latency_ms,
                "output_tokens": tokens,
                "confidence": None,
            },
        )
