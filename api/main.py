"""API FastAPI do AdaptLab: /predict, /model-card, /health, /metrics.

Modo escolhido por ``ADAPTLAB_MODE`` (``demo`` por omissão, ``real`` precisa do modelo).
Os testes criam a app com ``create_app(predictor)`` e não dependem do ambiente.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from adaptlab.paths import PROJECT_ROOT
from adaptlab.serving.predictor import MAX_MESSAGE_CHARS, DemoPredictor, Predictor

MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_card.md"
FIXTURES_PATH = PROJECT_ROOT / "data" / "demo" / "precomputed.jsonl"


class PredictRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("mensagem vazia")
        return stripped


class Metrics:
    """Contadores em memória (por processo). Não substituem um sistema de observabilidade."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.by_status: dict[int, int] = {}
        self.invalid_output = 0
        self.latency_ms_sum = 0.0
        self.latency_count = 0

    def record(self, status: int, body: dict[str, Any]) -> None:
        with self._lock:
            self.requests += 1
            self.by_status[status] = self.by_status.get(status, 0) + 1
            if body.get("error") == "invalid_model_output":
                self.invalid_output += 1
            latency = body.get("latency_ms")
            if isinstance(latency, (int, float)):
                self.latency_ms_sum += float(latency)
                self.latency_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self.requests,
                "requests_by_status": {str(k): v for k, v in sorted(self.by_status.items())},
                "invalid_model_output_total": self.invalid_output,
                "invalid_model_output_rate": round(self.invalid_output / self.requests, 4) if self.requests else None,
                "latency_ms_mean": round(self.latency_ms_sum / self.latency_count, 1) if self.latency_count else None,
            }


def create_app(predictor: Predictor, model_card_path: Path = MODEL_CARD_PATH) -> FastAPI:
    app = FastAPI(title="AdaptLab", version="0.1.0")
    metrics = Metrics()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": predictor.mode,
            "device": predictor.device,
            "model_loaded": predictor.model_loaded,
            "models": list(predictor.model_names),
        }

    @app.post("/predict")
    def predict(
        payload: PredictRequest,
        model: str = Query(default="adapted", pattern="^(base|adapted|regras_fixture)$"),
    ) -> JSONResponse:
        outcome = predictor.predict(payload.message, model)
        metrics.record(outcome.status, outcome.body)
        return JSONResponse(status_code=outcome.status, content=outcome.body)

    @app.get("/model-card")
    def model_card() -> dict[str, Any]:
        if not model_card_path.exists():
            raise HTTPException(
                status_code=404,
                detail="model card ainda não gerado (scripts/export_adapter.py --write-card)",
            )
        return {"format": "markdown", "content": model_card_path.read_text(encoding="utf-8")}

    @app.get("/metrics")
    def get_metrics() -> dict[str, Any]:
        return metrics.snapshot()

    return app


def build_default_predictor() -> Predictor:
    mode = os.environ.get("ADAPTLAB_MODE", "demo").strip().lower()
    if mode == "demo":
        return DemoPredictor(FIXTURES_PATH)
    if mode == "real":
        from adaptlab.serving.loader import load_real_predictor

        return load_real_predictor()
    raise RuntimeError(f"ADAPTLAB_MODE inválido: {mode!r} (use 'demo' ou 'real')")


app = create_app(build_default_predictor())


if __name__ == "__main__":
    # Só localhost por omissão: a API de demonstração não tem autenticação (ver README, "Autenticação").
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
