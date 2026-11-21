"""Schema de saída (label) partilhado por dados, treino, avaliação e API.

Todas as funções aqui são puras e defensivas: recebem texto arbitrário
(vindo de um modelo ou de um cliente) e nunca lançam excepção; devolvem
``None`` e um motivo quando a validação falha.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

Intent = Literal[
    "recarga",
    "avaria_rede",
    "faturacao",
    "portabilidade",
    "cancelamento",
    "pacote_dados",
    "outro",
]
Priority = Literal["baixa", "media", "alta"]

INTENTS: tuple[str, ...] = (
    "recarga",
    "avaria_rede",
    "faturacao",
    "portabilidade",
    "cancelamento",
    "pacote_dados",
    "outro",
)
PRIORITIES: tuple[str, ...] = ("baixa", "media", "alta")
ENTITY_KEYS: tuple[str, ...] = ("numero", "valor", "localidade", "data")


class Entities(BaseModel):
    """Entidades extraídas. Um campo ausente na mensagem é ``null``, nunca inventado."""

    model_config = ConfigDict(extra="forbid")

    numero: str | None = None
    valor: str | None = None
    localidade: str | None = None
    data: str | None = None


class Label(BaseModel):
    """Saída estruturada que o modelo deve devolver (e que o dataset rotula)."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    priority: Priority
    entities: Entities
    requires_human: bool


def extract_first_json_object(text: object) -> dict[str, Any] | None:
    """Devolve o primeiro objecto JSON de ``text`` ou ``None``.

    Tenta ``raw_decode`` a partir de cada ``{``, por ordem. Aceita texto à volta
    do JSON (um modelo pode escrever prosa antes ou depois).
    """
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    return None


def parse_label(text: object) -> tuple[Label | None, str | None]:
    """Valida a saída de um modelo. Devolve ``(label, None)`` ou ``(None, motivo)``.

    Este é o único parsing usado por todas as abordagens (zero-shot, few-shot,
    LoRA), para que a comparação seja justa.
    """
    obj = extract_first_json_object(text)
    if obj is None:
        return None, "no_json_object"
    try:
        return Label.model_validate(obj), None
    except ValidationError as exc:
        return None, f"schema_error:{exc.error_count()}"


def label_to_json(label: Label) -> str:
    """Serialização canónica (ordem fixa das chaves) usada como alvo de treino."""
    return json.dumps(label.model_dump(mode="json"), ensure_ascii=False)
