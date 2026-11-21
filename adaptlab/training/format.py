"""Formato conversacional de treino: system + user (mensagem) -> assistant (JSON).

O mesmo ``SYSTEM_PROMPT`` é usado no treino, na avaliação zero-shot e no serviço,
para que a comparação entre abordagens seja justa.

Formato ``prompt``/``completion`` (listas de mensagens). Com este formato, o TRL
aplica o chat template ao prompt e à completion e, por omissão, calcula a loss
apenas sobre a completion (ver ``train.py`` e docs/learn/04).
"""

from __future__ import annotations

import json
from typing import Any

from adaptlab.schema import INTENTS, PRIORITIES, label_to_json, parse_label

SYSTEM_PROMPT = (
    "És o classificador de apoio ao cliente da Mukua Telecom. "
    "Lês a mensagem do cliente e respondes apenas com um objecto JSON, sem texto extra, "
    "com estas chaves: "
    f'"intent" (um de: {", ".join(INTENTS)}); '
    f'"priority" (um de: {", ".join(PRIORITIES)}); '
    '"entities" (objecto com "numero", "valor", "localidade" e "data", '
    "cada um com o texto literal da mensagem ou null); "
    '"requires_human" (true se for preciso um humano, por exemplo ameaças, '
    "dados pessoais ou reembolsos). Nunca inventes entidades que não estão na mensagem."
)


def build_prompt(message: str) -> list[dict[str, str]]:
    """Mensagens de entrada (system + user). ``message`` tem de ser texto."""
    if not isinstance(message, str):
        raise TypeError("message tem de ser str")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]


def to_prompt_completion(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Converte um exemplo do dataset (``message`` + ``label``) para prompt/completion.

    Valida o label com o schema antes de o transformar em alvo de treino, para que
    nenhum exemplo inválido chegue ao modelo.
    """
    message = record.get("message")
    if not isinstance(message, str):
        raise ValueError("exemplo sem 'message' textual")
    label, error = parse_label(json.dumps(record.get("label"), ensure_ascii=False))
    if label is None:
        raise ValueError(f"label inválido: {error}")
    return {
        "prompt": build_prompt(message),
        "completion": [{"role": "assistant", "content": label_to_json(label)}],
    }
