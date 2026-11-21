"""Baseline determinística: palavras-chave e regex. Piso mínimo que o LoRA tem de bater.

Limitação declarada: a lista de localidades e de expressões temporais é conhecida pelo
autor (é a mesma do catálogo sintético). Isto favorece a baseline face a um cliente real.
"""

from __future__ import annotations

import re
from typing import Any

from adaptlab.data.quality import normalize_text

# Ordem importa: a primeira regra que bater ganha.
INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("portabilidade", ("portabilidade", "portar", "portado", "mudar de operadora", "mudar o meu número")),
    ("cancelamento", ("cancelar", "cancel", "cancelem", "cancelamento")),
    ("faturacao", ("fatura", "factura", "cobraram", "debitado", "debitaram", "bill", "cobrança", "cobranca")),
    ("recarga", ("recarga", "recarregar", "recharge", "top up", "crédito", "credito")),
    (
        "avaria_rede",
        (
            "rede",
            "net ",
            "sinal",
            "avaria",
            "dados móveis",
            "dados moveis",
            "data is down",
            "network",
            "parado",
            "caiu",
            "funciona nada",
        ),
    ),
    ("pacote_dados", ("pacote", "bundle", "gigas", "consumo", "activate the data")),
)
ESCALATION_KEYWORDS: tuple[str, ...] = (
    "supervisor",
    "responsável",
    "responsavel",
    "queixa",
    "denunci",
    "imprensa",
    "vergonha",
    "reembolso",
    "dados pessoais",
    "privacidade",
    "urgente",
    "duas vezes",
    "apagar",
    "reguladora",
)
LOCALITIES: tuple[str, ...] = (
    "Kilamba",
    "Talatona",
    "Viana",
    "Cacuaco",
    "Benguela",
    "Lubango",
    "Huambo",
    "Cabinda",
    "Malanje",
    "Uíge",
    "Namibe",
    "Sumbe",
)
TIME_EXPRESSIONS: tuple[str, ...] = (
    "anteontem",
    "ontem",
    "hoje",
    "esta manhã",
    "segunda-feira",
    "terça-feira",
    "sábado",
    "há três dias",
    "dia 12",
    "dia 3",
)
NUMBER_RE = re.compile(r"(?:\+244\s?)?9\d{2}\s?\d{3}\s?\d{3}|\b9\d{8}\b")
VALUE_RE = re.compile(r"\d[\d.,]*\s?(?:mil\s+kwanzas|kwanzas|kz)", re.IGNORECASE)


def _find_intent(text: str) -> str:
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return intent
    return "outro"


def _find_name(message: str, names: tuple[str, ...]) -> str | None:
    normalized = normalize_text(message)
    for name in names:
        if normalize_text(name) in normalized:
            return name
    return None


def rules_label(message: str) -> dict[str, Any]:
    """Devolve sempre um label válido (schema) para uma mensagem de texto."""
    if not isinstance(message, str):
        message = ""
    text = normalize_text(message)
    intent = _find_intent(text)
    requires_human = any(keyword in text for keyword in ESCALATION_KEYWORDS)
    if requires_human:
        priority = "alta"
    elif intent == "avaria_rede":
        priority = "alta"
    elif intent in ("faturacao", "cancelamento", "pacote_dados"):
        priority = "media"
    else:
        priority = "baixa"

    number = NUMBER_RE.search(message)
    value = VALUE_RE.search(message)
    time_name = _find_name(message, TIME_EXPRESSIONS)
    return {
        "intent": intent,
        "priority": priority,
        "entities": {
            "numero": number.group(0) if number else None,
            "valor": value.group(0) if value else None,
            "localidade": _find_name(message, LOCALITIES),
            "data": time_name,
        },
        "requires_human": requires_human,
    }
