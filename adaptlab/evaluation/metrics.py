"""Métricas de avaliação: macro-F1 de intent, F1 de entidades, validade JSON, recall de alto risco.

Regras de contagem (iguais para todas as abordagens):
- Uma saída inválida (JSON ou schema) conta como erro em todos os campos.
  Para o intent, a predição é o marcador ``INVALID`` (não é uma classe válida).
- Alto risco = ``requires_human`` ou ``priority == "alta"``. O recall de alto risco
  conta as saídas inválidas como não detectadas.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from adaptlab.data.quality import normalize_entity
from adaptlab.schema import ENTITY_KEYS, INTENTS

INVALID = "__invalid__"


@dataclass(frozen=True)
class Prediction:
    """Resultado de uma abordagem para um exemplo. ``label`` é None se a saída for inválida."""

    example_id: str
    label: dict[str, Any] | None
    error: str | None
    latency_s: float
    output_tokens: int


def is_high_risk(label: dict[str, Any]) -> bool:
    return bool(label.get("requires_human")) or label.get("priority") == "alta"


def entity_set(label: dict[str, Any] | None) -> set[tuple[str, str]]:
    if label is None:
        return set()
    entities = label.get("entities") or {}
    return {
        (key, normalize_entity(entities[key]))
        for key in ENTITY_KEYS
        if isinstance(entities.get(key), str) and entities.get(key)
    }


def _intent_of(pred: Prediction) -> str:
    if pred.label is None:
        return INVALID
    return str(pred.label.get("intent", INVALID))


def compute_metrics(gold: Sequence[dict[str, Any]], preds: Sequence[Prediction]) -> dict[str, Any]:
    """Calcula todas as métricas sobre o mesmo conjunto (gold e predições alinhados por id)."""
    if [g["id"] for g in gold] != [p.example_id for p in preds]:
        raise ValueError("gold e predições não estão alinhados pelo id")
    y_true = [g["intent"] for g in gold]
    y_pred = [_intent_of(p) for p in preds]

    # Classes: as que aparecem no gold ou nas predições (como o sklearn por omissão), sem o marcador INVALID.
    present = sorted(set(y_true) | {p for p in y_pred if p != INVALID})
    macro_f1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(INTENTS) + [INVALID])

    tp = fp = fn = 0
    for g, p in zip(gold, preds, strict=True):
        gold_set = entity_set(g["label"])
        pred_set = entity_set(p.label)
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    entity_precision = tp / (tp + fp) if tp + fp else 0.0
    entity_recall = tp / (tp + fn) if tp + fn else 0.0
    entity_f1 = (
        2 * entity_precision * entity_recall / (entity_precision + entity_recall)
        if entity_precision + entity_recall
        else 0.0
    )

    gold_high = [is_high_risk(g["label"]) for g in gold]
    pred_high = [p.label is not None and is_high_risk(p.label) for p in preds]
    positives = sum(gold_high)
    detected = sum(1 for gh, ph in zip(gold_high, pred_high, strict=True) if gh and ph)
    high_risk_recall = detected / positives if positives else 0.0

    valid = sum(1 for p in preds if p.label is not None)
    latencies = sorted(p.latency_s for p in preds)
    return {
        "n": len(gold),
        "macro_f1_intent": round(float(macro_f1), 4),
        "entity_f1": round(float(entity_f1), 4),
        "entity_precision": round(entity_precision, 4),
        "entity_recall": round(entity_recall, 4),
        "json_valid_rate": round(valid / len(preds), 4) if preds else 0.0,
        "high_risk_recall": round(high_risk_recall, 4),
        "high_risk_positives": positives,
        "latency_mean_s": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "latency_p50_s": round(latencies[len(latencies) // 2], 4) if latencies else 0.0,
        "output_tokens_mean": round(sum(p.output_tokens for p in preds) / len(preds), 2) if preds else 0.0,
        "confusion_labels": list(INTENTS) + [INVALID],
        "confusion": matrix.tolist(),
        "errors_by_gold_intent": _errors_by_intent(y_true, y_pred),
    }


def _errors_by_intent(y_true: list[str], y_pred: list[str]) -> dict[str, int]:
    errors: Counter[str] = Counter()
    for t, p in zip(y_true, y_pred, strict=True):
        if t != p:
            errors[t] += 1
    return dict(sorted(errors.items()))


def bootstrap_ci(
    gold: Sequence[dict[str, Any]],
    preds: Sequence[Prediction],
    metric: str,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """IC bootstrap percentil sobre exemplos de teste, para uma métrica escalar de ``compute_metrics``."""
    rng = np.random.default_rng(seed)
    n = len(gold)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        sub_gold = [gold[i] for i in idx]
        sub_preds = [preds[i] for i in idx]
        values.append(compute_metrics(sub_gold, sub_preds)[metric])
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return round(float(low), 4), round(float(high), 4)
