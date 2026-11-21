"""Execução das abordagens sobre um split. Todas usam o mesmo parsing (``parse_label``).

- ``rules``: baseline determinística (sem modelo).
- ``zero_shot`` / ``lora``: modelo com ``SYSTEM_PROMPT`` e a mensagem.
- ``few_shot``: igual, mais k exemplos fixos do treino como turnos anteriores.

O gerador é injectado (``Generate``), para que os testes usem um falso e a execução real
use ``adaptlab.evaluation.generation.ChatGenerator``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from random import Random
from typing import Any

from adaptlab.evaluation.metrics import INVALID, Prediction
from adaptlab.evaluation.rules import rules_label
from adaptlab.schema import INTENTS, label_to_json, parse_label
from adaptlab.training.format import build_prompt

# (messages) -> (texto gerado, número de tokens gerados)
Generate = Callable[[list[dict[str, str]]], tuple[str, int]]

APPROACHES: tuple[str, ...] = ("rules", "zero_shot", "few_shot", "lora")


def select_few_shot(train: Sequence[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    """k exemplos fixos do treino, com intents distintos (estratificação simples por intent)."""
    by_intent: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(train, key=lambda r: r["id"]):
        by_intent.setdefault(record["intent"], []).append(record)
    rng = Random(f"few_shot:{seed}")
    intents = [intent for intent in INTENTS if intent in by_intent]
    rng.shuffle(intents)
    chosen = [rng.choice(by_intent[intent]) for intent in intents[:k]]
    return chosen


def few_shot_turns(examples: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Turnos user/assistant com as respostas canónicas (mesmo formato do treino)."""
    from adaptlab.schema import Label

    turns: list[dict[str, str]] = []
    for record in examples:
        label = Label.model_validate(record["label"])
        turns.append({"role": "user", "content": record["message"]})
        turns.append({"role": "assistant", "content": label_to_json(label)})
    return turns


def run_rules(records: Sequence[dict[str, Any]]) -> list[Prediction]:
    predictions: list[Prediction] = []
    for record in records:
        started = time.perf_counter()
        label = rules_label(record["message"])
        elapsed = time.perf_counter() - started
        predictions.append(Prediction(record["id"], label, None, elapsed, 0))
    return predictions


def run_model(
    records: Sequence[dict[str, Any]],
    generate: Generate,
    prefix_turns: Sequence[dict[str, str]] = (),
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for record in records:
        messages = build_prompt(record["message"])
        if prefix_turns:
            system, user = messages[0], messages[1]
            messages = [system, *prefix_turns, user]
        started = time.perf_counter()
        raw, output_tokens = generate(messages)
        elapsed = time.perf_counter() - started
        label, error = parse_label(raw)
        predictions.append(
            Prediction(
                example_id=record["id"],
                label=label.model_dump(mode="json") if label is not None else None,
                error=error,
                latency_s=elapsed,
                output_tokens=output_tokens,
            )
        )
    return predictions


def predictions_to_rows(predictions: Sequence[Prediction]) -> list[dict[str, Any]]:
    return [
        {
            "id": p.example_id,
            "label": p.label,
            "intent_pred": p.label["intent"] if p.label else INVALID,
            "error": p.error,
            "latency_s": round(p.latency_s, 6),
            "output_tokens": p.output_tokens,
        }
        for p in predictions
    ]


def write_predictions(path: Any, predictions: Sequence[Prediction]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in predictions_to_rows(predictions):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
