"""Leitura do JSONL do dataset e conversão para o formato prompt/completion do TRL."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adaptlab.training.format import to_prompt_completion


class DatasetFormatError(ValueError):
    """Linha do JSONL inválida (com número da linha)."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(f"{path}:{number}: JSON inválido ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise DatasetFormatError(f"{path}:{number}: a linha não é um objecto")
            records.append(row)
    return records


def spread_sample(records: Sequence[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Amostra determinística e espalhada (passo fixo), para cobrir todos os intents no dry-run."""
    if size >= len(records):
        return list(records)
    step = len(records) / size
    return [records[int(i * step)] for i in range(size)]


def to_prompt_completion_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [to_prompt_completion(record) for record in records]


def stratified_sample(records: Sequence[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Amostra estratificada por ``intent`` (método dos maiores restos) com seed registada.

    Cada intent recebe uma quota proporcional ao seu tamanho; a escolha dentro do intent é
    determinística (ordenação por id + baralhamento com ``Random(seed)``). Devolve a lista
    ordenada por id.
    """
    from random import Random

    if size <= 0:
        raise ValueError("size tem de ser > 0")
    if size >= len(records):
        return sorted(records, key=lambda r: r["id"])
    by_intent: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_intent.setdefault(str(record["intent"]), []).append(record)
    total = len(records)
    quotas = {k: size * len(v) / total for k, v in by_intent.items()}
    take = {k: int(q) for k, q in quotas.items()}
    remaining = size - sum(take.values())
    by_remainder = sorted(quotas, key=lambda k: (-(quotas[k] - take[k]), k))
    for intent in by_remainder[:remaining]:
        take[intent] += 1
    chosen: list[dict[str, Any]] = []
    for intent in sorted(by_intent):
        pool = sorted(by_intent[intent], key=lambda r: r["id"])
        Random(f"stratified:{seed}:{intent}").shuffle(pool)
        chosen.extend(pool[: take[intent]])
    return sorted(chosen, key=lambda r: r["id"])
