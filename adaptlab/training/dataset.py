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
