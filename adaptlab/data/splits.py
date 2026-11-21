"""Splits train/dev/test por cenário, estratificados por intent.

Um cenário inteiro vai para um único split, por isso nenhum template do teste
aparece no treino (ver ``quality.check_leakage`` para a verificação empírica).
Equivale a um GroupShuffleSplit em que o grupo é o cenário e a amostragem é
feita dentro de cada intent, para que todas as classes existam em todos os splits.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from random import Random

from adaptlab.data.catalog import SCENARIOS, Scenario
from adaptlab.data.generator import Example

SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "test")


def assign_scenarios(
    scenarios: Sequence[Scenario],
    seed: int,
    per_intent: Mapping[str, int],
) -> dict[str, str]:
    """Devolve ``{scenario_id: split}``.

    ``per_intent`` dá o número de cenários por split em cada intent, por exemplo
    ``{"train": 2, "dev": 1, "test": 1}``. A soma tem de ser igual ao número de
    cenários de cada intent.
    """
    by_intent: dict[str, list[str]] = {}
    for scenario in scenarios:
        by_intent.setdefault(scenario.intent, []).append(scenario.scenario_id)

    assignment: dict[str, str] = {}
    for intent in sorted(by_intent):
        ids = sorted(by_intent[intent])
        expected = sum(per_intent.values())
        if len(ids) != expected:
            raise ValueError(f"intent '{intent}' tem {len(ids)} cenários, mas a configuração pede {expected}")
        Random(f"{seed}:{intent}").shuffle(ids)
        cursor = 0
        for split in SPLIT_NAMES:
            for scenario_id in ids[cursor : cursor + per_intent.get(split, 0)]:
                assignment[scenario_id] = split
            cursor += per_intent.get(split, 0)
    return assignment


def split_examples(
    examples: Sequence[Example],
    assignment: Mapping[str, str],
) -> dict[str, list[Example]]:
    splits: dict[str, list[Example]] = {name: [] for name in SPLIT_NAMES}
    for example in examples:
        split = assignment.get(example.scenario_id)
        if split is None:
            raise KeyError(f"cenário sem split: {example.scenario_id}")
        splits[split].append(example)
    return splits


def default_assignment(seed: int, per_intent: Mapping[str, int]) -> dict[str, str]:
    return assign_scenarios(SCENARIOS, seed, per_intent)


def hash_records(records: Sequence[dict]) -> str:
    """SHA-256 de registos (dicts), independente da ordem: ordenado por id, JSON canónico."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda rec: rec["id"]):
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_hash(examples: Sequence[Example]) -> str:
    """SHA-256 de um conjunto de exemplos (mesma canonicalização que ``hash_records``)."""
    return hash_records([example.to_dict() for example in examples])
