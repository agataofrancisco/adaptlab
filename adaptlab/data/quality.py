"""Validação de labels, deduplicação e detecção de fugas entre splits.

Quase-duplicados: similaridade de Jaccard sobre conjuntos de n-gramas de
caracteres (n=5) do texto normalizado. É determinístico, sem dependências
pesadas, e O(n^2), o que é aceitável para o tamanho deste dataset (<1000).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from adaptlab.data.generator import ENTITY_KEYS, Example, strip_accents
from adaptlab.schema import Label

SHINGLE_SIZE = 5
_WHITESPACE_RE = re.compile(r"\s+")
_ENTITY_NOISE_RE = re.compile(r"[\s.,\-]")


def normalize_text(text: str) -> str:
    """Minúsculas, sem acentos, espaços colapsados. Usado para comparar textos."""
    return _WHITESPACE_RE.sub(" ", strip_accents(text).casefold()).strip()


def normalize_entity(value: str) -> str:
    """Normalização para comparar entidades: sem espaços nem pontuação de milhares."""
    return _ENTITY_NOISE_RE.sub("", normalize_text(value))


def shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[str]:
    normalized = normalize_text(text)
    if len(normalized) <= size:
        return frozenset({normalized})
    return frozenset(normalized[i : i + size] for i in range(len(normalized) - size + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def validate_label_schema(label: object) -> list[str]:
    """Valida o label contra o schema de saída. Devolve a lista de erros (vazia se ok)."""
    from pydantic import ValidationError

    if not isinstance(label, dict):
        return ["label_not_object"]
    try:
        Label.model_validate(label)
    except ValidationError as exc:
        return [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()]
    return []


def validate_entities_literal(message: str, label: dict) -> list[str]:
    """Cada entidade não nula tem de aparecer literalmente na mensagem."""
    errors: list[str] = []
    entities = label.get("entities") if isinstance(label, dict) else None
    if not isinstance(entities, dict):
        return ["entities_missing"]
    haystack = normalize_entity(message)
    for key in ENTITY_KEYS:
        value = entities.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            errors.append(f"entities.{key}: not_a_string")
            continue
        if normalize_entity(value) not in haystack:
            errors.append(f"entities.{key}: '{value}' not in message")
    return errors


def validate_example(example: Example) -> list[str]:
    errors = validate_label_schema(example.label)
    if not errors:
        errors.extend(validate_entities_literal(example.message, example.label))
    return errors


@dataclass(frozen=True)
class DuplicatePair:
    left: str
    right: str
    score: float


def find_exact_duplicates(examples: Sequence[Example]) -> list[str]:
    """IDs de exemplos cujo texto normalizado já apareceu antes (o primeiro fica)."""
    seen: set[str] = set()
    dropped: list[str] = []
    for example in examples:
        key = normalize_text(example.message)
        if key in seen:
            dropped.append(example.example_id)
        else:
            seen.add(key)
    return dropped


def find_near_pairs(
    left: Iterable[Example],
    right: Iterable[Example],
    threshold: float,
) -> list[DuplicatePair]:
    """Pares (esquerda, direita) com Jaccard de 5-gramas >= ``threshold``."""
    right_shingles = [(ex.example_id, shingles(ex.message)) for ex in right]
    pairs: list[DuplicatePair] = []
    for ex in left:
        left_sh = shingles(ex.message)
        for rid, rsh in right_shingles:
            score = jaccard(left_sh, rsh)
            if score >= threshold:
                pairs.append(DuplicatePair(ex.example_id, rid, round(score, 4)))
    return pairs


def drop_near_duplicates(examples: Sequence[Example], threshold: float) -> tuple[list[Example], list[str]]:
    """Remove o segundo elemento de cada par quase-duplicado (ordem estável)."""
    kept: list[Example] = []
    kept_shingles: list[frozenset[str]] = []
    dropped: list[str] = []
    for example in examples:
        sh = shingles(example.message)
        if any(jaccard(sh, other) >= threshold for other in kept_shingles):
            dropped.append(example.example_id)
            continue
        kept.append(example)
        kept_shingles.append(sh)
    return kept, dropped


def check_leakage(
    splits: dict[str, Sequence[Example]],
    threshold: float,
) -> dict[str, list[DuplicatePair] | list[str]]:
    """Fugas entre splits: templates partilhados e quase-duplicados entre splits diferentes."""
    report: dict[str, list[DuplicatePair] | list[str]] = {
        "shared_templates": [],
        "near_duplicates": [],
    }
    names = sorted(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            a_templates = {ex.template_id for ex in splits[a]}
            b_templates = {ex.template_id for ex in splits[b]}
            for tid in sorted(a_templates & b_templates):
                report["shared_templates"].append(f"{a}/{b}:{tid}")
            report["near_duplicates"].extend(  # type: ignore[union-attr]
                find_near_pairs(splits[a], splits[b], threshold)
            )
    return report
