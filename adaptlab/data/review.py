"""Revisão humana de uma amostra do léxico PT-AO.

Fluxo:
1. ``build_sample`` escolhe N exemplos do split de treino (seed fixa) e escreve
   ``data/review/sample.jsonl`` com ``decision: null``. Só o treino é revisto,
   para o teste ficar intocado.
2. A pessoa revisora preenche ``decision`` (``aceite``, ``corrigido``,
   ``rejeitado``) e, se for ``corrigido``, ``message_corrigida``, e guarda o
   ficheiro como ``data/review/decisions.jsonl``.
3. ``apply_decisions`` aplica as decisões ao treino: rejeitados saem, corrigidos
   trocam o texto (com validação das entidades literais). Pendentes são reportados.

A revisão não é feita por este código: o estado ``pendente`` fica registado no
manifesto e no dataset card.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from adaptlab.data.generator import Example
from adaptlab.data.quality import validate_entities_literal

DECISIONS = ("aceite", "corrigido", "rejeitado")


@dataclass
class ReviewResult:
    examples: list[Example]
    pending: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    corrected: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors:
            return "erro"
        return "pendente" if self.pending else "concluida"


def build_sample(train: Sequence[Example], size: int, seed: int) -> list[dict[str, Any]]:
    """Amostra aleatória (seed fixa) do treino, pronta para preencher."""
    ordered = sorted(train, key=lambda ex: ex.example_id)
    chosen = Random(f"review:{seed}").sample(ordered, k=min(size, len(ordered)))
    return [
        {
            "id": ex.example_id,
            "message": ex.message,
            "intent": ex.intent,
            "decision": None,
            "message_corrigida": None,
            "nota": None,
        }
        for ex in sorted(chosen, key=lambda ex: ex.example_id)
    ]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_decisions(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Lê ``decisions.jsonl``. Devolve ``({id: linha}, erros)``; nunca lança por conteúdo."""
    decisions: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if not path.exists():
        return decisions, errors
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            errors.append(f"linha {number}: JSON inválido")
            continue
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            errors.append(f"linha {number}: falta 'id'")
            continue
        decision = row.get("decision")
        if decision not in DECISIONS:
            errors.append(f"{row['id']}: decisão inválida ({decision!r})")
            continue
        if decision == "corrigido":
            corrected = row.get("message_corrigida")
            if not isinstance(corrected, str) or not corrected.strip():
                errors.append(f"{row['id']}: 'corrigido' exige message_corrigida")
                continue
        decisions[row["id"]] = row
    return decisions, errors


def apply_decisions(
    train: Sequence[Example],
    sample_ids: Sequence[str],
    decisions: dict[str, dict[str, Any]],
) -> ReviewResult:
    result = ReviewResult(examples=[])
    for ex in train:
        if ex.example_id not in sample_ids:
            result.examples.append(ex)
            continue
        row = decisions.get(ex.example_id)
        if row is None:
            result.pending.append(ex.example_id)
            result.examples.append(ex)
            continue
        decision = row["decision"]
        if decision == "rejeitado":
            result.rejected.append(ex.example_id)
            continue
        if decision == "corrigido":
            new_message = row["message_corrigida"]
            errs = validate_entities_literal(new_message, ex.label)
            if errs:
                result.errors.append(f"{ex.example_id}: correcção quebra entidades {errs}")
                result.examples.append(ex)
                continue
            result.corrected.append(ex.example_id)
            result.examples.append(
                Example(
                    example_id=ex.example_id,
                    scenario_id=ex.scenario_id,
                    template_id=ex.template_id,
                    intent=ex.intent,
                    message=new_message,
                    label=ex.label,
                    tags=ex.tags,
                )
            )
            continue
        result.examples.append(ex)
    return result
