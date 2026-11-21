"""Critério de paragem e verificação do conjunto de teste.

Regra (specs/model-evaluation): o LoRA só segue para publicação se a macro-F1 no DEV
superar a melhor baseline por pelo menos ``margin``. Sem o LoRA medido, o estado é
``pendente`` (nunca "aprovado" por omissão). Se o LoRA não ganhar, o estado é ``bloqueado``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from adaptlab.data.splits import hash_records

BASELINES: tuple[str, ...] = ("rules", "zero_shot", "few_shot")


class TamperedTestSetError(RuntimeError):
    """O conjunto de teste não bate com o hash registado no manifesto."""


@dataclass(frozen=True)
class StopDecision:
    status: str  # "aprovado" | "bloqueado" | "pendente"
    reason: str
    margin: float
    lora_macro_f1: float | None
    best_baseline: str | None
    best_baseline_macro_f1: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(dev_macro_f1: Mapping[str, float | None], margin: float) -> StopDecision:
    lora = dev_macro_f1.get("lora")
    baseline_scores = {name: dev_macro_f1[name] for name in BASELINES if dev_macro_f1.get(name) is not None}
    if lora is None:
        return StopDecision("pendente", "LoRA ainda não medido no dev", margin, None, None, None)
    if not baseline_scores:
        return StopDecision("pendente", "nenhuma baseline medida no dev", margin, lora, None, None)
    best_name = max(baseline_scores, key=lambda name: baseline_scores[name])
    best = baseline_scores[best_name]
    gain = lora - best
    if gain >= margin:
        return StopDecision(
            "aprovado",
            f"LoRA {lora:.4f} supera {best_name} {best:.4f} em {gain:.4f} (margem {margin})",
            margin,
            lora,
            best_name,
            best,
        )
    return StopDecision(
        "bloqueado",
        f"LoRA {lora:.4f} não supera {best_name} {best:.4f} pela margem {margin} (diferença {gain:+.4f}). "
        "Não exportar para publicação; registar o resultado negativo.",
        margin,
        lora,
        best_name,
        best,
    )


def verify_test_hash(test_records: Sequence[dict[str, Any]], expected_sha256: str) -> None:
    """Falha se o teste foi alterado depois de registado no manifesto."""
    actual = hash_records(list(test_records))
    if actual != expected_sha256:
        raise TamperedTestSetError(f"hash do teste diferente do manifesto: {actual} != {expected_sha256}")
