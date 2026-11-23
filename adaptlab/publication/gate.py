"""Portão de publicação: todas as condições têm de passar, senão a publicação é bloqueada.

Condições (specs/model-publication):
1. Licença do modelo base conhecida e que permite redistribuir derivados.
2. Licença do dataset definida.
3. Critério de paragem aprovado no dev (``aprovado``).
4. Confirmação manual explícita.

Este módulo não publica nada. A publicação no Hugging Face, se for feita, é um passo manual.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Licenças que permitem redistribuir derivados com aviso (verificadas na model card antes de usar).
REDISTRIBUTABLE_LICENSES: frozenset[str] = frozenset({"apache-2.0", "mit"})


@dataclass(frozen=True)
class PublicationCheck:
    allowed: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def check_publication(
    base_license: str | None,
    dataset_license: str | None,
    stop_status: str,
    manual_confirmation: bool,
) -> PublicationCheck:
    reasons: list[str] = []
    normalized = (base_license or "").strip().lower()
    if not normalized:
        reasons.append("licença do modelo base desconhecida")
    elif normalized not in REDISTRIBUTABLE_LICENSES:
        reasons.append(
            f"licença do modelo base '{base_license}' não permite redistribuir derivados "
            f"(permitidas: {', '.join(sorted(REDISTRIBUTABLE_LICENSES))})"
        )
    if not (dataset_license or "").strip():
        reasons.append("licença do dataset não definida")
    if stop_status != "aprovado":
        reasons.append(f"critério de paragem não aprovado (estado: {stop_status})")
    if not manual_confirmation:
        reasons.append("confirmação manual em falta (--confirm)")
    return PublicationCheck(allowed=not reasons, reasons=reasons)
