"""Gerador determinístico de exemplos sintéticos com labels controlados.

Mesma seed, mesmo dataset: um único ``random.Random`` percorre os cenários
por ordem fixa. O ruído (PT-AO, erros de escrita, minúsculas) só altera o
texto literal do template; os valores das entidades entram sempre tal como
foram sorteados, para que a validação de entidades literais seja exacta.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any

from adaptlab.data.catalog import SCENARIOS, Scenario

ENTITY_KEYS: tuple[str, ...] = ("numero", "valor", "localidade", "data")
TAG_NOISE = "ruido"
TAG_MIX = "mistura_pt_en"
TAG_INCOMPLETE = "incompleto"
TAG_MISSING_ENTITY = "entidade_ausente"
TAG_ESCALATION = "escalonamento"
ALL_TAGS: tuple[str, ...] = (TAG_NOISE, TAG_MIX, TAG_INCOMPLETE, TAG_MISSING_ENTITY, TAG_ESCALATION)

LOCALIDADES: tuple[str, ...] = (
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
DATAS: tuple[str, ...] = (
    "ontem",
    "hoje",
    "anteontem",
    "segunda-feira",
    "terça-feira",
    "sábado",
    "dia 12",
    "dia 3",
    "há três dias",
    "esta manhã",
)
VALORES_BASE: tuple[int, ...] = (200, 300, 500, 1000, 1500, 2000, 2500, 5000, 10000)
SLANG_PAIRS: tuple[tuple[str, str], ...] = (
    ("está", "ta"),
    ("estou", "tou"),
    ("porque", "pq"),
    ("para", "pa"),
    ("você", "vc"),
    ("obrigado", "obg"),
    ("também", "tbm"),
)
OPTIONAL_RE = re.compile(r"\[\[(.*?)\]\]")
PLACEHOLDER_RE = re.compile(r"\{(numero|valor|localidade|data)\}")
OMIT_PROBABILITY = 0.35
NOISE_PROBABILITY = 0.6
TYPO_PROBABILITY = 0.02


@dataclass(frozen=True)
class Example:
    """Um exemplo rotulado. ``label`` segue o schema de ``adaptlab.schema.Label``."""

    example_id: str
    scenario_id: str
    template_id: str
    intent: str
    message: str
    label: dict[str, Any]
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "scenario_id": self.scenario_id,
            "template_id": self.template_id,
            "intent": self.intent,
            "message": self.message,
            "label": self.label,
            "tags": list(self.tags),
        }


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _sample_numero(rng: Random) -> str:
    digits = "".join(str(rng.randint(0, 9)) for _ in range(8))
    shape = rng.choice(("spaced", "compact", "international"))
    number = f"9{digits}"
    if shape == "spaced":
        return f"{number[:3]} {number[3:6]} {number[6:]}"
    if shape == "international":
        return f"+244 {number[:3]} {number[3:6]} {number[6:]}"
    return number


def _sample_valor(rng: Random) -> str:
    amount = rng.choice(VALORES_BASE)
    shape = rng.choice(("kz", "kwanzas", "dot"))
    if shape == "kwanzas" and amount >= 1000:
        return f"{amount // 1000} mil kwanzas"
    if shape == "dot":
        return f"{amount:,}".replace(",", ".") + " Kz"
    if shape == "kwanzas":
        return f"{amount} kwanzas"
    return f"{amount} Kz"


def sample_slots(rng: Random) -> dict[str, str]:
    """Sorteia um valor para cada entidade. A ordem de sorteio é fixa (determinismo)."""
    return {
        "numero": _sample_numero(rng),
        "valor": _sample_valor(rng),
        "localidade": rng.choice(LOCALIDADES),
        "data": rng.choice(DATAS),
    }


def add_noise(text: str, rng: Random) -> tuple[str, bool]:
    """Aplica ruído PT-AO a um fragmento literal. Devolve (texto, houve_alteração)."""
    out = text
    for formal, slang in SLANG_PAIRS:
        if rng.random() < 0.7:
            out = re.sub(rf"\b{formal}\b", slang, out)
    if rng.random() < 0.5:
        out = strip_accents(out)
    if rng.random() < 0.25:
        out = out.lower()
    chars: list[str] = []
    for ch in out:
        if ch.isascii() and ch.isalpha() and rng.random() < TYPO_PROBABILITY:
            continue
        chars.append(ch)
    out = "".join(chars)
    return out, out != text


def render_template(
    template: str, slots: dict[str, str], rng: Random, noisy: bool
) -> tuple[str, dict[str, str | None], set[str]]:
    """Renderiza um template. Devolve (mensagem, entidades, tags locais).

    Segmentos ``[[...]]`` são omitidos com probabilidade ``OMIT_PROBABILITY``.
    A ordem de chamadas ao ``rng`` é fixa para o mesmo template e seed.
    """
    tags: set[str] = set()

    def _optional(match: re.Match[str]) -> str:
        if rng.random() < OMIT_PROBABILITY:
            tags.add(TAG_INCOMPLETE)
            return ""
        return match.group(1)

    expanded = OPTIONAL_RE.sub(_optional, template)
    pieces = PLACEHOLDER_RE.split(expanded)
    used: dict[str, str | None] = {key: None for key in ENTITY_KEYS}
    rendered: list[str] = []
    for index, piece in enumerate(pieces):
        if index % 2 == 1:
            used[piece] = slots[piece]
            rendered.append(slots[piece])
            continue
        if noisy and piece:
            piece, changed = add_noise(piece, rng)
            if changed:
                tags.add(TAG_NOISE)
        rendered.append(piece)
    message = re.sub(r"\s+", " ", "".join(rendered)).strip()
    return message, used, tags


def _realization_rng(seed: int, scenario_index: int, template_index: int) -> Random:
    return Random(f"{seed}:{scenario_index}:{template_index}")


def generate(
    seed: int,
    realizations: int,
    scenarios: Sequence[Scenario] = SCENARIOS,
) -> list[Example]:
    """Gera ``len(templates) * realizations`` exemplos por cenário, de forma determinística."""
    examples: list[Example] = []
    for s_index, scenario in enumerate(scenarios):
        for t_index, template in enumerate(scenario.templates):
            template_id = f"{scenario.scenario_id}/t{t_index}"
            rng = _realization_rng(seed, s_index, t_index)
            for r_index in range(realizations):
                slots = sample_slots(rng)
                noisy = rng.random() < NOISE_PROBABILITY
                message, entities, local_tags = render_template(template, slots, rng, noisy)
                tags = set(local_tags)
                if scenario.mix:
                    tags.add(TAG_MIX)
                if scenario.requires_human:
                    tags.add(TAG_ESCALATION)
                # Entidade ausente: um placeholder do template foi omitido por um segmento
                # opcional, ou a mensagem não traz nenhuma entidade.
                template_keys = set(PLACEHOLDER_RE.findall(template))
                dropped_keys = {key for key in template_keys if entities[key] is None}
                if dropped_keys or all(value is None for value in entities.values()):
                    tags.add(TAG_MISSING_ENTITY)
                label = {
                    "intent": scenario.intent,
                    "priority": scenario.priority,
                    "entities": entities,
                    "requires_human": scenario.requires_human,
                }
                examples.append(
                    Example(
                        example_id=f"{template_id}/r{r_index:02d}",
                        scenario_id=scenario.scenario_id,
                        template_id=template_id,
                        intent=scenario.intent,
                        message=message,
                        label=label,
                        tags=tuple(sorted(tags)),
                    )
                )
    return examples
