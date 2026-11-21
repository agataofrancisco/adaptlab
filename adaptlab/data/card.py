"""Renderiza o dataset card a partir do manifesto (números sempre medidos na run).

O texto narrativo é fixo; as tabelas e contagens vêm do manifesto. ``validate_card``
verifica que todas as secções obrigatórias existem e não estão vazias.
"""

from __future__ import annotations

from typing import Any

REQUIRED_SECTIONS: tuple[str, ...] = (
    "## Origem",
    "## Licença",
    "## Classes e distribuições",
    "## Método de geração",
    "## Splits e prevenção de fugas",
    "## Revisão humana",
    "## Limitações",
    "## Riscos de enviesamento",
    "## Dados privados",
    "## Reprodução",
)

LICENSE_NOTE = (
    "CC-BY-4.0 (escolha do autor do projecto; **proposta a confirmar**). "
    "Os dados são sintéticos, criados para este projecto, e não contêm dados pessoais reais."
)


def _table(rows: list[tuple[str, Any]], header: tuple[str, str]) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "|---|---|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(lines)


def _split_table(splits: dict[str, Any]) -> str:
    lines = [
        "| Split | Exemplos | requires_human=true | Mensagem média (caracteres) |",
        "|---|---|---|---|",
    ]
    for name in ("train", "dev", "test"):
        info = splits[name]
        lines.append(f"| {name} | {info['n']} | {info['requires_human_true']} | {info['message_chars_mean']} |")
    return "\n".join(lines)


def _intent_table(splits: dict[str, Any]) -> str:
    intents = sorted({key for info in splits.values() for key in info["intents"]})
    header = "| Intent | train | dev | test |"
    lines = [header, "|---|---|---|---|"]
    for intent in intents:
        counts = [splits[name]["intents"].get(intent, 0) for name in ("train", "dev", "test")]
        lines.append(f"| {intent} | {counts[0]} | {counts[1]} | {counts[2]} |")
    return "\n".join(lines)


def _tag_table(splits: dict[str, Any]) -> str:
    totals: dict[str, int] = {}
    for info in splits.values():
        for tag, count in info["tags"].items():
            totals[tag] = totals.get(tag, 0) + count
    return _table(sorted(totals.items()), ("Dificuldade (tag)", "Exemplos"))


def render_card(manifest: dict[str, Any], config: dict[str, Any]) -> str:
    splits = manifest["splits"]
    review = manifest["review"]
    gen = manifest["generator"]
    leak = manifest["leakage"]
    total = sum(info["n"] for info in splits.values())
    return f"""# Dataset card: AdaptLab (Mukua Telecom, sintético)

Gerado por `scripts/prepare_data.py` com a configuração `configs/data.yaml`.
Seed: `{manifest["seed"]}`. Hash SHA-256 do teste: `{manifest["test_sha256"]}`.

## Origem

Dados sintéticos gerados por templates e cenários escritos para o AdaptLab. A empresa
Mukua Telecom é fictícia. Nenhuma mensagem foi copiada de clientes reais.

## Licença

{LICENSE_NOTE}

## Classes e distribuições

Total: **{total}** exemplos após deduplicação. Intents: `recarga`, `avaria_rede`,
`faturacao`, `portabilidade`, `cancelamento`, `pacote_dados`, `outro`.
Prioridades: `baixa`, `media`, `alta`.

{_split_table(splits)}

Distribuição de intents por split:

{_intent_table(splits)}

Dificuldades (um exemplo pode ter várias tags):

{_tag_table(splits)}

## Método de geração

1. Catálogo de 28 cenários (4 por intent), cada um com 2 a 3 templates (`adaptlab/data/catalog.py`).
2. Cada template é realizado {gen["realizations_per_template"]} vezes com valores sorteados (número,
   valor em kwanzas, localidade, data). Ruído PT-AO, erros de escrita e minúsculas só afectam o texto
   literal, nunca os valores das entidades.
3. Segmentos opcionais omitem entidades (`incompleto`, `entidade_ausente`).
4. Labels validados contra o schema Pydantic (`adaptlab/schema.py`). Cada entidade não nula tem de
   aparecer literalmente na mensagem (normalização de espaços e pontuação).
5. Deduplicação: {gen["exact_duplicates_dropped"]} duplicados exactos e {gen["near_duplicates_dropped"]}
   quase-duplicados removidos (Jaccard de 5-gramas >= {config["dedupe_threshold"]}).
   Gerados antes da deduplicação: {gen["raw_count"]}.

## Splits e prevenção de fugas

Os splits são por cenário, estratificados por intent (`{config["scenarios_per_intent"]}`).
Nenhum template aparece em dois splits (`shared_templates = 0`). Quase-duplicados entre
splits são verificados com limiar {leak["threshold"]}: pares encontrados = {leak["near_duplicate_pairs_across_splits"]}.
O conjunto de teste tem hash registado (`manifest.json`, campo `test_sha256`) e só deve ser
usado na avaliação final.

## Revisão humana

Estado: **{review["status"].upper()}**. Amostra de {review["sample_size"]} exemplos do treino
(`data/review/sample.jsonl`). Decisões registadas: {review["decisions_recorded"]}; pendentes:
{review["pending"]}; rejeitados: {review["rejected"]}; corrigidos: {review["corrected"]}.
A revisão do léxico PT-AO por uma pessoa ainda não foi feita. Enquanto estiver pendente, o
naturalismo do português de Angola **não está validado**.

## Limitações

- Dados sintéticos: o modelo pode aprender os templates e não a linguagem real dos clientes.
- Poucos cenários por intent (4) e um teste de tamanho limitado.
- Português de Angola representado por léxico escolhido pelo autor, ainda sem revisão externa.
- Sem mensagens de voz transcritas, sem emojis e sem erros de digitação reais de telemóvel.

## Riscos de enviesamento

- Localidades, nomes de bairros e padrões de número são escolhidos a dedo e não representam a
  distribuição real de clientes.
- Mensagens em inglês são uma fracção artificial, com mistura controlada.
- Escalonamento (`requires_human=true`) é definido pelos autores, não por política da operadora.

## Dados privados

Não foram incluídos dados pessoais, números reais de clientes, facturas reais nem registos
de atendimento. Os números de telefone são gerados aleatoriamente.

## Reprodução

```
.venv\\Scripts\\python scripts\\prepare_data.py --check
.venv\\Scripts\\python scripts\\prepare_data.py
.venv\\Scripts\\python -m pytest tests -q
```
"""


def validate_card(text: str) -> list[str]:
    """Devolve as secções obrigatórias em falta ou vazias (lista vazia = card completo)."""
    problems: list[str] = []
    for section in REQUIRED_SECTIONS:
        index = text.find(section)
        if index == -1:
            problems.append(f"secção em falta: {section}")
            continue
        body_start = index + len(section)
        next_index = text.find("\n## ", body_start)
        body = text[body_start : next_index if next_index != -1 else len(text)].strip()
        if not body:
            problems.append(f"secção vazia: {section}")
    return problems
