"""Testes do pipeline de dados: reprodutibilidade, labels, splits, fugas, revisão e card."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from adaptlab.data.card import render_card, validate_card
from adaptlab.data.catalog import SCENARIOS
from adaptlab.data.generator import Example, generate
from adaptlab.data.pipeline import build_dataset, load_config
from adaptlab.data.quality import (
    check_leakage,
    drop_near_duplicates,
    find_exact_duplicates,
    jaccard,
    shingles,
    validate_entities_literal,
    validate_example,
)
from adaptlab.data.review import apply_decisions, build_sample, read_decisions
from adaptlab.data.splits import assign_scenarios, canonical_hash, split_examples
from adaptlab.schema import INTENTS

PER_INTENT = {"train": 2, "dev": 1, "test": 1}


@pytest.fixture(scope="module")
def raw() -> list[Example]:
    return generate(seed=20261120, realizations=11)


def test_generation_is_reproducible(raw: list[Example]) -> None:
    again = generate(seed=20261120, realizations=11)
    assert [ex.to_dict() for ex in raw] == [ex.to_dict() for ex in again]


def test_different_seed_changes_dataset(raw: list[Example]) -> None:
    other = generate(seed=1, realizations=11)
    assert [ex.message for ex in raw] != [ex.message for ex in other]


def test_every_label_passes_schema_and_literal_entities(raw: list[Example]) -> None:
    failures = {ex.example_id: validate_example(ex) for ex in raw}
    assert {k: v for k, v in failures.items() if v} == {}


def test_entity_not_in_message_is_detected() -> None:
    label = {
        "intent": "recarga",
        "priority": "baixa",
        "entities": {"numero": "923 456 789", "valor": None, "localidade": None, "data": None},
        "requires_human": False,
    }
    errors = validate_entities_literal("recarga falhou no meu telefone", label)
    assert errors == ["entities.numero: '923 456 789' not in message"]


def test_every_intent_and_scenario_is_generated(raw: list[Example]) -> None:
    assert {ex.intent for ex in raw} == set(INTENTS)
    assert {ex.scenario_id for ex in raw} == {s.scenario_id for s in SCENARIOS}


def test_exact_duplicates_are_found() -> None:
    a = Example("a", "s", "s/t0", "recarga", "Boa tarde", {}, ())
    b = replace(a, example_id="b", message="boa  tarde")
    assert find_exact_duplicates([a, b]) == ["b"]


def test_near_duplicates_are_dropped_keeping_first() -> None:
    base = "a net ta fraca no Kilamba desde ontem"
    first = Example("a", "s", "s/t0", "avaria_rede", base, {}, ())
    second = replace(first, example_id="b", message=base + "!")
    third = replace(first, example_id="c", message="pagamento falhou no cartão do cliente")
    kept, dropped = drop_near_duplicates([first, second, third], threshold=0.85)
    assert [ex.example_id for ex in kept] == ["a", "c"]
    assert dropped == ["b"]


def test_split_assignment_is_scenario_grouped_and_stratified(raw: list[Example]) -> None:
    assignment = assign_scenarios(SCENARIOS, seed=20261120, per_intent=PER_INTENT)
    splits = split_examples(raw, assignment)
    train_templates = {ex.template_id for ex in splits["train"]}
    test_templates = {ex.template_id for ex in splits["test"]}
    assert train_templates.isdisjoint(test_templates)
    for name in ("train", "dev", "test"):
        assert {ex.intent for ex in splits[name]} == set(INTENTS)


def test_artificial_leak_is_detected(raw: list[Example]) -> None:
    assignment = assign_scenarios(SCENARIOS, seed=20261120, per_intent=PER_INTENT)
    splits = split_examples(raw, assignment)
    victim = splits["train"][0]
    leaked = replace(
        splits["test"][0],
        message=victim.message,
        template_id=victim.template_id,
    )
    report = check_leakage({"train": splits["train"], "test": [leaked, *splits["test"][1:]]}, 0.6)
    assert report["shared_templates"] == [f"test/train:{victim.template_id}"]
    assert report["near_duplicates"]


def test_near_duplicate_leak_across_splits_is_detected() -> None:
    a = Example("a", "s1", "s1/t0", "recarga", "fiz uma recarga de 500 Kz e o saldo não mudou", {}, ())
    b = replace(a, example_id="b", scenario_id="s2", template_id="s2/t0", message=a.message + " ok")
    report = check_leakage({"train": [a], "test": [b]}, 0.6)
    assert len(report["near_duplicates"]) == 1
    pair = report["near_duplicates"][0]
    assert {pair.left, pair.right} == {"a", "b"}


def test_test_hash_changes_when_content_changes(raw: list[Example]) -> None:
    base = canonical_hash(raw[:20])
    mutated = [replace(raw[0], message=raw[0].message + "x"), *raw[1:20]]
    assert canonical_hash(raw[:20]) == base
    assert canonical_hash(mutated) != base


def test_shingles_and_jaccard_behave() -> None:
    assert jaccard(shingles("abcdef"), shingles("abcdef")) == 1.0
    assert jaccard(shingles("aaaa"), shingles("zzzz")) == 0.0


def test_review_sample_and_decisions(raw: list[Example]) -> None:
    assignment = assign_scenarios(SCENARIOS, seed=20261120, per_intent=PER_INTENT)
    train = split_examples(raw, assignment)["train"]
    sample = build_sample(train, size=6, seed=1)
    ids = [row["id"] for row in sample]
    assert len(ids) == 6 and all(row["decision"] is None for row in sample)

    pending = apply_decisions(train, ids, {})
    assert len(pending.pending) == 6 and pending.status == "pendente"

    second = next(ex for ex in train if ex.example_id == ids[1])
    decisions = {
        ids[0]: {"id": ids[0], "decision": "rejeitado"},
        ids[1]: {
            "id": ids[1],
            "decision": "corrigido",
            "message_corrigida": second.message.upper(),
        },
    }
    result = apply_decisions(train, ids, decisions)
    assert ids[0] in result.rejected
    assert len(result.examples) == len(train) - 1
    assert result.status == "pendente"


def test_review_correction_that_breaks_entities_is_an_error(raw: list[Example]) -> None:
    assignment = assign_scenarios(SCENARIOS, seed=20261120, per_intent=PER_INTENT)
    train = split_examples(raw, assignment)["train"]
    target = next(ex for ex in train if ex.label["entities"]["valor"] is not None)
    decisions = {
        target.example_id: {
            "id": target.example_id,
            "decision": "corrigido",
            "message_corrigida": "mensagem sem o valor",
        }
    }
    result = apply_decisions(train, [target.example_id], decisions)
    assert result.errors and result.status == "erro"


def test_read_decisions_rejects_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        '{"id": "a", "decision": "ok"}\n'
        "não é json\n"
        '{"id": "b", "decision": "corrigido"}\n'
        '{"id": "c", "decision": "aceite"}\n',
        encoding="utf-8",
    )
    decisions, errors = read_decisions(path)
    assert set(decisions) == {"c"}
    assert len(errors) == 3


def test_full_pipeline_meets_requirements() -> None:
    config = load_config()
    config["review_decisions"] = str(Path("/nonexistent-decisions.jsonl").resolve())
    result = build_dataset(config)
    total = sum(len(v) for v in result.splits.values())
    assert total >= 500
    assert result.manifest["leakage"]["shared_templates"] == 0
    assert result.manifest["review"]["status"] == "pendente"
    assert result.manifest["test_sha256"] == canonical_hash(result.splits["test"])


def test_card_has_all_sections_and_detects_gaps() -> None:
    config = load_config()
    config["review_decisions"] = "/nonexistent-decisions.jsonl"
    manifest = build_dataset(config).manifest
    card = render_card(manifest, config)
    assert validate_card(card) == []
    assert validate_card(card.replace("## Dados privados", "## Outros")) == ["secção em falta: ## Dados privados"]


def test_missing_entity_tag_is_not_universal(raw: list[Example]) -> None:
    tagged = [ex for ex in raw if "entidade_ausente" in ex.tags]
    assert 0 < len(tagged) < len(raw)
    for ex in tagged:
        assert ex.label["entities"] is not None
