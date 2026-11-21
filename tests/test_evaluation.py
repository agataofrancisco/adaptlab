"""Testes da avaliação: métricas com exemplos feitos à mão, regras, critério de paragem, teste intocado."""

from __future__ import annotations

import json

import pytest

from adaptlab.evaluation.approaches import run_model, select_few_shot
from adaptlab.evaluation.metrics import INVALID, Prediction, compute_metrics, entity_set
from adaptlab.evaluation.report import render_report
from adaptlab.evaluation.rules import rules_label
from adaptlab.evaluation.stopping import TamperedTestSetError, decide, verify_test_hash
from adaptlab.schema import Label, label_to_json


def _label(intent: str, priority: str = "baixa", requires_human: bool = False, **entities: str | None) -> dict:
    base = {"numero": None, "valor": None, "localidade": None, "data": None}
    base.update(entities)
    return {"intent": intent, "priority": priority, "entities": base, "requires_human": requires_human}


def _gold(example_id: str, label: dict) -> dict:
    return {"id": example_id, "intent": label["intent"], "label": label}


def _pred(example_id: str, label: dict | None, latency: float = 0.1, tokens: int = 10) -> Prediction:
    return Prediction(example_id, label, None if label else "no_json_object", latency, tokens)


# --- métricas -----------------------------------------------------------------


def test_perfect_predictions_score_one() -> None:
    gold = [
        _gold("a", _label("recarga", valor="500 Kz")),
        _gold("b", _label("avaria_rede", "alta", True, localidade="Kilamba")),
    ]
    preds = [_pred(g["id"], g["label"]) for g in gold]
    m = compute_metrics(gold, preds)
    assert m["macro_f1_intent"] == 1.0
    assert m["entity_f1"] == 1.0
    assert m["json_valid_rate"] == 1.0
    assert m["high_risk_recall"] == 1.0


def test_invalid_output_counts_as_error_everywhere() -> None:
    gold = [
        _gold("a", _label("recarga", valor="500 Kz")),
        _gold("b", _label("cancelamento", "alta", True, numero="923 456 789")),
    ]
    preds = [_pred("a", None), _pred("b", None)]
    m = compute_metrics(gold, preds)
    assert m["json_valid_rate"] == 0.0
    assert m["macro_f1_intent"] == 0.0
    assert m["entity_f1"] == 0.0
    assert m["high_risk_recall"] == 0.0
    assert m["confusion_labels"][-1] == INVALID
    assert m["errors_by_gold_intent"] == {"cancelamento": 1, "recarga": 1}


def test_hand_computed_entity_f1() -> None:
    # gold: {valor 500 Kz, localidade Kilamba} ; pred: {valor 500 Kz, localidade Viana}
    # TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    gold = [_gold("a", _label("recarga", valor="500 Kz", localidade="Kilamba"))]
    preds = [_pred("a", _label("recarga", valor="500 Kz", localidade="Viana"))]
    m = compute_metrics(gold, preds)
    assert m["entity_precision"] == 0.5
    assert m["entity_recall"] == 0.5
    assert m["entity_f1"] == 0.5


def test_entity_comparison_ignores_spacing_and_thousand_dots() -> None:
    assert entity_set(_label("recarga", valor="2.000 Kz")) == entity_set(_label("recarga", valor="2000kz"))


def test_high_risk_recall_hand_example() -> None:
    # 2 positivos (um escalado, um prioridade alta); o modelo apanha só um.
    gold = [
        _gold("a", _label("avaria_rede", "alta", True)),
        _gold("b", _label("pacote_dados", "alta", False)),
        _gold("c", _label("recarga", "baixa", False)),
    ]
    preds = [
        _pred("a", _label("avaria_rede", "alta", True)),
        _pred("b", _label("pacote_dados", "baixa", False)),
        _pred("c", _label("recarga", "baixa", False)),
    ]
    m = compute_metrics(gold, preds)
    assert m["high_risk_positives"] == 2
    assert m["high_risk_recall"] == 0.5


def test_misaligned_ids_are_rejected() -> None:
    gold = [_gold("a", _label("recarga"))]
    with pytest.raises(ValueError, match="alinhados"):
        compute_metrics(gold, [_pred("zz", _label("recarga"))])


# --- regras -------------------------------------------------------------------


def test_rules_on_spec_example() -> None:
    label = rules_label("boa tarde mano, a net ta fraca no kilamba desde ontem")
    assert label["intent"] == "avaria_rede"
    assert label["priority"] == "alta"
    assert label["entities"]["localidade"] == "Kilamba"
    assert label["entities"]["data"] == "ontem"
    assert label["requires_human"] is False


def test_rules_escalation_and_numbers() -> None:
    label = rules_label("quero falar com o supervisor, o número é 923 456 789 e paguei 2.000 Kz")
    assert label["requires_human"] is True
    assert label["priority"] == "alta"
    assert label["entities"]["numero"] == "923 456 789"
    assert label["entities"]["valor"] == "2.000 Kz"


@pytest.mark.parametrize("message", ["", None, 42, "zzz qqq", "bom dia"])
def test_rules_always_return_schema_valid_label(message: object) -> None:
    label = rules_label(message)  # type: ignore[arg-type]
    Label.model_validate(label)


def test_rules_are_deterministic() -> None:
    message = "cancelem a linha 923 456 789 agora"
    assert rules_label(message) == rules_label(message)


# --- abordagens com gerador falso ---------------------------------------------


def test_run_model_parses_valid_and_invalid_outputs() -> None:
    good = label_to_json(Label.model_validate(_label("recarga")))
    outputs = iter([good, "não sei"])

    def fake(messages: list[dict[str, str]]) -> tuple[str, int]:
        assert messages[0]["role"] == "system"
        return next(outputs), 7

    records = [{"id": "a", "message": "x"}, {"id": "b", "message": "y"}]
    preds = run_model(records, fake)
    assert preds[0].label is not None and preds[0].label["intent"] == "recarga"
    assert preds[1].label is None and preds[1].error == "no_json_object"
    assert preds[0].output_tokens == 7


def test_few_shot_selection_is_deterministic_with_distinct_intents() -> None:
    train = [
        {"id": f"{i}", "intent": intent, "message": "m", "label": _label(intent)}
        for i, intent in enumerate(["recarga", "avaria_rede", "faturacao", "outro", "recarga", "cancelamento"])
    ]
    first = select_few_shot(train, k=4, seed=42)
    assert first == select_few_shot(train, k=4, seed=42)
    assert len({r["intent"] for r in first}) == len(first) == 4


# --- critério de paragem ------------------------------------------------------


def test_stop_approved_when_lora_beats_best_baseline_by_margin() -> None:
    d = decide({"rules": 0.5, "zero_shot": 0.6, "few_shot": 0.7, "lora": 0.75}, margin=0.02)
    assert d.status == "aprovado"
    assert d.best_baseline == "few_shot"


def test_stop_blocks_when_lora_does_not_improve() -> None:
    d = decide({"rules": 0.5, "zero_shot": 0.6, "few_shot": 0.8, "lora": 0.79}, margin=0.02)
    assert d.status == "bloqueado"
    assert "Não exportar" in d.reason


def test_stop_is_pending_without_lora_or_baselines() -> None:
    assert decide({"rules": 0.5, "lora": None}, margin=0.02).status == "pendente"
    assert decide({"rules": None, "zero_shot": None, "few_shot": None, "lora": 0.9}, margin=0.02).status == "pendente"


def test_stop_margin_boundary_is_inclusive() -> None:
    assert decide({"few_shot": 0.5, "lora": 0.52}, margin=0.02).status == "aprovado"


# --- teste intocado -----------------------------------------------------------


def test_tampered_test_set_is_refused() -> None:
    records = [{"id": "a", "message": "x", "intent": "recarga", "label": {}}]
    from adaptlab.data.splits import hash_records

    good = hash_records(records)
    verify_test_hash(records, good)
    changed = [{**records[0], "message": "y"}]
    with pytest.raises(TamperedTestSetError):
        verify_test_hash(changed, good)


# --- relatório ----------------------------------------------------------------


def test_report_without_runs_shows_only_todo_values(tmp_path) -> None:
    text = render_report([], margin=0.02, dataset_hash="abc")
    assert "PENDENTE" in text.upper() or "PENDENTE" in text
    assert "TODO: medir" in text
    assert "abc" in text


def test_report_uses_measured_values_with_run_reference() -> None:
    run = {
        "mode": "evaluate",
        "approach": "rules",
        "split": "dev",
        "_run": "eval-rules-dev-x",
        "seeds": 1,
        "bootstrap_ci": {"macro_f1_intent": [0.1, 0.2]},
        "metrics": {
            "macro_f1_intent": 0.5,
            "entity_f1": 0.4,
            "json_valid_rate": 1.0,
            "high_risk_recall": 0.3,
            "latency_mean_s": 0.001,
            "output_tokens_mean": 0.0,
            "confusion_labels": ["recarga", INVALID],
            "confusion": [[1, 0], [0, 0]],
        },
    }
    text = render_report([run], margin=0.02, dataset_hash=None)
    assert "0.5000" in text
    assert "[0.1000, 0.2000]" in text


def test_metrics_payload_is_json_serialisable() -> None:
    gold = [_gold("a", _label("recarga"))]
    m = compute_metrics(gold, [_pred("a", _label("recarga"))])
    json.dumps(m)
