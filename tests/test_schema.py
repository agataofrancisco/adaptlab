"""Parsing defensivo da saída do modelo: nunca lança, nunca inventa campos."""

from __future__ import annotations

import pytest

from adaptlab.schema import Label, extract_first_json_object, label_to_json, parse_label

VALID = (
    '{"intent": "avaria_rede", "priority": "alta", '
    '"entities": {"numero": null, "valor": null, "localidade": "Kilamba", "data": "desde ontem"}, '
    '"requires_human": false}'
)


def test_parse_valid_label() -> None:
    label, error = parse_label(VALID)
    assert error is None
    assert label is not None
    assert label.intent == "avaria_rede"
    assert label.entities.localidade == "Kilamba"


def test_parse_accepts_prose_around_json() -> None:
    label, error = parse_label(f"Claro, aqui está:\n{VALID}\nEspero que ajude.")
    assert error is None and label is not None


@pytest.mark.parametrize("bad", [None, 42, b"{}", ["{}"], "", "sem json nenhum"])
def test_parse_never_raises_on_non_text(bad: object) -> None:
    label, error = parse_label(bad)
    assert label is None
    assert error == "no_json_object"


def test_parse_rejects_unknown_intent() -> None:
    text = VALID.replace("avaria_rede", "inventado")
    label, error = parse_label(text)
    assert label is None
    assert error is not None and error.startswith("schema_error")


def test_parse_rejects_extra_fields() -> None:
    text = VALID.replace('"requires_human": false', '"requires_human": false, "confianca": 0.9')
    label, error = parse_label(text)
    assert label is None and error is not None


def test_parse_rejects_truncated_json() -> None:
    label, error = parse_label(VALID[:40])
    assert label is None and error == "no_json_object"


def test_extract_skips_non_dict_objects_and_broken_prefixes() -> None:
    assert extract_first_json_object('{ x } [1,2] {"a": 1}') == {"a": 1}


def test_label_to_json_is_canonical() -> None:
    label = Label.model_validate_json(VALID)
    assert label_to_json(label) == label_to_json(Label.model_validate_json(label_to_json(label)))
    assert label_to_json(label).startswith('{"intent": "avaria_rede", "priority": "alta"')
