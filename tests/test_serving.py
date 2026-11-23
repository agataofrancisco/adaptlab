"""Testes da API (modo demo e modo real com gerador falso), do portão de publicação e da exportação."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adaptlab.evaluation.stopping import StopDecision
from adaptlab.publication.gate import check_publication
from adaptlab.publication.model_card import render_model_card
from adaptlab.schema import Label, label_to_json
from adaptlab.serving.predictor import DemoPredictor, RealPredictor
from api.main import create_app

VALID_LABEL = label_to_json(
    Label.model_validate(
        {
            "intent": "avaria_rede",
            "priority": "alta",
            "entities": {"numero": None, "valor": None, "localidade": "Kilamba", "data": "ontem"},
            "requires_human": False,
        }
    )
)


def _fixtures(path: Path) -> Path:
    row = {
        "message": "a net ta fraca no kilamba desde ontem",
        "model": "regras_fixture",
        "label": json.loads(VALID_LABEL),
        "source": "runs/eval-rules-dev-TEST/predictions.jsonl",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def demo_client(tmp_path: Path) -> TestClient:
    predictor = DemoPredictor(_fixtures(tmp_path / "pre.jsonl"))
    return TestClient(create_app(predictor, model_card_path=tmp_path / "card.md"))


def test_health_reports_demo_on_cpu(demo_client: TestClient) -> None:
    body = demo_client.get("/health").json()
    assert body["mode"] == "precomputed_demo"
    assert body["device"] == "cpu"
    assert body["model_loaded"] is False


def test_demo_fixture_message_is_marked_precomputed(demo_client: TestClient) -> None:
    response = demo_client.post("/predict", json={"message": "A net ta fraca no Kilamba desde ontem"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "precomputed_demo"
    assert body["label"]["intent"] == "avaria_rede"
    assert "Não é uma inferência" in body["note"]


def test_demo_new_message_returns_503_with_explanation(demo_client: TestClient) -> None:
    response = demo_client.post("/predict", json={"message": "quero mudar de operadora"})
    assert response.status_code == 503
    body = response.json()
    assert body["mode"] == "precomputed_demo"
    assert body["error"] == "demo_message_not_precomputed"
    assert "ADAPTLAB_MODE=real" in body["explanation"]


@pytest.mark.parametrize("message", ["", "   ", "x" * 2001])
def test_input_validation_rejects_blank_and_huge_messages(demo_client: TestClient, message: str) -> None:
    assert demo_client.post("/predict", json={"message": message}).status_code == 422


def test_model_card_404_until_generated(demo_client: TestClient) -> None:
    assert demo_client.get("/model-card").status_code == 404


def test_model_card_served_when_present(tmp_path: Path) -> None:
    card = tmp_path / "card.md"
    card.write_text("# Card\n", encoding="utf-8")
    client = TestClient(create_app(DemoPredictor(tmp_path / "none.jsonl"), model_card_path=card))
    assert client.get("/model-card").json()["content"] == "# Card\n"


def test_real_mode_invalid_output_is_structured_error(tmp_path: Path) -> None:
    def garbage(messages):  # noqa: ANN001
        return "claro! aqui vai: {intent: recarga", 9

    predictor = RealPredictor({"adapted": garbage}, device="cpu")
    client = TestClient(create_app(predictor, model_card_path=tmp_path / "c.md"))
    response = client.post("/predict?model=adapted", json={"message": "recarga falhou"})
    assert response.status_code == 200
    body = response.json()
    assert body["error"] == "invalid_model_output"
    assert body["requires_human"] is True
    assert body["label"] is None
    metrics = client.get("/metrics").json()
    assert metrics["invalid_model_output_total"] == 1


def test_real_mode_valid_output_is_returned_and_validated(tmp_path: Path) -> None:
    predictor = RealPredictor({"base": lambda m: (VALID_LABEL, 40), "adapted": lambda m: (VALID_LABEL, 40)}, "cpu")
    client = TestClient(create_app(predictor, model_card_path=tmp_path / "c.md"))
    body = client.post("/predict?model=base", json={"message": "a net ta fraca"}).json()
    assert body["error"] is None
    assert body["label"]["entities"]["localidade"] == "Kilamba"
    assert body["mode"] == "real"


def test_real_mode_unknown_model_is_400(tmp_path: Path) -> None:
    predictor = RealPredictor({"adapted": lambda m: (VALID_LABEL, 1)}, "cpu")
    client = TestClient(create_app(predictor, model_card_path=tmp_path / "c.md"))
    assert client.post("/predict?model=base", json={"message": "x"}).status_code == 400


def test_metrics_endpoint_counts_requests(demo_client: TestClient) -> None:
    demo_client.post("/predict", json={"message": "a net ta fraca no kilamba desde ontem"})
    demo_client.post("/predict", json={"message": "mensagem nova sem fixture"})
    body = demo_client.get("/metrics").json()
    assert body["requests_total"] == 2
    assert body["requests_by_status"] == {"200": 1, "503": 1}


# --- publicação ---------------------------------------------------------------


def test_gate_refuses_restricted_base_license() -> None:
    check = check_publication("llama3.2", "CC-BY-4.0", "aprovado", manual_confirmation=True)
    assert check.allowed is False
    assert any("não permite redistribuir" in reason for reason in check.reasons)


def test_gate_refuses_when_stop_blocked_or_unconfirmed() -> None:
    check = check_publication("apache-2.0", "CC-BY-4.0", "bloqueado", manual_confirmation=False)
    assert check.allowed is False
    assert len(check.reasons) == 2


def test_gate_opens_only_when_all_conditions_hold() -> None:
    assert check_publication("apache-2.0", "CC-BY-4.0", "aprovado", manual_confirmation=True).allowed


def test_model_card_has_no_invented_numbers() -> None:
    decision = StopDecision("pendente", "LoRA ainda não medido no dev", 0.02, None, None, None)
    card = render_model_card(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="7ae5",
        base_license="apache-2.0",
        dataset_license="CC-BY-4.0",
        hyperparameters={"lora": {"r": 16}},
        manifest=None,
        eval_runs=[],
        decision=decision,
    )
    assert "TODO: medir" in card
    assert "pendente" in card


def test_export_copies_only_adapter_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("export_adapter", root / "scripts" / "export_adapter.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["export_adapter"] = module
    spec.loader.exec_module(module)

    run = tmp_path / "train-fake"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"x" * 10)
    (run / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run / "adapter" / "base_model_weights.safetensors").write_bytes(b"should-not-copy")
    monkeypatch.setattr(module, "EXPORTS_DIR", tmp_path / "exports")
    staging = module.export(run)
    names = sorted(p.name for p in staging.iterdir())
    assert "base_model_weights.safetensors" not in names
    manifest = json.loads((staging / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_weights_included"] is False
    assert {f["file"] for f in manifest["files"]} == {"adapter_model.safetensors", "adapter_config.json"}
