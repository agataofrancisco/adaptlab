"""Registo auditável das leituras finais do teste (``runs/final_eval_log.jsonl``).

Usa a baseline de regras (sem modelo) para correr ``evaluate()`` de ponta a ponta.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from adaptlab.evaluation import final_log
from adaptlab.evaluation.report import render_report

ROOT = Path(__file__).resolve().parents[1]
REAL_PROCESSED = ROOT / "data" / "processed"


def _load_evaluate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_cli", ROOT / "scripts" / "evaluate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_cli"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = _load_evaluate()
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(module, "RUNS_DIR", runs)
    monkeypatch.setattr(module, "LOG_PATH", runs / final_log.LOG_FILENAME)
    monkeypatch.setattr(module, "PROCESSED_DIR", REAL_PROCESSED)
    # bootstrap curto: o teste da cadeia de custódia não precisa de IC de 1000 amostras.
    monkeypatch.setattr(
        module,
        "load_eval_config",
        lambda: {"bootstrap_samples": 5, "seed": 42, "max_new_tokens": 96, "few_shot_k": 5, "stop_margin": 0.02},
    )
    return module


def _args(**overrides: object) -> argparse.Namespace:
    base = {
        "approach": "rules",
        "split": "test",
        "final": True,
        "run_dir": None,
        "limit": 0,
        "allow_retest": None,
        "sample": 0,
        "seed": 42,
        "model": None,
        "revision": None,
        "dtype": "float32",
        "report": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _log_rows(cli: ModuleType) -> list[dict]:
    return final_log.read_log(cli.LOG_PATH)


def _run_dirs(cli: ModuleType) -> list[Path]:
    return sorted(p for p in cli.RUNS_DIR.glob("eval-*") if p.is_dir())


def test_first_final_run_is_logged(cli: ModuleType) -> None:
    assert cli.evaluate(_args()) == 0
    rows = _log_rows(cli)
    assert len(rows) == 1
    row = rows[0]
    assert row["use_index"] == 1
    assert row["retest_reason"] is None
    assert row["approach"] == "rules"
    assert row["model_id"] == "rules"
    assert row["examples"] == 137
    assert len(row["config_hash"]) == 64
    assert len(row["git_commit"]) == 40 or row["git_commit"] == "unknown"
    manifest = json.loads((REAL_PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    assert row["test_sha256"] == manifest["test_sha256"]
    assert row["timestamp_utc"]


def test_second_final_run_is_refused_without_flag(cli: ModuleType) -> None:
    assert cli.evaluate(_args()) == 0
    runs_before = _run_dirs(cli)
    assert cli.evaluate(_args()) == 5
    assert len(_log_rows(cli)) == 1  # a recusa não regista leitura
    assert _run_dirs(cli) == runs_before  # nem gera predições


def test_retest_with_flag_logs_the_reason(cli: ModuleType) -> None:
    assert cli.evaluate(_args()) == 0
    reason = "corrigida a normalização de localidades; nova leitura documentada"
    assert cli.evaluate(_args(allow_retest=reason)) == 0
    rows = _log_rows(cli)
    assert [r["use_index"] for r in rows] == [1, 2]
    assert rows[0]["retest_reason"] is None
    assert rows[1]["retest_reason"] == reason


def test_retest_flag_with_blank_reason_is_refused(cli: ModuleType) -> None:
    assert cli.evaluate(_args()) == 0
    assert cli.evaluate(_args(allow_retest="   ")) == 5
    assert len(_log_rows(cli)) == 1


def test_tampered_test_set_is_refused_and_not_logged(cli: ModuleType, tmp_path: Path) -> None:
    tampered = tmp_path / "processed"
    shutil.copytree(REAL_PROCESSED, tampered)
    test_file = tampered / "test.jsonl"
    lines = test_file.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["message"] = row["message"] + " (alterado)"
    lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True)
    test_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cli.PROCESSED_DIR = tampered
    assert cli.evaluate(_args()) == 4
    assert _log_rows(cli) == []
    assert _run_dirs(cli) == []


def test_report_counts_test_set_uses(cli: ModuleType) -> None:
    assert cli.evaluate(_args()) == 0
    assert cli.evaluate(_args(allow_retest="segunda leitura, motivo registado")) == 0
    entries = _log_rows(cli)
    text = render_report([], margin=0.02, dataset_hash="abc", final_log_entries=entries)
    assert "Leituras finais registadas: **2**" in text
    assert "segunda leitura, motivo registado" in text


def test_report_without_uses_says_so() -> None:
    text = render_report([], margin=0.02, dataset_hash=None, final_log_entries=[])
    assert "Leituras finais registadas: **0**" in text


def test_config_hash_is_deterministic_and_sensitive_to_config() -> None:
    a = final_log.canonical_config_hash({"seed": 1}, "rules")
    assert a == final_log.canonical_config_hash({"seed": 1}, "rules")
    assert a != final_log.canonical_config_hash({"seed": 2}, "rules")


def test_read_log_rejects_corrupt_line(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text('{"ok": 1}\nnão é json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        final_log.read_log(path)
