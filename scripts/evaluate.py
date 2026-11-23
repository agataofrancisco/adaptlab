"""Avaliação do AdaptLab: baselines e LoRA no mesmo conjunto, critério de paragem e relatório.

Exemplos (a partir da raiz do projecto):
    .venv\\Scripts\\python scripts\\evaluate.py --approach rules --split dev
    .venv\\Scripts\\python scripts\\evaluate.py --approach zero_shot --split dev --limit 40
    .venv\\Scripts\\python scripts\\evaluate.py --approach lora --run-dir runs\\train-XXXX --split dev
    .venv\\Scripts\\python scripts\\evaluate.py --approach lora --run-dir runs\\train-XXXX --split test --final
    .venv\\Scripts\\python scripts\\evaluate.py --report

O conjunto de teste só é lido com ``--final`` e só se o hash bater com o manifesto.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptlab.data.card import validate_card  # noqa: E402
from adaptlab.evaluation import final_log  # noqa: E402
from adaptlab.evaluation.approaches import (  # noqa: E402
    APPROACHES,
    few_shot_turns,
    run_model,
    run_rules,
    select_few_shot,
    write_predictions,
)
from adaptlab.evaluation.metrics import bootstrap_ci, compute_metrics  # noqa: E402
from adaptlab.evaluation.report import load_eval_runs, load_train_runs, render_report  # noqa: E402
from adaptlab.evaluation.stopping import TamperedTestSetError, verify_test_hash  # noqa: E402
from adaptlab.paths import PROCESSED_DIR  # noqa: E402
from adaptlab.training.config import load_train_config  # noqa: E402
from adaptlab.training.dataset import read_jsonl, spread_sample  # noqa: E402

EVAL_CONFIG = PROJECT_ROOT / "configs" / "evaluate.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"
LOG_PATH = RUNS_DIR / final_log.LOG_FILENAME
REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation.md"


def load_eval_config(path: Path = EVAL_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuração de avaliação inválida: {path}")
    return data


def _model_id(args: argparse.Namespace, train_cfg: Any) -> str:
    if args.approach == "rules":
        return "rules"
    if args.approach == "lora":
        return f"lora:{Path(args.run_dir).name}"
    return f"{train_cfg.model_name}@{train_cfg.model_revision}"


def evaluate(args: argparse.Namespace) -> int:
    ev = load_eval_config()
    train_cfg = load_train_config()
    if args.split == "test" and not args.final:
        print("ERRO: o teste só é lido com --final (uma vez, depois da escolha no dev).", file=sys.stderr)
        return 2
    if args.approach == "lora" and not args.run_dir:
        print("ERRO: --approach lora exige --run-dir (pasta de uma run de treino).", file=sys.stderr)
        return 2

    adapter: Path | None = None
    if args.approach == "lora":
        adapter = Path(args.run_dir) / "adapter"
        if not adapter.is_dir():
            print(f"ERRO: adapter não encontrado em {adapter}", file=sys.stderr)
            return 2
    model_id = _model_id(args, train_cfg)

    if args.split == "test":
        # 1) Recusa antes de tocar no teste se já foi lido e não há --allow-retest.
        try:
            prior = final_log.read_log(LOG_PATH)
            retest_reason = final_log.check_retest(len(prior), args.allow_retest)
        except (final_log.RetestRefusedError, ValueError) as exc:
            print(f"RECUSADO: {exc}", file=sys.stderr)
            return 5
        # 2) Verifica o hash do teste contra o manifesto antes de o usar.
        records = read_jsonl(PROCESSED_DIR / "test.jsonl")
        manifest = json.loads((PROCESSED_DIR / "manifest.json").read_text(encoding="utf-8"))
        try:
            verify_test_hash(records, manifest["test_sha256"])
        except TamperedTestSetError as exc:
            print(f"RECUSADO: {exc}", file=sys.stderr)
            return 4
        if args.limit:
            records = spread_sample(records, args.limit)
        # 3) Regista a leitura ANTES de gerar predições (o acesso fica registado mesmo se falhar).
        config_hash = final_log.canonical_config_hash(ev, train_cfg.raw, args.approach, model_id)
        final_log.append_entry(
            LOG_PATH,
            final_log.build_entry(
                git_root=PROJECT_ROOT,
                config_hash=config_hash,
                model_id=model_id,
                approach=args.approach,
                test_sha256=manifest["test_sha256"],
                examples=len(records),
                use_index=len(prior) + 1,
                retest_reason=retest_reason,
            ),
        )
    else:
        records = read_jsonl(PROCESSED_DIR / f"{args.split}.jsonl")
        if args.limit:
            records = spread_sample(records, args.limit)

    started = datetime.now(UTC)
    extra: dict[str, Any] = {}
    if args.approach == "rules":
        predictions = run_rules(records)
    else:
        from adaptlab.evaluation.generation import ChatGenerator

        generator = ChatGenerator(
            train_cfg.model_name,
            train_cfg.model_revision,
            max_new_tokens=int(ev["max_new_tokens"]),
            adapter_dir=adapter,
        )
        prefix: list[dict[str, str]] = []
        if args.approach == "few_shot":
            train_records = read_jsonl(PROCESSED_DIR / "train.jsonl")
            shots = select_few_shot(train_records, int(ev["few_shot_k"]), int(ev["seed"]))
            prefix = few_shot_turns(shots)
            extra["few_shot_ids"] = [s["id"] for s in shots]
        predictions = run_model(records, generator, prefix_turns=prefix)
        extra["device"] = generator.device

    metrics = compute_metrics(records, predictions)
    samples = int(ev["bootstrap_samples"])
    ci = {
        "macro_f1_intent": bootstrap_ci(records, predictions, "macro_f1_intent", samples, int(ev["seed"])),
        "high_risk_recall": bootstrap_ci(records, predictions, "high_risk_recall", samples, int(ev["seed"])),
    }
    run_dir = (
        RUNS_DIR / f"eval-{args.approach}-{args.split}-{started.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:6]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    write_predictions(run_dir / "predictions.jsonl", predictions)
    payload = {
        "mode": "evaluate",
        "approach": args.approach,
        "split": args.split,
        "final": bool(args.final),
        "limit": args.limit,
        "seeds": 1,
        "examples": len(records),
        "lora_run": Path(args.run_dir).name if args.run_dir else None,
        "model": {"name": train_cfg.model_name, "revision": train_cfg.model_revision},
        "bootstrap_ci": ci,
        "metrics": metrics,
        "started_utc": started.isoformat(),
        **extra,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"run: {run_dir}")
    print(
        json.dumps(
            {k: metrics[k] for k in ("n", "macro_f1_intent", "entity_f1", "json_valid_rate", "high_risk_recall")},
            ensure_ascii=False,
        )
    )
    return 0


def report() -> int:
    ev = load_eval_config()
    manifest_path = PROCESSED_DIR / "manifest.json"
    dataset_hash = None
    if manifest_path.exists():
        dataset_hash = json.loads(manifest_path.read_text(encoding="utf-8")).get("test_sha256")
    card_path = PROJECT_ROOT / "data" / "DATASET_CARD.md"
    if card_path.exists():
        problems = validate_card(card_path.read_text(encoding="utf-8"))
        if problems:
            print(f"AVISO: dataset card incompleto: {problems}", file=sys.stderr)
    text = render_report(
        load_eval_runs(RUNS_DIR),
        margin=float(ev["stop_margin"]),
        dataset_hash=dataset_hash,
        train_runs=load_train_runs(RUNS_DIR),
        final_log_entries=final_log.read_log(LOG_PATH),
    )
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"relatório: {REPORT_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approach", choices=APPROACHES)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--final", action="store_true", help="necessário para ler o teste (fica registado)")
    parser.add_argument(
        "--allow-retest",
        metavar="MOTIVO",
        help="obrigatório para qualquer leitura final depois da primeira; o motivo fica no registo",
    )
    parser.add_argument("--run-dir", help="pasta de uma run de treino (para --approach lora)")
    parser.add_argument("--limit", type=int, default=0, help="avalia só uma amostra espalhada (CPU)")
    parser.add_argument("--report", action="store_true", help="gera reports/evaluation.md")
    args = parser.parse_args(argv)
    if args.report:
        return report()
    if not args.approach:
        parser.error("indique --approach ou --report")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
