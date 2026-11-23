"""Exporta o adapter LoRA (só adapter + configuração PEFT + hashes) e gera o model card.

Nunca copia os pesos do modelo base. Não publica nada: o portão de publicação
(``--check-publication``) só diz se a publicação estaria permitida, e a publicação
no Hugging Face é um passo manual.

Uso:
    .venv\\Scripts\\python scripts\\export_adapter.py --run-dir runs\\train-XXXX
    .venv\\Scripts\\python scripts\\export_adapter.py --write-card
    .venv\\Scripts\\python scripts\\export_adapter.py --check-publication --confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptlab.evaluation.report import dev_scores, load_eval_runs  # noqa: E402
from adaptlab.evaluation.stopping import decide  # noqa: E402
from adaptlab.paths import PROCESSED_DIR  # noqa: E402
from adaptlab.publication.gate import check_publication  # noqa: E402
from adaptlab.publication.model_card import render_model_card  # noqa: E402
from adaptlab.training.config import load_train_config  # noqa: E402

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
RUNS_DIR = PROJECT_ROOT / "runs"
EXPORTS_DIR = PROJECT_ROOT / "exports"
CARD_PATH = PROJECT_ROOT / "reports" / "model_card.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_yaml(name: str) -> dict[str, Any]:
    with (PROJECT_ROOT / "configs" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stop_status() -> str:
    margin = float(_raw_yaml("evaluate.yaml")["stop_margin"])
    return decide(dev_scores(load_eval_runs(RUNS_DIR)), margin).status


def export(run_dir: Path) -> Path:
    adapter_dir = run_dir / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter não encontrado em {adapter_dir}")
    missing = [name for name in ADAPTER_FILES if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"ficheiros do adapter em falta: {missing}")
    staging = EXPORTS_DIR / f"adapter-{run_dir.name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    sums = []
    for name in ADAPTER_FILES:
        target = staging / name
        shutil.copy2(adapter_dir / name, target)
        sums.append({"file": name, "sha256": _sha256(target), "bytes": target.stat().st_size})
    config = load_train_config()
    manifest = {
        "base_model": config.model_name,
        "base_model_revision": config.model_revision,
        "base_weights_included": False,
        "files": sums,
        "adapter_total_bytes": sum(item["bytes"] for item in sums),
        "source_run": run_dir.name,
    }
    (staging / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (staging / "SHA256SUMS.txt").write_text(
        "".join(f"{item['sha256']}  {item['file']}\n" for item in sums), encoding="utf-8"
    )
    if CARD_PATH.exists():
        shutil.copy2(CARD_PATH, staging / "README.md")
    return staging


def write_card() -> Path:
    model = _raw_yaml("train.yaml")["model"]
    data_license = _raw_yaml("data.yaml").get("dataset_license")
    config = load_train_config()
    manifest_path = PROCESSED_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    runs = load_eval_runs(RUNS_DIR)
    margin = float(_raw_yaml("evaluate.yaml")["stop_margin"])
    decision = decide(dev_scores(runs), margin)
    card = render_model_card(
        model_name=config.model_name,
        model_revision=config.model_revision,
        base_license=str(model.get("license", "desconhecida")),
        dataset_license=data_license,
        hyperparameters={
            "lora": config.raw["lora"],
            "training": config.raw["training"],
            "quantization": config.quantization,
        },
        manifest=manifest,
        eval_runs=runs,
        decision=decision,
    )
    CARD_PATH.write_text(card, encoding="utf-8")
    return CARD_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, help="pasta de uma run de treino (runs/train-*)")
    parser.add_argument("--write-card", action="store_true", help="escreve reports/model_card.md")
    parser.add_argument("--check-publication", action="store_true", help="mostra o portão de publicação (não publica)")
    parser.add_argument("--confirm", action="store_true", help="confirmação manual para o portão")
    args = parser.parse_args(argv)

    if args.write_card:
        print(f"model card: {write_card()}")
    if args.run_dir:
        print(f"adapter exportado (sem pesos do modelo base): {export(args.run_dir)}")
    if args.check_publication:
        model = _raw_yaml("train.yaml")["model"]
        check = check_publication(
            base_license=str(model.get("license", "")),
            dataset_license=_raw_yaml("data.yaml").get("dataset_license"),
            stop_status=stop_status(),
            manual_confirmation=args.confirm,
        )
        print(json.dumps(check.as_dict(), ensure_ascii=False, indent=2))
        if not check.allowed:
            print("PUBLICAÇÃO BLOQUEADA:", "; ".join(check.reasons), file=sys.stderr)
            return 1
        print("Portão aberto. A publicação no Hugging Face é um passo manual; este script não envia nada.")
    if not (args.write_card or args.run_dir or args.check_publication):
        parser.error("indique --write-card, --run-dir ou --check-publication")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
