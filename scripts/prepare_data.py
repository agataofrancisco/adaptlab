"""Gera o dataset sintético, valida-o, verifica fugas e escreve os artefactos.

Uso (a partir da raiz do projecto):
    .venv\\Scripts\\python scripts\\prepare_data.py
    .venv\\Scripts\\python scripts\\prepare_data.py --check   # não escreve ficheiros
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptlab.data.card import render_card  # noqa: E402
from adaptlab.data.pipeline import (  # noqa: E402
    DataPipelineError,
    build_dataset,
    load_config,
    write_outputs,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CARD_PATH = PROJECT_ROOT / "data" / "DATASET_CARD.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml")
    parser.add_argument("--check", action="store_true", help="valida sem escrever ficheiros")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    try:
        result = build_dataset(config)
    except DataPipelineError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({k: v["n"] for k, v in result.manifest["splits"].items()}, ensure_ascii=False))
    print(f"revisao: {result.manifest['review']['status']} (pendentes={result.manifest['review']['pending']})")
    if result.problems:
        print("avisos:", result.problems)
    if args.check:
        print("check ok (nada escrito)")
        return 0

    write_outputs(result, config, PROCESSED_DIR)
    CARD_PATH.write_text(render_card(result.manifest, config), encoding="utf-8")
    print(f"escrito em {PROCESSED_DIR} e {CARD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
