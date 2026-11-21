"""Treino LoRA/QLoRA do AdaptLab.

Uso (a partir da raiz do projecto):
    .venv\\Scripts\\python scripts\\train.py --dry-run          # CPU, valida dados e tokenização
    .venv\\Scripts\\python scripts\\train.py                    # treino real (exige GPU; ver notebook)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptlab.training.config import ConfigError, load_train_config  # noqa: E402
from adaptlab.training.tracking import (  # noqa: E402
    EnvironmentRequirementError,
    new_run_dir,
    write_json,
    write_run_metadata,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train.yaml")
    parser.add_argument("--dry-run", action="store_true", help="valida dados e tokenização em CPU, sem treino real")
    args = parser.parse_args(argv)

    try:
        config = load_train_config(args.config)
    except ConfigError as exc:
        print(f"ERRO de configuração: {exc}", file=sys.stderr)
        return 2

    mode = "dry_run" if args.dry_run else "train"
    run_dir = new_run_dir(config.runs_dir, mode)
    write_run_metadata(run_dir, config, mode)
    try:
        if args.dry_run:
            from adaptlab.training.dryrun import run_dry_run

            report = run_dry_run(config, run_dir)
            write_json(run_dir / "dry_run_report.json", report)
        else:
            from adaptlab.training.trainer import run_training

            report = run_training(config, run_dir)
    except EnvironmentRequirementError as exc:
        print(f"ERRO de ambiente: {exc}", file=sys.stderr)
        return 3

    print(f"run: {run_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if args.dry_run and report.get("status") != "ok":
        print(f"ATENÇÃO: estado do dry-run = {report.get('status')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
