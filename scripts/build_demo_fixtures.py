"""Gera as fixtures do modo demo a partir de uma run MEDIDA (previsões da baseline de regras, dev).

Não gera nem inventa respostas de modelo. Cada linha guarda a run de origem.

Uso:
    .venv\\Scripts\\python scripts\\build_demo_fixtures.py --run runs\\eval-rules-dev-XXXX --size 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptlab.paths import PROCESSED_DIR  # noqa: E402
from adaptlab.training.dataset import read_jsonl, spread_sample  # noqa: E402

OUT = PROJECT_ROOT / "data" / "demo" / "precomputed.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="pasta runs/eval-rules-dev-*")
    parser.add_argument("--size", type=int, default=20)
    args = parser.parse_args(argv)

    predictions = {}
    for line in (args.run / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            predictions[row["id"]] = row
    dev = read_jsonl(PROCESSED_DIR / "dev.jsonl")
    chosen = spread_sample([r for r in dev if r["id"] in predictions], args.size)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    source = f"{args.run.as_posix()}/predictions.jsonl"
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for record in chosen:
            row = {
                "message": record["message"],
                "model": "regras_fixture",
                "label": predictions[record["id"]]["label"],
                "source": source,
                "example_id": record["id"],
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{len(chosen)} fixtures -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
