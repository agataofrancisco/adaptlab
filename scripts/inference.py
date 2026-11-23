"""Inferência de uma mensagem pela linha de comandos (mesmo caminho de código da API).

Uso:
    .venv\\Scripts\\python scripts\\inference.py --message "a net ta fraca no kilamba desde ontem"          # demo
    Modo real: definir ADAPTLAB_MODE=real e ADAPTLAB_ADAPTER_DIR=exports\\adapter-train-XXXX e depois
    .venv\\Scripts\\python scripts\\inference.py --model base --message "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptlab.serving.predictor import DemoPredictor  # noqa: E402
from api.main import FIXTURES_PATH, MAX_MESSAGE_CHARS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--message", required=True)
    parser.add_argument("--model", choices=("base", "adapted"), default="adapted")
    args = parser.parse_args(argv)

    message = args.message.strip()
    if not message or len(message) > MAX_MESSAGE_CHARS:
        print(f"ERRO: a mensagem tem de ter entre 1 e {MAX_MESSAGE_CHARS} caracteres", file=sys.stderr)
        return 2

    mode = os.environ.get("ADAPTLAB_MODE", "demo").strip().lower()
    if mode == "real":
        from adaptlab.serving.loader import load_real_predictor

        predictor = load_real_predictor()
    else:
        predictor = DemoPredictor(FIXTURES_PATH)
    outcome = predictor.predict(message, args.model)
    print(json.dumps(outcome.body, ensure_ascii=False, indent=2))
    return 0 if outcome.status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
