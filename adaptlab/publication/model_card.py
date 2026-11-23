"""Model card gerado a partir de dados medidos (config, manifesto, runs de avaliação).

Valores ausentes ficam ``TODO: medir``. Não se afirma nenhuma métrica sem run de origem.
"""

from __future__ import annotations

from typing import Any

from adaptlab.evaluation.report import TODO, dev_scores, latest_by
from adaptlab.evaluation.stopping import StopDecision


def render_model_card(
    *,
    model_name: str,
    model_revision: str,
    base_license: str,
    dataset_license: str | None,
    hyperparameters: dict[str, Any],
    manifest: dict[str, Any] | None,
    eval_runs: list[dict[str, Any]],
    decision: StopDecision,
) -> str:
    dataset_hash = (manifest or {}).get("test_sha256", TODO)
    lora_test = latest_by(eval_runs, "lora", "test")
    lora_dev = latest_by(eval_runs, "lora", "dev")
    dev = dev_scores(eval_runs)
    metric_lines = []
    for label, run in (("dev", lora_dev), ("teste", lora_test)):
        if run is None:
            metric_lines.append(f"- LoRA no {label}: {TODO} (run {TODO})")
        else:
            m = run["metrics"]
            metric_lines.append(
                f"- LoRA no {label}: macro-F1 {m['macro_f1_intent']}, F1 entidades {m['entity_f1']}, "
                f"recall alto risco {m['high_risk_recall']} (run `{run.get('_run')}`)"
            )
    dev_line = ", ".join(f"{k}={v if v is not None else TODO}" for k, v in dev.items())
    return f"""# Model card: AdaptLab (classificador de intents, Mukua Telecom sintética)

## Modelo base

- Nome: `{model_name}`
- Revisão (commit): `{model_revision}`
- Licença: {base_license}

## Dataset

- Sintético, gerado por `scripts/prepare_data.py` (ver `data/DATASET_CARD.md`).
- Licença: {dataset_license or TODO}
- Hash SHA-256 do teste: `{dataset_hash}`

## Hiperparâmetros

```
{hyperparameters}
```

## Métricas (com run de origem)

Macro-F1 no dev por abordagem: {dev_line}

{chr(10).join(metric_lines)}

## Critério de paragem

- Estado: **{decision.status}**. {decision.reason}

## Utilização prevista

Classificar mensagens de apoio ao cliente de uma operadora (fictícia), para encaminhar pedidos
e marcar escalonamentos. Apoio à decisão com revisão humana.

## Utilização não prevista

- Decisões automáticas sem revisão humana, sobretudo quando `requires_human` é verdadeiro.
- Dados reais de clientes, outras operadoras, outros idiomas.
- Factos que mudam (preços, cobertura): não são ensinados por fine-tuning.

## Limitações

- Dados sintéticos; o modelo pode aprender os templates e não a fala real.
- Português de Angola ainda sem revisão humana (ver dataset card).
- Teste pequeno e escalonamentos escassos: IC largo.
- Uma só seed, salvo indicação em contrário nas métricas.

## Riscos de enviesamento

- Localidades e padrões de número escolhidos pelos autores.
- Escalonamento definido pelos autores, não por política da operadora.

## Reprodução

```
.venv\\Scripts\\python scripts\\prepare_data.py --check
.venv\\Scripts\\python scripts\\train.py --config configs\\train_gpu.yaml   (GPU)
.venv\\Scripts\\python scripts\\evaluate.py --approach lora --run-dir runs\\train-XXXX --split dev
.venv\\Scripts\\python scripts\\evaluate.py --report
```
"""
