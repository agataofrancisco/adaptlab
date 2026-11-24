"""Relatório de avaliação em Markdown, só com valores de runs registadas.

Cada número vem de um ``metrics.json`` em ``runs/``. Valores ainda não medidos ficam
``TODO: medir``. O gráfico PNG não é gerado (matplotlib não está instalado); as curvas
aparecem como tabela.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptlab.evaluation.stopping import StopDecision, decide

TODO = "TODO: medir"
HEADER = (
    "| Abordagem | Macro-F1 intent | IC 95% | F1 entidades | JSON válido | Recall alto risco "
    "| Latência média (s) | Tokens de saída (média) | Seeds |"
)
APPROACH_LABELS = {
    "rules": "Regras (palavras-chave e regex)",
    "zero_shot": "Base zero-shot",
    "few_shot": "Base few-shot (k=5)",
    "lora": "LoRA (QLoRA NF4)",
}


def load_eval_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(runs_dir.glob("eval-*/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("mode") == "evaluate":
            payload["_run"] = path.parent.name
            runs.append(payload)
    return runs


def latest_by(runs: list[dict[str, Any]], approach: str, split: str) -> dict[str, Any] | None:
    matches = [
        r
        for r in runs
        if r.get("approach") == approach
        and r.get("split") == split
        and not r.get("sample")
        and not r.get("limit")
    ]
    return matches[-1] if matches else None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return TODO
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(run: dict[str, Any], metric: str) -> str:
    ci = (run.get("bootstrap_ci") or {}).get(metric)
    if not ci:
        return TODO
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def dev_scores(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for approach in ("rules", "zero_shot", "few_shot", "lora"):
        run = latest_by(runs, approach, "dev")
        scores[approach] = run["metrics"]["macro_f1_intent"] if run else None
    return scores


def render_report(
    runs: list[dict[str, Any]],
    margin: float,
    dataset_hash: str | None,
    train_runs: list[dict[str, Any]] | None = None,
    final_log_entries: list[dict[str, Any]] | None = None,
) -> str:
    decision: StopDecision = decide(dev_scores(runs), margin)
    lines: list[str] = [
        "# Relatório de avaliação AdaptLab",
        "",
        "Gerado por `scripts/evaluate.py --report` a partir de `runs/eval-*/metrics.json`.",
        "Só contém valores medidos; o resto está `TODO: medir`.",
        "",
        "## Dataset",
        "",
        f"- Hash SHA-256 do teste (manifesto): `{dataset_hash or TODO}`",
        "",
        *_test_use_section(final_log_entries or []),
        "",
        "## Critério de paragem",
        "",
        f"- Estado: **{decision.status.upper()}**",
        f"- Motivo: {decision.reason}",
        f"- Margem configurada: {margin} (macro-F1 no dev)",
        "",
        "## Resultados no DEV (seleccionar decisões aqui)",
        "",
        HEADER,
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_rows(runs, "dev"))
    lines.extend(
        [
            "",
            "## Resultados no TESTE (uma vez, com `--final`)",
            "",
            HEADER,
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    lines.extend(_rows(runs, "test"))
    lines.extend(_matrix_section(runs))
    lines.extend(_training_section(train_runs or []))
    lines.extend(
        [
            "",
            "## Custo de inferência",
            "",
            f"- Custo por pedido em USD: {TODO} (não há modelo de preços configurado).",
            "- Tokens de saída por pedido: ver a coluna acima (medidos).",
            "",
            "## Limitações",
            "",
            "- Dev e teste têm poucos exemplos com escalonamento (ver `data/DATASET_CARD.md`); "
            "o IC do recall de alto risco é largo.",
            "- A baseline de regras usa as mesmas localidades e expressões temporais do catálogo sintético.",
            "- Com uma só seed, o relatório não apresenta diferenças como robustas.",
            "",
        ]
    )
    return "\n".join(lines)


def _rows(runs: list[dict[str, Any]], split: str) -> list[str]:
    rows = []
    for approach in ("rules", "zero_shot", "few_shot", "lora"):
        run = latest_by(runs, approach, split)
        if run is None:
            rows.append(f"| {APPROACH_LABELS[approach]} |" + f" {TODO} |" * 8)
            continue
        m = run["metrics"]
        rows.append(
            f"| {APPROACH_LABELS[approach]} | {_fmt(m['macro_f1_intent'])} | {_ci(run, 'macro_f1_intent')} | "
            f"{_fmt(m['entity_f1'])} | {_fmt(m['json_valid_rate'])} | {_fmt(m['high_risk_recall'])} | "
            f"{_fmt(m['latency_mean_s'])} | {_fmt(m['output_tokens_mean'], 1)} | {run.get('seeds', 1)} |"
        )
    return rows


def _matrix_section(runs: list[dict[str, Any]]) -> list[str]:
    run = latest_by(runs, "rules", "dev")
    if run is None:
        return ["", "## Matriz de confusão", "", TODO]
    labels = run["metrics"]["confusion_labels"]
    lines = ["", "## Matriz de confusão (regras, DEV)", "", "| gold \\ pred | " + " | ".join(labels) + " |"]
    lines.append("|---|" + "---|" * len(labels))
    for label, row in zip(labels, run["metrics"]["confusion"], strict=True):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    return lines


def load_train_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(runs_dir.glob("train-*/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload["_run"] = path.parent.name
            runs.append(payload)
    return runs


def _training_section(train_runs: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Treino (LoRA)", ""]
    if not train_runs:
        lines.append(
            f"- Adapter, tamanho, VRAM máxima, tempo e curvas de loss: {TODO} (run GPU do notebook ainda não feita)."
        )
        lines.append("- Quantização prevista: NF4 (configs/train_gpu.yaml).")
        return lines
    run = train_runs[-1]
    lines.append(f"- Run: `{run.get('_run')}`")
    lines.append(f"- Quantização: {run.get('quantization', TODO)}")
    lines.append(f"- VRAM máxima (MiB): {_fmt(run.get('max_vram_mib'), 1)}")
    lines.append(f"- Tempo de treino (s): {_fmt(run.get('train_runtime_seconds'), 1)}")
    lines.append(f"- Passos: {run.get('global_step', TODO)}")
    losses = [(e.get("step"), e.get("loss")) for e in run.get("log_history", []) if "loss" in e]
    lines.append("")
    lines.append("| Passo | Loss de treino |")
    lines.append("|---|---|")
    lines.extend(f"| {step} | {loss} |" for step, loss in losses)
    lines.append("")
    lines.append("- Gráfico PNG: TODO (matplotlib não instalado); curvas em tabela.")
    return lines


def _test_use_section(entries: list[dict[str, Any]]) -> list[str]:
    """Quantas vezes o teste foi lido, segundo o registo ``runs/final_eval_log.jsonl``."""
    lines = ["## Uso do conjunto de teste", ""]
    lines.append(f"- Leituras finais registadas: **{len(entries)}**")
    if not entries:
        lines.append("- Ainda não houve leitura final do teste.")
        return lines
    lines.append("")
    lines.append("| # | UTC | Abordagem | Modelo | Commit | Motivo de nova leitura |")
    lines.append("|---|---|---|---|---|---|")
    for e in entries:
        reason = e.get("retest_reason") or "—"
        lines.append(
            f"| {e.get('use_index')} | {e.get('timestamp_utc')} | {e.get('approach')} | "
            f"`{e.get('model_id')}` | `{str(e.get('git_commit'))[:10]}` | {reason} |"
        )
    return lines
