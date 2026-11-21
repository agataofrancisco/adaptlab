"""Pipeline de dados: gerar -> validar -> deduplicar -> splits -> fugas -> revisão -> escrever.

Falha (``DataPipelineError``) se: algum label for inválido, houver fuga entre
splits, ou a cobertura mínima configurada não for cumprida. A revisão humana
pendente não falha o pipeline, mas fica registada no manifesto e no card.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from adaptlab.data.generator import ALL_TAGS, ENTITY_KEYS, Example, generate
from adaptlab.data.quality import (
    check_leakage,
    drop_near_duplicates,
    find_exact_duplicates,
    validate_example,
)
from adaptlab.data.review import (
    apply_decisions,
    build_sample,
    read_decisions,
    write_jsonl,
)
from adaptlab.data.splits import canonical_hash, default_assignment, split_examples
from adaptlab.schema import INTENTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"


class DataPipelineError(RuntimeError):
    """Erro de qualidade de dados que impede gerar um dataset válido."""


@dataclass
class PipelineResult:
    splits: dict[str, list[Example]]
    manifest: dict[str, Any]
    review_rows: list[dict[str, Any]]
    problems: list[str] = field(default_factory=list)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise DataPipelineError(f"configuração inválida em {path}")
    for key in ("review_sample_path", "review_decisions"):
        if isinstance(data.get(key), str):
            data[key] = str(PROJECT_ROOT / data[key])
    return data


def build_dataset(config: dict[str, Any]) -> PipelineResult:
    seed = int(config["seed"])
    raw = generate(seed=seed, realizations=int(config["realizations_per_template"]))
    problems: list[str] = []

    invalid = {ex.example_id: validate_example(ex) for ex in raw}
    invalid = {k: v for k, v in invalid.items() if v}
    if invalid:
        sample = list(invalid.items())[:5]
        raise DataPipelineError(f"{len(invalid)} exemplos inválidos, por exemplo: {sample}")

    exact = set(find_exact_duplicates(raw))
    unique = [ex for ex in raw if ex.example_id not in exact]
    unique, near_dropped = drop_near_duplicates(unique, float(config["dedupe_threshold"]))

    assignment = default_assignment(seed, config["scenarios_per_intent"])
    splits = split_examples(unique, assignment)

    leakage = check_leakage(splits, float(config["leakage_threshold"]))
    if leakage["shared_templates"]:
        raise DataPipelineError(f"templates partilhados entre splits: {leakage['shared_templates']}")
    if leakage["near_duplicates"]:
        pairs = [(p.left, p.right, p.score) for p in leakage["near_duplicates"][:5]]  # type: ignore[index]
        raise DataPipelineError(f"quase-duplicados entre splits: {pairs}")

    sample_size = int(config["review_sample_size"])
    review_rows = build_sample(splits["train"], sample_size, seed)
    sample_ids = [row["id"] for row in review_rows]
    decisions, decision_errors = read_decisions(Path(config["review_decisions"]))
    review = apply_decisions(splits["train"], sample_ids, decisions)
    problems.extend(decision_errors)
    problems.extend(review.errors)
    if review.errors or decision_errors:
        raise DataPipelineError(f"decisões de revisão inválidas: {problems}")
    splits["train"] = review.examples

    coverage_problems = check_coverage(splits, config)
    problems.extend(coverage_problems)
    if coverage_problems:
        raise DataPipelineError(f"cobertura mínima não cumprida: {coverage_problems}")

    manifest = build_manifest(
        config=config,
        raw_count=len(raw),
        exact_dropped=len(exact),
        near_dropped=len(near_dropped),
        splits=splits,
        review_rows=review_rows,
        review=review,
        decisions_count=len(decisions),
        leakage_pairs=len(leakage["near_duplicates"]),  # type: ignore[arg-type]
    )
    return PipelineResult(splits=splits, manifest=manifest, review_rows=review_rows, problems=problems)


def check_coverage(splits: dict[str, list[Example]], config: dict[str, Any]) -> list[str]:
    """Mínimos por intent (no total) e por dificuldade (no total), configurados em YAML."""
    problems: list[str] = []
    all_examples = [ex for split in splits.values() for ex in split]
    intent_counts = Counter(ex.intent for ex in all_examples)
    for intent in INTENTS:
        if intent_counts[intent] < int(config["min_per_intent"]):
            problems.append(f"intent {intent}: {intent_counts[intent]} < {config['min_per_intent']}")
    tag_counts = Counter(tag for ex in all_examples for tag in ex.tags)
    for tag in ALL_TAGS:
        if tag_counts[tag] < int(config["min_per_tag"]):
            problems.append(f"tag {tag}: {tag_counts[tag]} < {config['min_per_tag']}")
    for split_name in splits:
        if not splits[split_name]:
            problems.append(f"split {split_name} vazio")
    return problems


def summarize(examples: list[Example]) -> dict[str, Any]:
    """Estatísticas de um conjunto: intents, prioridades, tags, entidades nulas, comprimentos."""
    intents = Counter(ex.intent for ex in examples)
    priorities = Counter(ex.label["priority"] for ex in examples)
    tags = Counter(tag for ex in examples for tag in ex.tags)
    null_rates = {
        key: round(
            sum(1 for ex in examples if ex.label["entities"][key] is None) / max(len(examples), 1),
            4,
        )
        for key in ENTITY_KEYS
    }
    lengths = [len(ex.message) for ex in examples]
    return {
        "n": len(examples),
        "intents": dict(sorted(intents.items())),
        "priorities": dict(sorted(priorities.items())),
        "requires_human_true": sum(1 for ex in examples if ex.label["requires_human"]),
        "tags": dict(sorted(tags.items())),
        "entity_null_rate": null_rates,
        "message_chars_mean": round(sum(lengths) / max(len(lengths), 1), 2),
        "message_chars_max": max(lengths) if lengths else 0,
    }


def build_manifest(
    *,
    config: dict[str, Any],
    raw_count: int,
    exact_dropped: int,
    near_dropped: int,
    splits: dict[str, list[Example]],
    review_rows: list[dict[str, Any]],
    review: Any,
    decisions_count: int,
    leakage_pairs: int,
) -> dict[str, Any]:
    pending = len(review.pending)
    return {
        "seed": config["seed"],
        "generator": {
            "realizations_per_template": config["realizations_per_template"],
            "raw_count": raw_count,
            "exact_duplicates_dropped": exact_dropped,
            "near_duplicates_dropped": near_dropped,
        },
        "splits": {name: summarize(items) for name, items in splits.items()},
        "split_hashes": {name: canonical_hash(items) for name, items in splits.items()},
        "test_sha256": canonical_hash(splits["test"]),
        "leakage": {
            "shared_templates": 0,
            "near_duplicate_pairs_across_splits": leakage_pairs,
            "threshold": config["leakage_threshold"],
        },
        "review": {
            "status": "pendente" if pending else "concluida",
            "sample_size": len(review_rows),
            "decisions_recorded": decisions_count,
            "pending": pending,
            "rejected": len(review.rejected),
            "corrected": len(review.corrected),
            "note": "Revisão humana pendente: a pessoa revisora preenche data/review/decisions.jsonl.",
        },
    }


def write_outputs(result: PipelineResult, config: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in result.splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", [ex.to_dict() for ex in items])
    (out_dir / "manifest.json").write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(Path(config["review_sample_path"]), result.review_rows)
