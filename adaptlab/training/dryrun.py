"""Dry-run em CPU: valida dados, chat template, tokenização e máscara de loss.

Não carrega os pesos do modelo base (~1 GB). Usa o tokenizer REAL do modelo
configurado e um modelo Qwen2 minúsculo construído localmente com a mesma
vocabulário. Depois corre o SFTTrainer do TRL com LoRA e 1-2 passos, para
verificar exactamente o caminho de código que o treino real usa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptlab.training.config import TrainConfig
from adaptlab.training.dataset import read_jsonl, spread_sample, to_prompt_completion_rows
from adaptlab.training.format import SYSTEM_PROMPT

IGNORE_INDEX = -100


def build_tiny_model(tokenizer: Any, config: TrainConfig) -> Any:
    """Qwen2 com os mesmos nomes de módulos do modelo real, mas dimensões minúsculas."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    dry = config.dry_run
    model_config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=dry.tiny_hidden_size,
        intermediate_size=dry.tiny_intermediate,
        num_hidden_layers=dry.tiny_layers,
        num_attention_heads=dry.tiny_heads,
        num_key_value_heads=dry.tiny_kv_heads,
        max_position_embeddings=config.max_length,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(model_config)


def inspect_masking(trainer: Any, tokenizer: Any, index: int = 0) -> dict[str, Any]:
    """Mostra o que a loss vê: tokens com label (completion) e tokens mascarados (prompt)."""
    features = trainer.train_dataset[index]
    batch = trainer.data_collator([features])
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    masked_prefix = 0
    while masked_prefix < len(labels) and labels[masked_prefix] == IGNORE_INDEX:
        masked_prefix += 1
    label_ids = [tok for tok, lab in zip(input_ids, labels, strict=True) if lab != IGNORE_INDEX]
    tail_masked = all(lab == IGNORE_INDEX for lab in labels[masked_prefix + len(label_ids) :])
    return {
        "example_index": index,
        "total_tokens": len(input_ids),
        "masked_prompt_tokens": masked_prefix,
        "labeled_tokens": len(label_ids),
        "contiguous_labels": tail_masked,
        "prompt_tail_text": tokenizer.decode(input_ids[max(0, masked_prefix - 12) : masked_prefix]),
        "labeled_text": tokenizer.decode(label_ids),
    }


def run_dry_run(config: TrainConfig, run_dir: Path) -> dict[str, Any]:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    train_path = config.data_dir / config.train_file
    records = read_jsonl(train_path)
    sample = spread_sample(records, config.dry_run.sample_size)
    rows = to_prompt_completion_rows(sample)

    lengths = []
    for row in rows:
        text = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    truncated = sum(1 for n in lengths if n > config.max_length)

    dataset = Dataset.from_list(rows)
    model = build_tiny_model(tokenizer, config)
    lora = config.lora
    args = SFTConfig(
        output_dir=str(run_dir / "trainer"),
        max_length=config.max_length,
        per_device_train_batch_size=2,
        max_steps=config.dry_run.max_steps,
        learning_rate=config.training.learning_rate,
        logging_steps=1,
        seed=config.training.seed,
        completion_only_loss=config.training.completion_only_loss,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
    )
    peft_config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias=lora.bias,
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    masking = inspect_masking(trainer, tokenizer, index=0)
    result = trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]

    report = {
        "mode": "dry_run",
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "weights": "tiny Qwen2 construído localmente (sem download dos pesos)",
        "tokenizer_vocab_size": len(tokenizer),
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "examples": len(rows),
        "tokens_per_example": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
        },
        "max_length": config.max_length,
        "truncated_examples": truncated,
        "masking": masking,
        "steps": int(result.global_step),
        "train_loss_per_step": [round(float(x), 4) for x in losses],
        "trainable_parameters": _trainable_parameters(trainer.model),
    }
    if masking["masked_prompt_tokens"] == 0 or not masking["contiguous_labels"]:
        report["status"] = "fail_masking"
    elif truncated:
        report["status"] = "warn_truncation"
    else:
        report["status"] = "ok"
    return report


def _trainable_parameters(model: Any) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
