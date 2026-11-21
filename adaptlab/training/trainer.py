"""Treino LoRA/QLoRA real (corre em GPU; no PC local só é alcançado sem CUDA via erro claro).

O caminho de código é o mesmo do dry-run (``dryrun.py``): mesmo formato, mesma
loss completion-only, mesmo SFTTrainer. A diferença é o modelo (pesos reais, opcionalmente
em 4-bit) e o dispositivo.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from adaptlab.training.config import TrainConfig
from adaptlab.training.dataset import read_jsonl, to_prompt_completion_rows
from adaptlab.training.tracking import require_quantization_support, write_json


def run_training(config: TrainConfig, run_dir: Path) -> dict[str, Any]:
    require_quantization_support(config)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cuda = torch.cuda.is_available()
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16 else torch.float16 if cuda else torch.float32

    quant_config = None
    if config.quantization == "nf4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        dtype=compute_dtype,
        quantization_config=quant_config,
    )

    train_rows = to_prompt_completion_rows(read_jsonl(config.data_dir / config.train_file))
    dev_rows = to_prompt_completion_rows(read_jsonl(config.data_dir / config.dev_file))
    train_ds = Dataset.from_list(train_rows)
    dev_ds = Dataset.from_list(dev_rows)

    lora = config.lora
    t = config.training
    args = SFTConfig(
        output_dir=str(run_dir / "trainer"),
        max_length=config.max_length,
        learning_rate=t.learning_rate,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        save_strategy=t.save_strategy,
        seed=t.seed,
        completion_only_loss=t.completion_only_loss,
        bf16=bf16,
        fp16=bool(cuda and not bf16),
        report_to="none",
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
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    seconds = round(time.perf_counter() - started, 1)
    vram_mib = round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None

    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)

    metrics = {
        "train_runtime_seconds": seconds,
        "global_step": int(result.global_step),
        "max_vram_mib": vram_mib,
        "quantization": config.quantization,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "log_history": trainer.state.log_history,
        "adapter_dir": str(adapter_dir.relative_to(run_dir)),
    }
    write_json(run_dir / "metrics.json", metrics)
    return metrics
