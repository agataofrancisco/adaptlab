"""Gerador real (HF transformers) para zero-shot, few-shot e LoRA.

Decoding greedy (temperatura 0), ``max_new_tokens`` limitado. Importa torch e
transformers só ao instanciar, para que o resto do pacote funcione sem eles.
Não é executado no PC local sem tempo de descarga: ver ``reports/evaluation.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ChatGenerator:
    def __init__(
        self,
        model_name: str,
        revision: str,
        max_new_tokens: int = 96,
        adapter_dir: Path | None = None,
        cpu_dtype: str = "float32",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            dtype = torch.float16
        elif cpu_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif cpu_dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"cpu_dtype inválido: {cpu_dtype!r}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model: Any = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=dtype)
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
        self.model = model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, int(new_tokens.shape[0])
