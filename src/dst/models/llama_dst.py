from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _is_local_model_path(name_or_path: str) -> bool:
    p = Path(name_or_path)
    return p.exists() and p.is_dir()


def _norm_pred(text: str) -> str:
    text = (text or "").strip().lower()
    if text in {"", "none", "not mentioned", "not given"}:
        return "none"
    return text


class LlamaDSTModel:
    """
    Llama 3.1 Instruct wrapper for DST slot filling.

    Works for both:
      - meta-llama/Llama-3.1-70B-Instruct  (~80GB VRAM, 4×A100)
      - meta-llama/Llama-3.1-8B-Instruct   (~16GB VRAM, 1×A100)

    Requires HuggingFace access to the meta-llama gated repo:
      https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
    """

    DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading model:", self.model_name)
        print("Device:", self.device)

        local = _is_local_model_path(self.model_name)
        load_kwargs: dict = dict(
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device != "cpu" else None,
        )

        if local:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, local_files_only=True, **load_kwargs
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )

        # Llama 3.1 has no pad token by default — use eos
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()

    def predict(self, prompt: str, max_new_tokens: int = 20) -> str:
        """
        Run slot-value prediction using Llama 3.1's chat template.
        A system message instructs the model to output only the slot value.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a dialogue state tracker. "
                    "Given a dialogue and a slot, output ONLY the slot value — "
                    "no explanation, no punctuation. "
                    "If the slot is not mentioned, output: none"
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Render to a plain string using Llama 3.1's built-in chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _norm_pred(text)
