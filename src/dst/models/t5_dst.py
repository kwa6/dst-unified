from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def _is_local_model_path(name_or_path: str) -> bool:
    p = Path(name_or_path)
    return p.exists() and p.is_dir()


def _norm_pred(text: str) -> str:
    text = (text or "").strip().lower()
    if text in {"", "none", "not mentioned", "not given"}:
        return "none"
    return text


class T5DSTModel:
    def __init__(self, model_name: str = "google/flan-t5-base", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print("Loading model:", model_name)
        print("Device:", self.device)

        local = _is_local_model_path(model_name)

        if local:
            # Force local loading; do not fall back to hub
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, prompt: str, max_new_tokens: int = 10) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return _norm_pred(text)