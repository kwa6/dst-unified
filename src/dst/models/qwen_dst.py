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


class QwenDSTModel:
    """
    Qwen2.5-7B-Instruct wrapper for DST slot filling.

    Uses the model's chat template so the prompt is passed as a
    user message and the generated reply is extracted as the answer.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading model:", self.model_name)
        print("Device:", self.device)

        local = _is_local_model_path(self.model_name)
        load_kwargs: dict = dict(
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
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

        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()

    def predict(self, prompt: str, max_new_tokens: int = 20) -> str:
        """
        Run slot-value prediction.

        The prompt (from prompting.format_slot_prompt) is wrapped in the
        Qwen2.5 chat template so the model receives it as a user message.
        Only the newly generated tokens are returned.
        """
        messages = [{"role": "user", "content": prompt}]

        # apply_chat_template adds the special tokens Qwen2.5 expects
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy for deterministic DST
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the input)
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _norm_pred(text)
