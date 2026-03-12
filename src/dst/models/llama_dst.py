from __future__ import annotations

from pathlib import Path
from typing import List, Dict

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
    Llama Instruct wrapper for DST slot filling.

    Tested with (all use the same chat template):
      - meta-llama/Llama-3.3-70B-Instruct   (~80GB VRAM, 4×A100)
      - meta-llama/Llama-3.1-8B-Instruct    (~16GB VRAM, 1×A100)
      - meta-llama/Llama-3.2-3B-Instruct    (~8GB  VRAM, 1×A100)
      - meta-llama/Llama-3.2-1B-Instruct    (~4GB  VRAM, CPU ok)

    Requires HuggingFace access:
      https://huggingface.co/meta-llama
    """

    DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

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

        # Clean up generation config to avoid warnings:
        # - remove temperature/top_p (not valid with do_sample=False)
        # - remove max_length (we use max_new_tokens instead)
        gc = self.model.generation_config
        if hasattr(gc, "temperature"):
            gc.temperature = None
        if hasattr(gc, "top_p"):
            gc.top_p = None
        if hasattr(gc, "max_length"):
            gc.max_length = None

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

    # ------------------------------------------------------------------
    # LoRA fine-tuning
    # ------------------------------------------------------------------

    def prepare_for_training(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        """
        Attach LoRA adapters to the model so it can be fine-tuned efficiently.
        Requires the `peft` package  (pip install peft).

        Call this once before passing the model to a Trainer.
        Puts the model back into training mode.
        """
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError as e:
            raise ImportError(
                "peft is required for LoRA training. "
                "Install it with:  pip install peft"
            ) from e

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # target the attention projection layers (works for Llama 2 & 3)
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()
        self.model.train()

    def build_training_batch(
        self, examples: List[Dict[str, str]], max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a list of {"input_text": ..., "target_text": ...} dicts into
        token tensors for causal-LM training.

        The format is:
            <prompt> <target_value> <eos>

        Labels are -100 for the prompt tokens (masked out) so the loss is
        computed only on the target tokens.
        """
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for ex in examples:
            # Build the full chat-formatted prompt string
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
                {"role": "user", "content": ex["input_text"]},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            target_str = ex["target_text"] + self.tokenizer.eos_token

            prompt_ids = self.tokenizer(
                prompt_str, add_special_tokens=False
            )["input_ids"]
            target_ids = self.tokenizer(
                target_str, add_special_tokens=False
            )["input_ids"]

            full_ids = (prompt_ids + target_ids)[:max_length]
            # mask prompt tokens from the loss
            labels = ([-100] * len(prompt_ids) + target_ids)[: len(full_ids)]

            input_ids_list.append(full_ids)
            attention_mask_list.append([1] * len(full_ids))
            labels_list.append(labels)

        # left-pad to the same length
        max_len = max(len(x) for x in input_ids_list)
        pad_id = self.tokenizer.pad_token_id

        def pad(seq, pad_val, length):
            return [pad_val] * (length - len(seq)) + seq

        input_ids  = torch.tensor([pad(x, pad_id,  max_len) for x in input_ids_list])
        attn_mask  = torch.tensor([pad(x, 0,       max_len) for x in attention_mask_list])
        labels_t   = torch.tensor([pad(x, -100,    max_len) for x in labels_list])

        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels_t}
