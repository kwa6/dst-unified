from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress CUDA initialization warnings (common with older drivers on UCloud)
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")


def _is_local_model_path(name_or_path: str) -> bool:
    p = Path(name_or_path).resolve()
    return p.exists() and p.is_dir()


def _looks_like_local_path(name_or_path: str) -> bool:
    p = Path(name_or_path)
    if p.is_absolute():
        return True
    if name_or_path.startswith(("./", "../", "~/")):
        return True
    # HF repo IDs allow at most one slash: namespace/repo_name
    if name_or_path.count("/") >= 2:
        return True
    return False


def _resolve_model_path(name_or_path: str) -> str:
    """Convert relative paths to absolute paths for robustness."""
    p = Path(name_or_path)
    if not p.is_absolute():
        p = p.resolve()
    return str(p)


def _is_lora_checkpoint(path: str) -> bool:
    """Check if a local directory contains a LoRA adapter (not full weights)."""
    p = Path(path)
    return p.is_dir() and (p / "adapter_config.json").exists()


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

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        load_in_4bit: bool = False,
        force_cuda: bool = False,
        for_training: bool = False,
        local_rank: int | None = None,
        use_device_map: bool | None = None,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.for_training = for_training
        
        # Determine device: explicit > force_cuda > auto-detect
        if device:
            self.device = device
        elif local_rank is not None:
            self.device = f"cuda:{local_rank}"
        elif force_cuda:
            self.device = "cuda"
        else:
            # Try to detect CUDA, but handle old driver gracefully
            try:
                cuda_available = torch.cuda.is_available()
            except Exception:
                # CUDA initialization might fail with very old drivers
                cuda_available = False
            self.device = "cuda" if cuda_available else "cpu"

        # Enforce GPU requirement: fail fast if CUDA is not available
        if self.device == "cpu":
            raise RuntimeError(
                "ERROR: GPU (CUDA) is required for training but is not available.\n"
                "  - Check that you have a GPU device\n"
                "  - Verify CUDA drivers are installed: nvidia-smi\n"
                "  - Ensure PyTorch was installed with CUDA support\n"
                "  - CPU training is not supported for performance reasons"
            )

        print("Loading model:", self.model_name)
        print("Device:", self.device)

        # Detect and resolve local model paths
        is_local = _is_local_model_path(self.model_name) or _looks_like_local_path(self.model_name)
        if is_local and not self.model_name.startswith(("meta-llama/", "microsoft/", "facebook/")):
            # Resolve relative paths to absolute paths for robustness
            self.model_name = _resolve_model_path(self.model_name)
            print(f"Resolved local path to: {self.model_name}")

        local = _is_local_model_path(self.model_name) or _looks_like_local_path(self.model_name)
        lora  = local and _is_lora_checkpoint(self.model_name)
        if use_device_map is None:
            use_device_map = self.device != "cpu" and local_rank is None

        self.use_device_map = use_device_map

        load_kwargs: dict = dict(
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.use_device_map and self.device != "cpu" else None,
            low_cpu_mem_usage=True,
        )

        if load_in_4bit and self.device != "cpu":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            print("  Loading in 4-bit (QLoRA)")

        if lora:
            # LoRA adapter checkpoint — load base model from adapter_config,
            # then apply adapter weights
            from peft import PeftModel, PeftConfig
            peft_cfg = PeftConfig.from_pretrained(self.model_name)
            base_id = peft_cfg.base_model_name_or_path
            print(f"  LoRA adapter found — loading base model: {base_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_id)
            
            # For LoRA loading, use explicit device mapping to avoid accelerate bugs
            lora_load_kwargs = dict(load_kwargs)
            if self.device != "cpu" and self.use_device_map:
                # Use explicit device mapping instead of "auto" to avoid accelerate.get_balanced_memory bug
                lora_load_kwargs["device_map"] = {"": self.device}
            else:
                lora_load_kwargs.pop("device_map", None)
            
            base_model = AutoModelForCausalLM.from_pretrained(base_id, **lora_load_kwargs)
            self.model = PeftModel.from_pretrained(base_model, self.model_name)
            
            # CRITICAL: Only merge if NOT loading for further training
            # If we merge adapters into base model, we can't train new adapters on top.
            # For training (e.g., stage 2), keep the PeftModel wrapper intact.
            if not load_in_4bit and not self.for_training:
                self.model = self.model.merge_and_unload()
            elif self.for_training:
                print("  Keeping LoRA adapters separate (for further training)")
        elif local:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, local_files_only=True, **load_kwargs
                )
            except OSError as e:
                raise OSError(
                    f"Local model path not found or incomplete: {self.model_name}. "
                    "Expected a local folder containing tokenizer/model files or LoRA adapter files."
                ) from e
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

        if self.device != "cpu" and not self.use_device_map:
            self.model.to(self.device)
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
        
        If the model already has LoRA adapters (from a checkpoint), this skips
        attachment and just enables gradient checkpointing and training mode.
        """
        try:
            from peft import get_peft_model, LoraConfig, TaskType, PeftModel
        except ImportError as e:
            raise ImportError(
                "peft is required for LoRA training. "
                "Install it with:  pip install peft"
            ) from e

        # Check if model is already wrapped with LoRA adapters
        if isinstance(self.model, PeftModel):
            print("  LoRA adapters already attached (from checkpoint)")
            # CRITICAL FIX for QLoRA + loaded checkpoints:
            # When loading a 4-bit checkpoint, adapter modules exist but aren't marked as trainable.
            # We need to explicitly enable gradients on adapter parameters.
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
        else:
            # Attach new adapters
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                # target all attention projection layers (works for Llama 2 & 3)
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_cfg)
        
        # Gradient checkpointing reduces activation memory at the cost of speed
        self.model.gradient_checkpointing_enable()
        
        # Set to training mode
        self.model.train()
        
        # Print AFTER train() to see actual trainable params
        self.model.print_trainable_parameters()

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
