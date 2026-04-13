"""
Fine-tune a Llama Instruct model on the DST task using LoRA (single-stage or two-stage).

LoRA (Low-Rank Adaptation) freezes the base model weights and only trains
small adapter layers — this makes it feasible to fine-tune a 7B model on a
single A100 (40GB), or a 1B model on CPU (slow but possible).

Requirements:
    pip install peft

Usage (Single-stage: UCloud / Linux):
    export PYTHONPATH=src
    python -m dst.runners.train_llama \\
        --train_path data_unified/multiwoz24/train.jsonl \\
        --model      meta-llama/Llama-2-7b-chat-hf \\
        --out_dir    runs/llama_mwoz_v1 \\
        --steps      500

Usage (Two-stage: Stage 1 - augmented data):
    python -m dst.runners.train_llama \\
        --train_path data_unified/luas/train.jsonl \\
        --stage 1 \\
        --model      meta-llama/Llama-2-7b-chat-hf \\
        --out_dir    runs/llama_stage1_luas \\
        --steps      500

Usage (Two-stage: Stage 2 - fine-tune on real data):
    python -m dst.runners.train_llama \\
        --train_path data_unified/multiwoz24/train.jsonl \\
        --stage 2 \\
        --checkpoint runs/llama_stage1_luas/final \\
        --out_dir    runs/llama_stage2_mwoz_final \\
        --steps      300 \\
        --lr_stage2  5e-5
"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from dst.data.jsonl_dataset import iter_jsonl
from dst.models.prompting import make_prompt_example
from dst.models.llama_dst import LlamaDSTModel


def norm(v: str) -> str:
    v = (v or "").strip().lower()
    if v in {"", "none", "not mentioned", "not given"}:
        return "none"
    return v


def load_rows(jsonl_path: str, limit: int | None = None) -> list[dict]:
    rows = []
    for obj in iter_jsonl(jsonl_path, limit=limit):
        pe = make_prompt_example(
            obj["dialogue_context"],
            obj["slot_name"],
            obj["slot_description"],
            obj["target_value"],
        )
        rows.append({
            "input_text":  pe.input_text,
            "target_text": norm(pe.target_text),
        })
    return rows


def make_balanced_dataset(rows: list[dict], total_examples: int, seed: int = 13) -> list[dict]:
    """50 % none / 50 % non-none, same strategy as train_t5_balanced."""
    rng = random.Random(seed)
    non_none = [r for r in rows if r["target_text"] != "none"]
    none     = [r for r in rows if r["target_text"] == "none"]

    if not non_none:
        raise ValueError("No non-none examples found in training data.")
    if not none:
        raise ValueError("No none examples found in training data.")

    half = total_examples // 2
    balanced = (
        [rng.choice(non_none) for _ in range(half)] +
        [rng.choice(none)     for _ in range(total_examples - half)]
    )
    rng.shuffle(balanced)
    return balanced


class CausalLMLeftPadCollator:
    """Left-pad input_ids, attention_mask, and labels to the longest item in each batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(f["input_ids"].shape[0] for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - f["input_ids"].shape[0]
            input_ids.append(torch.cat([torch.full((pad_len,), self.pad_token_id, dtype=f["input_ids"].dtype), f["input_ids"]]))
            attention_mask.append(torch.cat([torch.zeros(pad_len, dtype=f["attention_mask"].dtype), f["attention_mask"]]))
            labels.append(torch.cat([torch.full((pad_len,), -100, dtype=f["labels"].dtype), f["labels"]]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


class LlamaDSTDataset(torch.utils.data.Dataset):
    """
    Tokenises each example on-the-fly using the same chat template the model
    uses at inference time.  Labels mask out the prompt so the loss is computed
    only on the target (slot value) tokens.
    """

    def __init__(self, rows: list[dict], model: LlamaDSTModel, max_length: int = 512):
        self.rows       = rows
        self.model      = model
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = self.model.build_training_batch(
            [self.rows[idx]], max_length=self.max_length
        )
        return {k: v.squeeze(0) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Llama for DST with LoRA (single or two-stage).")
    ap.add_argument("--train_path",      default="data_unified/multiwoz24/train.jsonl")
    ap.add_argument("--model",           default="meta-llama/Llama-2-7b-chat-hf",
                    help="HF model ID or local path")
    ap.add_argument("--out_dir",         default="runs/llama_mwoz_v1")
    ap.add_argument("--limit_read",      type=int, default=None,
                    help="Cap number of JSONL lines read (debug)")
    ap.add_argument("--total_examples",  type=int, default=8000,
                    help="Balanced training set size")
    ap.add_argument("--steps",           type=int, default=500)
    ap.add_argument("--warmup_steps",    type=int, default=50)
    ap.add_argument("--batch_size",      type=int, default=4)
    ap.add_argument("--grad_accum",      type=int, default=4,
                    help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    ap.add_argument("--lr",              type=float, default=2e-4)
    ap.add_argument("--lora_r",          type=int, default=16)
    ap.add_argument("--lora_alpha",      type=int, default=32)
    ap.add_argument("--max_length",      type=int, default=512)
    ap.add_argument("--load_in_4bit",    action="store_true",
                    help="Load model in 4-bit quantization (QLoRA, saves VRAM)")
    ap.add_argument("--eval_path",       default=None,
                    help="JSONL validation file for eval during training")
    ap.add_argument("--eval_examples",   type=int, default=500,
                    help="Number of balanced eval examples")
    ap.add_argument("--seed",            type=int, default=13)
    
    # Two-stage training arguments
    ap.add_argument("--stage",           type=int, default=1,
                    help="Training stage: 1=augmented data (LUAS/D0T), 2=real data (MultiWOZ)")
    ap.add_argument("--checkpoint",      default=None,
                    help="Checkpoint path (for stage 2: load from stage 1)")
    ap.add_argument("--lr_stage2",       type=float, default=5e-5)
    ap.add_argument("--warmup_steps_stage2", type=int, default=100)
    args = ap.parse_args()

    # GPU requirement check: fail fast if CUDA is not available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ERROR: GPU (CUDA) is required for training but is not available.\n"
            "  - Check that you have a GPU device\n"
            "  - Verify CUDA drivers are installed: nvidia-smi\n"
            "  - Ensure PyTorch was installed with CUDA support"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & balance data
    print("Loading rows from:", args.train_path)
    rows = load_rows(args.train_path, limit=args.limit_read)
    print(f"  Total rows:   {len(rows)}")
    print(f"  Non-none:     {sum(1 for r in rows if r['target_text'] != 'none')}")
    print(f"  None:         {sum(1 for r in rows if r['target_text'] == 'none')}")

    balanced = make_balanced_dataset(rows, args.total_examples, seed=args.seed)
    print(f"  Balanced set: {len(balanced)}")

    # 2) Load base model (inference mode first so LoRA attaches cleanly)
    if args.stage == 2 and not args.checkpoint:
        raise ValueError("Stage 2 requires --checkpoint")
    
    if args.stage == 2:
        print(f"\n[STAGE {args.stage}] Loading checkpoint from previous stage:")
        print(f"  Checkpoint: {args.checkpoint}")
        llama = LlamaDSTModel(args.checkpoint, load_in_4bit=args.load_in_4bit)
        current_warmup = args.warmup_steps_stage2
        current_lr = args.lr_stage2
        stage_desc = "Stage 2 (Fine-tuning on Real Data - MultiWOZ)"
    else:
        print(f"\n[STAGE {args.stage}] Loading fresh model:")
        print(f"  Model: {args.model}")
        llama = LlamaDSTModel(args.model, load_in_4bit=args.load_in_4bit)
        current_warmup = args.warmup_steps
        current_lr = args.lr
        stage_desc = "Stage 1 (Training on Augmented Data - LUAS/D0T)"

    # 3) Attach LoRA adapters — only adapter weights will be trained
    llama.prepare_for_training(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    # 4) Build dataset
    ds = LlamaDSTDataset(balanced, llama, max_length=args.max_length)

    # 4b) Optional eval dataset
    eval_ds = None
    if args.eval_path:
        eval_rows = load_rows(args.eval_path, limit=args.limit_read)
        eval_balanced = make_balanced_dataset(eval_rows, args.eval_examples, seed=args.seed + 1)
        eval_ds = LlamaDSTDataset(eval_balanced, llama, max_length=args.max_length)
        print(f"  Eval set:     {len(eval_balanced)}")

    # 5) Training arguments
    fp16 = llama.device != "cpu" and torch.cuda.is_available()
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=current_lr,
        warmup_steps=current_warmup,
        max_steps=args.steps,
        logging_steps=20,  # More frequent logging for better real-time feedback
        logging_strategy="steps",
        logging_first_step=True,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=100 if eval_ds else None,
        save_strategy="no",  # Don't save intermediate checkpoints to save disk space
        load_best_model_at_end=False,
        report_to=[],
        fp16=fp16,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",  # Smoother LR decay
        dataloader_num_workers=0,
        optim="adamw_torch",
        seed=args.seed,
        remove_unused_columns=False,
    )

    # 6) Train
    trainer = Trainer(
        model=llama.model,
        args=train_args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=CausalLMLeftPadCollator(llama.tokenizer.pad_token_id),
    )

    print(f"\n{stage_desc}")
    print(f"  Steps:          {args.steps}")
    print(f"  Batch size:     {args.batch_size} × {args.grad_accum} accum = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  Warmup steps:   {current_warmup}")
    print(f"  Learning rate:  {current_lr}")
    print(f"  LR scheduler:   cosine (maintains min LR throughout training)")
    print(f"  LoRA r:         {args.lora_r}  alpha: {args.lora_alpha}")
    print(f"  Device:         {llama.device}  fp16: {fp16}")
    print(f"  Checkpointing:  disabled (saves to 'final' directory only)")
    print()

    trainer.train()

    # 7) Save — saves LoRA adapter weights + tokenizer
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    llama.tokenizer.save_pretrained(str(final_dir))
    
    # Verify adapter was saved correctly
    adapter_model_path = final_dir / "adapter_model.bin"
    if not adapter_model_path.exists():
        print("\n⚠️  WARNING: adapter_model.bin not found!")
        print("   The checkpoint may be incomplete or corrupted.")
    else:
        size_mb = adapter_model_path.stat().st_size / (1024**1024)
        print(f"\n✓ Adapter saved successfully: {size_mb:.1f}MB")
    
    # Fix adapter_config.json to include base_model_name_or_path
    # This is required for stage 2 to load the checkpoint properly
    adapter_config_path = final_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r") as f:
            adapter_cfg = json.load(f)
        
        # Determine the base model (what we started with)
        base_model_name = None
        
        if args.checkpoint:
            # Stage 2: try to get base model from checkpoint, else use llama's model_name
            try:
                from peft import PeftConfig
                peft_cfg = PeftConfig.from_pretrained(args.checkpoint)
                base_model_name = peft_cfg.base_model_name_or_path
            except Exception:
                pass
        
        # Fallback: use the model name from LlamaDSTModel (it resolves HF IDs)
        if not base_model_name:
            base_model_name = llama.model_name
        
        if base_model_name:
            adapter_cfg["base_model_name_or_path"] = base_model_name
            with open(adapter_config_path, "w") as f:
                json.dump(adapter_cfg, f, indent=2)
            print(f"  Updated adapter_config.json: base_model_name_or_path = {base_model_name}")
        else:
            print("  WARNING: Could not determine base_model_name_or_path")
    
    print("\nDone. Saved to:", final_dir)
    
    if args.stage == 1:
        print("\n" + "="*80)
        print("STAGE 1 COMPLETE. Ready for Stage 2 (fine-tune on MultiWOZ real data):")
        print("="*80)
        print(f"python -m dst.runners.train_llama \\")
        print(f"    --train_path data_unified/multiwoz24/train.jsonl \\")
        print(f"    --stage 2 \\")
        print(f"    --checkpoint {final_dir} \\")
        print(f"    --out_dir runs/llama_stage2_mwoz_final \\")
        print(f"    --steps 300")
        print("="*80)
    else:
        print("\nTo evaluate, run:")
        print(f"  python -m dst.runners.eval_jga_llama --model {final_dir} "
              f"--path data_unified/multiwoz24/test.jsonl")


if __name__ == "__main__":
    main()
