"""
Fine-tune a Llama Instruct model on the DST task using LoRA.

LoRA (Low-Rank Adaptation) freezes the base model weights and only trains
small adapter layers — this makes it feasible to fine-tune a 7B model on a
single A100 (40GB), or a 1B model on CPU (slow but possible).

Requirements:
    pip install peft

Usage (UCloud / Linux):
    export PYTHONPATH=src
    python -m dst.runners.train_llama \\
        --train_path data_unified/multiwoz24/train.jsonl \\
        --model      meta-llama/Llama-2-7b-chat-hf \\
        --out_dir    runs/llama_mwoz_v1 \\
        --steps      500

Usage (Windows PowerShell):
    $env:PYTHONPATH = "src"
    python -m dst.runners.train_llama `
        --train_path data_unified/multiwoz24/train.jsonl `
        --model      meta-llama/Llama-2-7b-chat-hf `
        --out_dir    runs/llama_mwoz_v1 `
        --steps      500
"""
import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

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
    ap = argparse.ArgumentParser(description="Fine-tune Llama for DST with LoRA.")
    ap.add_argument("--train_path",      default="data_unified/multiwoz24/train.jsonl")
    ap.add_argument("--model",           default="meta-llama/Llama-2-7b-chat-hf",
                    help="HF model ID or local path")
    ap.add_argument("--out_dir",         default="runs/llama_mwoz_v1")
    ap.add_argument("--limit_read",      type=int, default=None,
                    help="Cap number of JSONL lines read (debug)")
    ap.add_argument("--total_examples",  type=int, default=2000,
                    help="Balanced training set size")
    ap.add_argument("--steps",           type=int, default=500)
    ap.add_argument("--batch_size",      type=int, default=4)
    ap.add_argument("--grad_accum",      type=int, default=4,
                    help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    ap.add_argument("--lr",              type=float, default=2e-4)
    ap.add_argument("--lora_r",          type=int, default=16)
    ap.add_argument("--lora_alpha",      type=int, default=32)
    ap.add_argument("--max_length",      type=int, default=512)
    ap.add_argument("--seed",            type=int, default=13)
    args = ap.parse_args()

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
    llama = LlamaDSTModel(args.model)

    # 3) Attach LoRA adapters — only adapter weights will be trained
    llama.prepare_for_training(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    # 4) Build dataset
    ds = LlamaDSTDataset(balanced, llama, max_length=args.max_length)

    # 5) Training arguments
    fp16 = llama.device != "cpu" and torch.cuda.is_available()
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.steps,
        logging_steps=50,
        logging_strategy="steps",
        logging_first_step=True,
        report_to=[],
        fp16=fp16,
        max_grad_norm=1.0,
        save_strategy="no",
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
        processing_class=llama.tokenizer,
    )

    print("\nStarting LoRA fine-tuning...")
    print(f"  Steps:          {args.steps}")
    print(f"  Batch size:     {args.batch_size} × {args.grad_accum} accum = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  Learning rate:  {args.lr}")
    print(f"  LoRA r:         {args.lora_r}  alpha: {args.lora_alpha}")
    print(f"  Device:         {llama.device}  fp16: {fp16}\n")

    trainer.train()

    # 7) Save — saves LoRA adapter weights + tokenizer
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    llama.tokenizer.save_pretrained(str(final_dir))
    print("\nDone. Saved to:", final_dir)
    print("To evaluate, run:")
    print(f"  python -m dst.runners.eval_jga_llama --model {final_dir} "
          f"--path data_unified/multiwoz24/test.jsonl")


if __name__ == "__main__":
    main()
