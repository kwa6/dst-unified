import argparse
import random
import sys
from pathlib import Path

import torch
torch.backends.cudnn.benchmark = True
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from dst.data.jsonl_dataset import iter_jsonl
from dst.models.prompting import make_prompt_example


def norm(v: str) -> str:
    v = (v or "").strip().lower()
    if v in {"", "none", "not mentioned", "not given"}:
        return "none"
    return v


def load_rows(jsonl_path: str, limit: int | None = None):
    rows = []
    for obj in iter_jsonl(jsonl_path, limit=limit):
        pe = make_prompt_example(
            obj["dialogue_context"],
            obj["slot_name"],
            obj["slot_description"],
            obj["target_value"],
        )
        rows.append({"input_text": pe.input_text, "target_text": norm(pe.target_text)})
    return rows


def make_balanced_dataset(rows, total_examples: int, seed: int = 13) -> Dataset:
    rng = random.Random(seed)
    non_none = [r for r in rows if r["target_text"] != "none"]
    none = [r for r in rows if r["target_text"] == "none"]

    if len(non_none) == 0:
        raise ValueError("No non-none examples found. Expand schema or increase dialogue limit.")
    if len(none) == 0:
        raise ValueError("No none examples found.")

    half = total_examples // 2
    # sample with replacement if needed
    samp_non_none = [rng.choice(non_none) for _ in range(half)]
    samp_none = [rng.choice(none) for _ in range(total_examples - half)]

    balanced = samp_non_none + samp_none
    rng.shuffle(balanced)
    return Dataset.from_list(balanced)


def make_preprocess_fn(tok):
    pad_id = tok.pad_token_id

    def preprocess(batch):
        model_inputs = tok(batch["input_text"], truncation=True, max_length=512)
        labels = tok(text_target=batch["target_text"], truncation=True, max_length=32)["input_ids"]
        labels = [[(t if t != pad_id else -100) for t in seq] for seq in labels]
        model_inputs["labels"] = labels
        return model_inputs

    return preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data_unified/multiwoz24/val.jsonl")
    ap.add_argument("--out_dir", default="runs/t5_balanced")
    ap.add_argument("--model_name", default="google/flan-t5-base")
    ap.add_argument("--limit_read", type=int, default=None, help="How many JSONL lines to read before balancing")
    ap.add_argument("--total_examples", type=int, default=400, help="Size of balanced training set")
    ap.add_argument("--steps", type=int, default=None, help="(Deprecated: use num_epochs)")
    ap.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    ap.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps for learning rate scheduler")
    ap.add_argument("--eval_path", default=None, help="JSONL file for evaluation during training")
    ap.add_argument("--seed", type=int, default=13)
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

    print("Loading rows...")
    rows = load_rows(args.train_path, limit=args.limit_read)
    print("Rows loaded:", len(rows))
    print("Non-none:", sum(1 for r in rows if r["target_text"] != "none"))
    print("None:", sum(1 for r in rows if r["target_text"] == "none"))

    print("Building balanced dataset...")
    ds = make_balanced_dataset(rows, total_examples=args.total_examples, seed=args.seed)

    print("Loading model/tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = model.to("cuda")
    print(f"Model loaded and moved to CUDA")

    preprocess = make_preprocess_fn(tok)
    ds_tok = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)
    
    # Build optional eval dataset
    eval_ds = None
    if args.eval_path:
        eval_rows = load_rows(args.eval_path)
        eval_ds = make_balanced_dataset(eval_rows, total_examples=min(200, len(eval_rows)), seed=args.seed + 1)
        preprocess = make_preprocess_fn(tok)
        eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)
        print(f"Eval dataset loaded: {len(eval_ds)} examples")
    
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        logging_steps=20,
        logging_strategy="steps",
        logging_first_step=True,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=50 if eval_ds else None,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True if eval_ds else False,
        report_to=[],
        fp16=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",  # Better than linear decay to 0
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_tok,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print(f"\nTraining Configuration:")
    print(f"  Batch size: 8 × {train_args.gradient_accumulation_steps} (effective = 16)")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Learning rate: {train_args.learning_rate}")
    print(f"  LR scheduler: cosine (maintains min LR throughout training)")
    print(f"  Checkpointing: every {train_args.save_steps} steps")
    print()
    print("Training...")
    trainer.train()

    print("Saving final model...")
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    print("Done. Saved to:", final_dir)


if __name__ == "__main__":
    main()