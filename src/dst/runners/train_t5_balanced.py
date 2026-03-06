import argparse
import random
from pathlib import Path

import torch
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
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

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

    preprocess = make_preprocess_fn(tok)
    ds_tok = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)
    # fp16 = torch.cuda.is_available()
    # fp16 often causes NaNs in this environment; keep training in fp32 for stability
    fp16 = False
    
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=args.steps,
        logging_steps=50,
        logging_strategy="steps",
        logging_first_step=True,
        report_to=[],
        fp16=False,
        max_grad_norm=1.0,
        save_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    print("Training...")
    trainer.train()

    print("Saving final model...")
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    print("Done. Saved to:", final_dir)


if __name__ == "__main__":
    main()