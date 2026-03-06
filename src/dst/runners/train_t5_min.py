import argparse
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


def load_as_hf_dataset(jsonl_path: str, limit: int | None = None) -> Dataset:
    rows = []
    for obj in iter_jsonl(jsonl_path, limit=limit):
        pe = make_prompt_example(
            obj["dialogue_context"],
            obj["slot_name"],
            obj["slot_description"],
            obj["target_value"],
        )
        rows.append({"input_text": pe.input_text, "target_text": pe.target_text})
    return Dataset.from_list(rows)


def make_preprocess_fn(tok):
    pad_id = tok.pad_token_id

    def preprocess(batch):
        # Tokenize inputs
        model_inputs = tok(
            batch["input_text"],
            truncation=True,
            max_length=512,
        )

        # Tokenize targets (new Transformers API)
        labels = tok(
            text_target=batch["target_text"],
            truncation=True,
            max_length=32,
        )["input_ids"]

        # Mask padding tokens so they don't contribute to loss
        labels = [[(t if t != pad_id else -100) for t in seq] for seq in labels]
        model_inputs["labels"] = labels

        return model_inputs

    return preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data_unified/multiwoz24/val.jsonl")
    ap.add_argument("--out_dir", default="runs/t5_min")
    ap.add_argument("--model_name", default="google/flan-t5-base")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--steps", type=int, default=80)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    ds = load_as_hf_dataset(args.train_path, limit=args.limit)

    print("Loading model/tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    preprocess = make_preprocess_fn(tok)
    ds_tok = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    fp16 = torch.cuda.is_available()

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_steps=args.steps,
        logging_steps=10,
        logging_strategy="steps",
        logging_first_step=True,
        save_steps=40,
        save_total_limit=2,
        report_to=[],
        fp16=fp16,
        optim="adamw_torch",
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