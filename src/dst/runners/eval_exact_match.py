import argparse
from dst.data.jsonl_dataset import load_jsonl
from dst.models.prompting import make_prompt_example
from dst.models.t5_dst import T5DSTModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data_unified/multiwoz24/val.jsonl", help="JSONL file to evaluate")
    ap.add_argument("--limit", type=int, default=20, help="Limit number of examples (default: 20)")
    ap.add_argument("--model", default=None, help="Model path (default: T5 base)")
    ap.add_argument("--use_slot_description", action="store_true", help="Include slot descriptions in prompts (default: off)")
    ap.add_argument("--use_value_examples", action="store_true", help="Include value examples in prompts (default: off)")
    args = ap.parse_args()
    
    data = load_jsonl(args.path, limit=args.limit)
    model = T5DSTModel(args.model) if args.model else T5DSTModel()

    correct = 0
    for ex in data:
        pe = make_prompt_example(ex.dialogue_context, ex.slot_name, ex.target_value, slot_description=ex.slot_description, use_desc=args.use_slot_description, value_examples=ex.value_examples, use_examples=args.use_value_examples)
        pred = model.predict(pe.input_text)
        gold = ex.target_value.strip().lower()
        if gold in {"", "none", "not mentioned", "not given"}:
            gold = "none"
        if pred == gold:
            correct += 1

    print(f"Exact match: {correct}/{len(data)} = {correct/len(data):.3f}")


if __name__ == "__main__":
    main()