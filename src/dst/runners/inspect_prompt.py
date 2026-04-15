import argparse
from dst.data.jsonl_dataset import load_jsonl
from dst.models.prompting import make_prompt_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--use_slot_description", action="store_true", help="Include slot description in prompt")
    ap.add_argument("--use_value_examples", action="store_true", help="Include value examples in prompt")
    args = ap.parse_args()

    ex = load_jsonl(args.path, limit=args.index + 1)[args.index]
    pe = make_prompt_example(
        ex.dialogue_context,
        ex.slot_name,
        ex.target_value,
        slot_description=ex.slot_description,
        use_desc=args.use_slot_description,
        value_examples=ex.value_examples,
        use_examples=args.use_value_examples,
    )

    print("=== INPUT ===")
    print(pe.input_text)
    print("\n=== TARGET ===")
    print(pe.target_text)


if __name__ == "__main__":
    main()