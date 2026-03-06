import argparse
from dst.data.jsonl_dataset import load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    examples = load_jsonl(args.path, limit=args.limit)
    print(f"Loaded {len(examples)} examples\n")

    for i, ex in enumerate(examples, 1):
        print(f"--- Example {i} ---")
        print("slot_name:", ex.slot_name)
        print("slot_description:", ex.slot_description)
        print("target_value:", ex.target_value)
        print("context:", ex.dialogue_context[:200].replace("\n", "\\n"))
        print()


if __name__ == "__main__":
    main()