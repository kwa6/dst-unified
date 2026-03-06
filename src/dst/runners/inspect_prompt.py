import argparse
from dst.data.jsonl_dataset import load_jsonl
from dst.models.prompting import make_prompt_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    ex = load_jsonl(args.path, limit=args.index + 1)[args.index]
    pe = make_prompt_example(ex.dialogue_context, ex.slot_name, ex.slot_description, ex.target_value)

    print("=== INPUT ===")
    print(pe.input_text)
    print("\n=== TARGET ===")
    print(pe.target_text)


if __name__ == "__main__":
    main()