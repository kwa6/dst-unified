import argparse

from dst.data.jsonl_dataset import load_jsonl
from dst.models.prompting import make_prompt_example
from dst.models.t5_dst import T5DSTModel


def norm(v: str) -> str:
    v = (v or "").strip().lower()
    if v in {"", "none", "not mentioned", "not given"}:
        return "none"
    return v


def score(model: T5DSTModel, data):
    correct_all = 0
    correct_non_none = 0
    total_non_none = 0

    for ex in data:
        pe = make_prompt_example(ex.dialogue_context, ex.slot_name, ex.target_value, slot_description=ex.slot_description)
        pred = norm(model.predict(pe.input_text))
        gold = norm(ex.target_value)

        if pred == gold:
            correct_all += 1

        if gold != "none":
            total_non_none += 1
            if pred == gold:
                correct_non_none += 1

    acc_all = correct_all / len(data)
    acc_non_none = (correct_non_none / total_non_none) if total_non_none > 0 else 0.0
    return {
        "all": (correct_all, len(data), acc_all),
        "non_none": (correct_non_none, total_non_none, acc_non_none),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data_unified/multiwoz24/val.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ft_path", default="runs/t5_min_v3/final")
    args = ap.parse_args()

    data = load_jsonl(args.path, limit=args.limit)

    base = T5DSTModel("google/flan-t5-base")
    base_scores = score(base, data)

    ft = T5DSTModel(args.ft_path)
    ft_scores = score(ft, data)

    print("BASE:", base_scores)
    print("FT  :", ft_scores)


if __name__ == "__main__":
    main()