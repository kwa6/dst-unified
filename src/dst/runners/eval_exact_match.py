from dst.data.jsonl_dataset import load_jsonl
from dst.models.prompting import make_prompt_example
from dst.models.t5_dst import T5DSTModel


def main():
    data = load_jsonl("data_unified/multiwoz24/val.jsonl", limit=20)
    model = T5DSTModel()

    correct = 0
    for ex in data:
        pe = make_prompt_example(ex.dialogue_context, ex.slot_name, ex.slot_description, ex.target_value)
        pred = model.predict(pe.input_text)
        gold = ex.target_value.strip().lower()
        if gold in {"", "none", "not mentioned", "not given"}:
            gold = "none"
        if pred == gold:
            correct += 1

    print(f"Exact match: {correct}/{len(data)} = {correct/len(data):.3f}")


if __name__ == "__main__":
    main()