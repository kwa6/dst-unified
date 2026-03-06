from dst.data.jsonl_dataset import load_jsonl
from dst.models.prompting import make_prompt_example
from dst.models.t5_dst import T5DSTModel


def main():

    dataset = load_jsonl("data_unified/multiwoz24/val.jsonl", limit=3)

    model = T5DSTModel()

    for i, ex in enumerate(dataset):

        prompt = make_prompt_example(
            ex.dialogue_context,
            ex.slot_name,
            ex.slot_description,
            ex.target_value
        )

        pred = model.predict(prompt.input_text)

        print("\n========================")
        print("Slot:", ex.slot_name)
        print("Target:", ex.target_value)
        print("Prediction:", pred)


if __name__ == "__main__":
    main()