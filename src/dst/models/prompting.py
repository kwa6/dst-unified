from dataclasses import dataclass


@dataclass(frozen=True)
class PromptExample:
    input_text: str
    target_text: str


def format_slot_prompt(dialogue_context: str, slot_name: str, slot_description: str) -> str:
    """
    Minimal, model-agnostic prompt.
    Works for T5 (seq2seq) and later Llama (causal) with small changes.
    """
    # Keep it simple and consistent
    return (
        "Dialogue:\n"
        f"{dialogue_context}\n\n"
        "Slot:\n"
        f"{slot_name}\n\n"
        "Description:\n"
        f"{slot_description}\n\n"
        "Answer with the slot value. If it is not provided, answer 'none'."
    )


def make_prompt_example(dialogue_context: str, slot_name: str, slot_description: str, target_value: str) -> PromptExample:
    return PromptExample(
        input_text=format_slot_prompt(dialogue_context, slot_name, slot_description),
        target_text=target_value,
    )