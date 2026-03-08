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
    return (
        "Dialogue:\n"
        f"{dialogue_context}\n\n"
        "Slot:\n"
        f"{slot_name}\n\n"
        "Description:\n"
        f"{slot_description}\n\n"
        "Instructions:\n"
        "- Extract the exact slot value as mentioned or implied in the dialogue.\n"
        "- Use only the most recent value if it changes across turns.\n"
        "- If the user doesn't care about the value, answer 'dontcare'.\n"
        "- If the slot is not mentioned at all, answer 'none'.\n"
        "- Reply with the slot value only — no explanation.\n"
    )


def make_prompt_example(dialogue_context: str, slot_name: str, slot_description: str, target_value: str) -> PromptExample:
    return PromptExample(
        input_text=format_slot_prompt(dialogue_context, slot_name, slot_description),
        target_text=target_value,
    )