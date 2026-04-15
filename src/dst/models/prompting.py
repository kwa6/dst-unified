from dataclasses import dataclass


@dataclass(frozen=True)
class PromptExample:
    input_text: str
    target_text: str


def format_slot_prompt(
    dialogue_context: str,
    slot_name: str,
    slot_description: str | None = None,
    use_desc: bool = False,
    value_examples: list[str] | None = None,
    use_examples: bool = False,
) -> str:
    """
    Minimal, model-agnostic prompt with optional descriptions and examples.
    Works for T5 (seq2seq) and later Llama (causal) with small changes.
    
    Args:
        dialogue_context: The dialogue history
        slot_name: The slot to extract
        slot_description: Optional description of what the slot represents
        use_desc: Whether to include description in the prompt (default: False)
        value_examples: Optional list of example values for this slot
        use_examples: Whether to include examples in the prompt (default: False)
    """
    prompt = (
        "Dialogue:\n"
        f"{dialogue_context}\n\n"
        "Slot:\n"
        f"{slot_name}\n\n"
    )
    
    if use_desc and slot_description:
        prompt += (
            "Description:\n"
            f"{slot_description}\n\n"
        )
    
    if use_examples and value_examples:
        prompt += (
            "Examples:\n"
            f"{', '.join(value_examples)}\n\n"
        )
    
    prompt += (
        "Instructions:\n"
        "- Extract the exact slot value as mentioned or implied in the dialogue.\n"
        "- Use only the most recent value if it changes across turns.\n"
        "- If the user doesn't care about the value, answer 'dontcare'.\n"
        "- If the slot is not mentioned at all, answer 'none'.\n"
        "- Reply with the slot value only — no explanation.\n"
    )
    
    return prompt


def make_prompt_example(
    dialogue_context: str,
    slot_name: str,
    target_value: str,
    slot_description: str | None = None,
    use_desc: bool = False,
    value_examples: list[str] | None = None,
    use_examples: bool = False,
) -> PromptExample:
    """
    Create a prompt example with optional description and examples inclusion.
    
    Args:
        dialogue_context: The dialogue history
        slot_name: The slot to extract
        target_value: The ground truth value for training
        slot_description: Optional description of what the slot represents
        use_desc: Whether to include description in the prompt (default: False)
        value_examples: Optional list of example values for this slot
        use_examples: Whether to include examples in the prompt (default: False)
    """
    return PromptExample(
        input_text=format_slot_prompt(dialogue_context, slot_name, slot_description, use_desc, value_examples, use_examples),
        target_text=target_value,
    )