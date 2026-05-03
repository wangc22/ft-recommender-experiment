"""Prompt templates and formatters for classification + generation tasks."""

CLASSIFICATION_TEMPLATE = (
    "Choose one label from the following list:\n"
    "{labels}\n"
    "\n"
    "Input:\n"
    "{input_text}\n"
    "\n"
    "Answer with only the label."
)

GENERATION_SYSTEM = (
    "Answer as a helpful customer support assistant. "
    "Be concise, polite, policy-consistent, and ask follow-up questions "
    "when key information is missing."
)


def format_classification_prompt(input_text: str, labels: list[str]) -> str:
    label_block = "[" + ", ".join(labels) + "]"
    return CLASSIFICATION_TEMPLATE.format(labels=label_block, input_text=input_text)


def format_classification_for_sft(record: dict, labels: list[str]) -> dict:
    """record → {"prompt": ..., "completion": ...} for trl prompt-completion mode.

    The trailing "\n" lives on the prompt (not completion) so that tokenized
    prompt and tokenized prompt+completion share the same prefix bytes — this
    avoids trl's "Mismatch between tokenized prompt..." warning.
    """
    prompt = format_classification_prompt(record["input"], labels) + "\n"
    return {"prompt": prompt, "completion": record["label"]}


def format_generation_for_sft(record: dict) -> dict:
    """record: {"messages": [...]} → {"messages": [...]} for trl conversational mode."""
    messages = record["messages"]
    if messages and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": GENERATION_SYSTEM}] + messages
    return {"messages": messages}


def format_generation_prompt_only(user_query: str, tokenizer) -> str:
    """Build inference-time prompt (system + user, no assistant content)."""
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM},
        {"role": "user", "content": user_query},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
