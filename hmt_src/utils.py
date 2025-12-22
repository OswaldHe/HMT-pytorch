def apply_chat_template_with_fallback(tokenizer, messages):
    """
    Try to use tokenizer.apply_chat_template() if the tokenizer has a valid
    chat_template. If it fails (because the model is a base model without chat
    template), fall back to a manually constructed plain-text prompt.

    Returns:
        input_ids: torch.Tensor of shape [1, T]
    """
    # Attempt to use built-in chat template
    try:
        # If chat_template exists, try to apply it
        if getattr(tokenizer, "chat_template", None) is not None:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
            )
    except (ValueError, AttributeError):
        # If template is missing or invalid, fall through to fallback
        pass

    # Fallback: simple manual prompt formatting
    text_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # simple readable prefix for different roles
        if role == "system":
            prefix = "[SYSTEM]"
        elif role == "user":
            prefix = "[USER]"
        elif role == "assistant":
            prefix = "[ASSISTANT]"
        else:
            prefix = f"[{role.upper()}]"

        text_parts.append(f"{prefix}\n{content}")

    prompt = "\n\n".join(text_parts)

    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded["input_ids"]
