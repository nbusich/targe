def _prepare_image(sample):
    image = sample["images"][0]
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    """User-only prompt generation (no system message in context)."""
    text_input = processor.apply_chat_template(
        sample["messages"][1:2],
        add_generation_prompt=True,
    )
    image = _prepare_image(sample)

    model_inputs = processor(
        text=text_input,
        images=[[image]],
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def generate_text_from_sample_selector(
    model,
    processor,
    sample,
    max_new_tokens: int = 20,
    device: str = "cuda",
):
    """
    System+user prompt with guardrailed greedy decoding.

    Suited for exact-match VQA against the selector connector: forces at least one
    token, disables sampling, and applies a small repetition penalty.
    """
    text_input = processor.apply_chat_template(
        sample["messages"][:2],
        add_generation_prompt=True,
    )
    image = _prepare_image(sample)

    model_inputs = processor(
        text=text_input,
        images=[[image]],
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        do_sample=False,
        repetition_penalty=1.1,
    )
    trimmed = [
        out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()
