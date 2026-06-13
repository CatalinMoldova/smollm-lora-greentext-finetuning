"""Text generation helpers."""

from __future__ import annotations


def generate_response(model, tokenizer, prompt, max_length=100, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_generation(model, tokenizer, prompts):
    return {p: generate_response(model, tokenizer, p) for p in prompts}
