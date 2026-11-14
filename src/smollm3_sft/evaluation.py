from typing import Any

import torch


def generate_response(model, tokenizer, prompt_messages, generation_config):
    """Generate one response from a list of chat messages."""
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = model.get_input_embeddings().weight.device
    inputs = inputs.to(device)

    generation_kwargs = {
        "max_new_tokens": int(generation_config["max_new_tokens"]),
        "do_sample": bool(generation_config["do_sample"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(generation_config["temperature"])
        generation_kwargs["top_p"] = float(generation_config["top_p"])

    model.eval()
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    generated_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def _distinct_n(texts: list[str], n: int = 2) -> float:
    ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams.extend(tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def _rouge_l(predictions: list[str], references: list[str]) -> float | None:
    pairs = [
        (prediction, reference)
        for prediction, reference in zip(predictions, references)
        if reference.strip()
    ]
    if not pairs:
        return None

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in pairs
    ]
    return sum(scores) / len(scores)


def run_generation_evaluation(
    model,
    tokenizer,
    test_examples,
    generation_config: dict[str, Any],
    compute_rouge_l: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Generate responses for held-out prompts and calculate lightweight metrics."""
    number_of_samples = min(
        int(generation_config["num_samples"]),
        len(test_examples),
    )
    rows = []
    predictions = []
    references = []

    for example in test_examples.select(range(number_of_samples)):
        prompt_messages = example["prompt_messages"]
        if not prompt_messages:
            continue

        prediction = generate_response(
            model,
            tokenizer,
            prompt_messages,
            generation_config,
        )
        reference = example["reference"] or ""
        predictions.append(prediction)
        references.append(reference)
        rows.append({
            "prompt_messages": prompt_messages,
            "reference": reference,
            "generated_response": prediction,
        })

    metrics = {
        "generation_samples": float(len(rows)),
        "generation_distinct_2": _distinct_n(predictions, n=2),
        "generation_mean_response_words": (
            sum(len(text.split()) for text in predictions) / len(predictions)
            if predictions else 0.0
        ),
    }
    if compute_rouge_l:
        rouge_l = _rouge_l(predictions, references)
        if rouge_l is not None:
            metrics["generation_rouge_l"] = rouge_l

    return rows, metrics
