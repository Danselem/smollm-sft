from typing import Any

from datasets import Dataset, load_dataset


def _normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]


def _format_example(example: dict[str, Any], tokenizer) -> dict[str, Any]:
    if "messages" in example and example["messages"]:
        messages = _normalise_messages(example["messages"])
        assistant_indices = [
            index for index, message in enumerate(messages)
            if message["role"] == "assistant"
        ]
        if assistant_indices:
            final_assistant_index = assistant_indices[-1]
            prompt_messages = messages[:final_assistant_index]
            reference = messages[final_assistant_index]["content"]
        else:
            prompt_messages = messages
            reference = ""
    else:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        prompt_messages = messages[:-1]
        reference = messages[-1]["content"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": text,
        "prompt_messages": prompt_messages,
        "reference": reference if isinstance(reference, str) else str(reference),
    }


def _deduplicate(dataset: Dataset) -> Dataset:
    seen = set()
    keep_indices = []
    for index, text in enumerate(dataset["text"]):
        if text not in seen:
            seen.add(text)
            keep_indices.append(index)
    return dataset.select(keep_indices)


def _token_statistics(dataset: Dataset, tokenizer, max_length: int) -> dict[str, float | int]:
    encoded = tokenizer(
        dataset["text"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    lengths = [len(ids) for ids in encoded]
    return {
        "examples": len(lengths),
        "tokens": int(sum(lengths)),
        "mean_tokens_per_example": (
            float(sum(lengths) / len(lengths)) if lengths else 0.0
        ),
        "max_tokens_per_example": max(lengths, default=0),
    }


def prepare_datasets(config: dict[str, Any], tokenizer):
    """Load, format, deduplicate and create train/validation/test partitions."""
    data_cfg = config["dataset"]
    dataset = load_dataset(
        data_cfg["name"],
        data_cfg["config"],
        split=data_cfg["split"],
    )
    dataset = dataset.shuffle(seed=config["project"]["seed"])

    max_samples = data_cfg.get("max_samples")
    if max_samples is not None:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))

    formatted = dataset.map(
        lambda row: _format_example(row, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting conversations with the model chat template",
    )

    if data_cfg.get("deduplicate_exact_text", True):
        formatted = _deduplicate(formatted)

    train_fraction = float(data_cfg["train_fraction"])
    validation_fraction = float(data_cfg["validation_fraction"])
    test_fraction = float(data_cfg["test_fraction"])
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-6:
        raise ValueError("Train, validation and test fractions must sum to 1.0")

    seed = config["project"]["seed"]
    test_split = formatted.train_test_split(
        test_size=test_fraction,
        seed=seed,
    )
    validation_relative_fraction = validation_fraction / (
        train_fraction + validation_fraction
    )
    train_validation_split = test_split["train"].train_test_split(
        test_size=validation_relative_fraction,
        seed=seed,
    )

    train_full = train_validation_split["train"]
    validation_full = train_validation_split["test"]
    test_full = test_split["test"]

    max_length = int(data_cfg["max_length"])
    stats = {
        "train": _token_statistics(train_full, tokenizer, max_length),
        "validation": _token_statistics(validation_full, tokenizer, max_length),
        "test": _token_statistics(test_full, tokenizer, max_length),
    }

    return {
        "train": train_full.select_columns(["text"]),
        "validation": validation_full.select_columns(["text"]),
        "test": test_full.select_columns(["text"]),
        "test_generation": test_full.select_columns(
            ["prompt_messages", "reference"]
        ),
        "stats": stats,
    }
