from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(config: dict[str, Any]):
    """Load the configured causal language model and tokenizer."""
    model_cfg = config["model"]
    dtype = getattr(torch, model_cfg["dtype"])

    tokenizer_kwargs = {"trust_remote_code": model_cfg["trust_remote_code"]}
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": model_cfg["trust_remote_code"],
    }

    if model_cfg.get("revision"):
        tokenizer_kwargs["revision"] = model_cfg["revision"]
        model_kwargs["revision"] = model_cfg["revision"]

    if model_cfg.get("device_map") is not None:
        model_kwargs["device_map"] = model_cfg["device_map"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], **tokenizer_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], **model_kwargs
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if config["training"].get("gradient_checkpointing", False):
        model.config.use_cache = False

    return model, tokenizer


def model_metadata(model) -> dict[str, Any]:
    """Return architecture and parameter-count metadata."""
    config = model.config
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "model_parameter_size_gb": round(parameter_bytes / 1024**3, 3),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
    }
