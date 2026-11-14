import json
import logging
import math
from pathlib import Path
from typing import Any

import mlflow
import torch
from omegaconf import OmegaConf
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback

from .callbacks import MLflowMetricsCallback, TorchProfilerCallback
from .config import PROJECT_ROOT, load_config, load_credentials
from .data import prepare_datasets
from .evaluation import run_generation_evaluation
from .model import load_model_and_tokenizer, model_metadata
from .tracking import initialise_tracking, log_gpu_peak_memory, log_hardware_metadata


LOGGER = logging.getLogger("smollm3_sft")


def _training_arguments(config: dict[str, Any], repo_id: str) -> SFTConfig:
    training = config["training"]
    dataset = config["dataset"]
    output_dir = PROJECT_ROOT / config["project"]["output_dir"]

    return SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=int(dataset["max_length"]),
        packing=False,
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=int(training["max_steps"]),
        warmup_steps=int(training["warmup_steps"]),
        weight_decay=float(training["weight_decay"]),
        optim=training["optim"],
        lr_scheduler_type=training["lr_scheduler_type"],
        max_grad_norm=float(training["max_grad_norm"]),
        bf16=bool(training["bf16"]),
        fp16=bool(training["fp16"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        logging_strategy="steps",
        logging_steps=int(training["logging_steps"]),
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=int(training["eval_steps"]),
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=bool(training["load_best_model_at_end"]),
        metric_for_best_model=training["metric_for_best_model"],
        greater_is_better=bool(training["greater_is_better"]),
        dataloader_num_workers=int(training["dataloader_num_workers"]),
        group_by_length=bool(training["group_by_length"]),
        include_num_input_tokens_seen=bool(
            training.get("include_num_input_tokens_seen", True)
        ),
        report_to=[],
        push_to_hub=False,
        hub_model_id=repo_id,
        seed=int(config["project"]["seed"]),
        data_seed=int(config["project"]["seed"]),
    )


def _log_config_and_dataset(config, model_info, datasets):
    params = {
        "model_name": config["model"]["name"],
        "dataset_name": config["dataset"]["name"],
        "dataset_config": config["dataset"]["config"],
        "dataset_split": config["dataset"]["split"],
        "max_length": config["dataset"]["max_length"],
        "learning_rate": config["training"]["learning_rate"],
        "train_batch_size": config["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "max_steps": config["training"]["max_steps"],
        "early_stopping_patience": config["training"]["early_stopping_patience"],
        "seed": config["project"]["seed"],
    }
    params.update({
        key: value for key, value in model_info.items()
        if value is not None and isinstance(value, (str, int, float, bool))
    })
    mlflow.log_params(params)
    mlflow.log_dict(model_info, "metadata/model_architecture.json")
    mlflow.log_dict(datasets["stats"], "metadata/dataset_statistics.json")
    mlflow.log_dict(config, "metadata/experiment_config.json")


def _log_metrics(metrics: dict[str, Any], prefix: str | None = None):
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            cleaned[f"{prefix}_{key}" if prefix else key] = float(value)
    if cleaned:
        mlflow.log_metrics(cleaned)


def _log_generation_artifact(rows: list[dict[str, Any]], report_dir: Path, name: str):
    path = report_dir / name
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(path), artifact_path="generation_evaluation")


def run(config_path: str):
    config = load_config(config_path)
    credentials = load_credentials()
    seed = int(config["project"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    hub_config = config["hub"]
    repo_id = (
        hub_config.get("repo_id")
        or credentials.get("hf_repo_id")
        or (
            f"{credentials['hf_username']}/{Path(config['project']['output_dir']).name}"
            if credentials.get("hf_username") else None
        )
    )
    if hub_config.get("push_to_hub", True) and not repo_id:
        raise RuntimeError(
            "Set HF_REPO_ID in .env, HF_USERNAME, or hub.repo_id in the YAML config."
        )

    initialise_tracking(config, credentials)

    LOGGER.info("Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(config)
    model_info = model_metadata(model)

    LOGGER.info("Preparing train, validation and test datasets")
    datasets = prepare_datasets(config, tokenizer)

    train_args = _training_arguments(config, repo_id or "")
    callbacks = [
        MLflowMetricsCallback(),
        EarlyStoppingCallback(
            early_stopping_patience=int(
                config["training"]["early_stopping_patience"]
            ),
            early_stopping_threshold=float(
                config["training"]["early_stopping_threshold"]
            ),
        ),
    ]

    if config["profiler"].get("enabled", False):
        profiler_config = dict(config["profiler"])
        profiler_config["output_dir"] = str(
            PROJECT_ROOT / profiler_config["output_dir"]
        )
        callbacks.append(TorchProfilerCallback(profiler_config))

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    report_dir = PROJECT_ROOT / config["project"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir = PROJECT_ROOT / config["project"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_cfg = config["tracking"]
    with mlflow.start_run(run_name=tracking_cfg["run_name"]):
        mlflow.set_tags({
            "base_model": config["model"]["name"],
            "dataset": config["dataset"]["name"],
            "fine_tuning_method": "full_parameter_sft",
            "framework": "transformers_trl",
        })
        log_hardware_metadata()
        _log_config_and_dataset(config, model_info, datasets)

        LOGGER.info("Evaluating base model on held-out test set")
        baseline_test_metrics = trainer.evaluate(
            eval_dataset=datasets["test"],
            metric_key_prefix="baseline_test",
        )
        _log_metrics(baseline_test_metrics)

        generation_cfg = config["evaluation"]["generation"]
        if generation_cfg.get("enabled", True):
            LOGGER.info("Generating base-model test responses")
            baseline_rows, baseline_generation_metrics = run_generation_evaluation(
                trainer.model,
                tokenizer,
                datasets["test_generation"],
                generation_cfg,
                compute_rouge_l=bool(config["evaluation"].get("rouge_l", True)),
            )
            _log_metrics(baseline_generation_metrics, "baseline")
            _log_generation_artifact(
                baseline_rows, report_dir, "baseline_generations.json"
            )
        else:
            baseline_rows = []
            baseline_generation_metrics = {}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        LOGGER.info("Starting supervised fine-tuning")
        train_result = trainer.train()
        _log_metrics(train_result.metrics, "train")

        # Trainer counts input tokens when include_num_input_tokens_seen is enabled.
        tokens_seen = int(getattr(trainer.state, "num_input_tokens_seen", 0) or 0)
        train_runtime = train_result.metrics.get("train_runtime")
        if tokens_seen > 0:
            mlflow.log_metric("training_tokens_seen", tokens_seen)
            if train_runtime and train_runtime > 0:
                mlflow.log_metric(
                    "training_tokens_per_second",
                    tokens_seen / float(train_runtime),
                )

        if torch.cuda.is_available():
            log_gpu_peak_memory()

        LOGGER.info("Evaluating best checkpoint on held-out test set")
        test_metrics = trainer.evaluate(
            eval_dataset=datasets["test"],
            metric_key_prefix="test",
        )
        _log_metrics(test_metrics)

        for key, source_key in (
            ("baseline_test_perplexity", "baseline_test_loss"),
            ("test_perplexity", "test_loss"),
        ):
            loss = (
                baseline_test_metrics.get(source_key)
                if key.startswith("baseline")
                else test_metrics.get(source_key)
            )
            if loss is not None and math.isfinite(float(loss)):
                mlflow.log_metric(key, math.exp(min(float(loss), 100)))

        generation_comparison = []
        if generation_cfg.get("enabled", True):
            LOGGER.info("Generating fine-tuned model test responses")
            finetuned_rows, finetuned_generation_metrics = run_generation_evaluation(
                trainer.model,
                tokenizer,
                datasets["test_generation"],
                generation_cfg,
                compute_rouge_l=bool(config["evaluation"].get("rouge_l", True)),
            )
            _log_metrics(finetuned_generation_metrics, "finetuned")
            _log_generation_artifact(
                finetuned_rows, report_dir, "finetuned_generations.json"
            )

            generation_comparison = [
                {
                    "prompt_messages": base["prompt_messages"],
                    "reference": base["reference"],
                    "base_model_response": base["generated_response"],
                    "finetuned_model_response": tuned["generated_response"],
                }
                for base, tuned in zip(baseline_rows, finetuned_rows)
            ]
            _log_generation_artifact(
                generation_comparison,
                report_dir,
                "generation_comparison.json",
            )

        mlflow.log_metric("best_validation_loss", float(trainer.state.best_metric)) if trainer.state.best_metric is not None else None
        if trainer.state.best_model_checkpoint:
            mlflow.set_tag("best_model_checkpoint", trainer.state.best_model_checkpoint)

        LOGGER.info("Saving final model and tokenizer")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        trainer.save_state()
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_metrics("test", test_metrics)

        # Save local run summary and upload reports/configuration to MLflow.
        summary = {
            "baseline_test_metrics": baseline_test_metrics,
            "training_metrics": train_result.metrics,
            "test_metrics": test_metrics,
            "baseline_generation_metrics": baseline_generation_metrics,
            "best_validation_loss": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }
        (report_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        config_path_local = report_dir / "experiment_config.yaml"
        config_path_local.write_text(
            OmegaConf.to_yaml(OmegaConf.create(config)),
            encoding="utf-8",
        )
        mlflow.log_artifacts(str(report_dir), artifact_path="reports")

        if config["profiler"].get("enabled", False):
            profiler_dir = PROJECT_ROOT / config["profiler"]["output_dir"]
            if profiler_dir.exists():
                mlflow.log_artifacts(str(profiler_dir), artifact_path="profiler")

        if config["tracking"].get("log_final_model_artifact", False):
            mlflow.log_artifacts(str(output_dir), artifact_path="final_model")

        if hub_config.get("push_to_hub", True):
            LOGGER.info("Pushing final model to Hugging Face Hub: %s", repo_id)
            trainer.push_to_hub(
                repo_id=repo_id,
                token=credentials["hf_token"],
                private=bool(hub_config.get("private", False)),
                commit_message=hub_config["commit_message"],
            )

    LOGGER.info("Experiment completed")
    LOGGER.info(
        "DagsHub MLflow UI: https://dagshub.com/%s/%s.mlflow",
        credentials["dagshub_owner"],
        credentials["dagshub_repo"],
    )
    if hub_config.get("push_to_hub", True):
        LOGGER.info("Hugging Face model: https://huggingface.co/%s", repo_id)
