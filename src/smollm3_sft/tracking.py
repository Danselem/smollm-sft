import os
from typing import Any

import dagshub
import mlflow
import torch


def initialise_tracking(config: dict[str, Any], credentials: dict[str, str | None]):
    """Configure DagsHub-hosted MLflow and optional system metrics."""
    os.environ["DAGSHUB_USER_TOKEN"] = credentials["dagshub_token"]
    dagshub.init(
        repo_owner=credentials["dagshub_owner"],
        repo_name=credentials["dagshub_repo"],
        mlflow=True,
    )
    mlflow.set_experiment(config["tracking"]["experiment_name"])
    if config["tracking"].get("enable_system_metrics", True):
        mlflow.enable_system_metrics_logging()


def log_hardware_metadata():
    """Log visible GPU names and memory capacities as MLflow parameters."""
    if not torch.cuda.is_available():
        mlflow.set_tag("cuda_available", "false")
        return

    mlflow.set_tag("cuda_available", "true")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        mlflow.log_param(f"gpu_{index}_name", properties.name)
        mlflow.log_param(
            f"gpu_{index}_memory_gb",
            round(properties.total_memory / 1024**3, 2),
        )


def log_gpu_peak_memory():
    """Log peak allocated and reserved CUDA memory after training."""
    if not torch.cuda.is_available():
        return

    for index in range(torch.cuda.device_count()):
        mlflow.log_metric(
            f"gpu_{index}_peak_allocated_gb",
            torch.cuda.max_memory_allocated(index) / 1024**3,
        )
        mlflow.log_metric(
            f"gpu_{index}_peak_reserved_gb",
            torch.cuda.max_memory_reserved(index) / 1024**3,
        )
