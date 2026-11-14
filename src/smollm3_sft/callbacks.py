import math
import time
from pathlib import Path

import mlflow
import torch
from transformers import TrainerCallback


class MLflowMetricsCallback(TrainerCallback):
    """Send Trainer log metrics and training FLOPs estimates to the active MLflow run."""

    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or mlflow.active_run() is None:
            return
        values = {
            key: float(value)
            for key, value in logs.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        if values:
            mlflow.log_metrics(values, step=max(int(state.global_step), 0))

    def on_train_end(self, args, state, control, **kwargs):
        if self.start_time is None or mlflow.active_run() is None:
            return
        runtime = time.perf_counter() - self.start_time
        mlflow.log_metric("training_wall_time_seconds", runtime)

        total_flos = getattr(state, "total_flos", None)
        if total_flos and total_flos > 0:
            mlflow.log_metric("trainer_estimated_total_flops", float(total_flos))
            mlflow.log_metric(
                "trainer_estimated_flops_per_second",
                float(total_flos) / runtime,
            )


class TorchProfilerCallback(TrainerCallback):
    """Optionally profile a short training window with shapes, memory and FLOP estimates."""

    def __init__(self, config: dict):
        self.config = config
        self.profiler = None
        self.output_dir = Path(config["output_dir"])

    def on_train_begin(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        schedule = torch.profiler.schedule(
            wait=int(self.config["wait_steps"]),
            warmup=int(self.config["warmup_steps"]),
            active=int(self.config["active_steps"]),
            repeat=1,
        )
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.output_dir)
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        )
        self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is None:
            return
        self.profiler.stop()
        sort_key = (
            "self_cuda_time_total"
            if torch.cuda.is_available()
            else "self_cpu_time_total"
        )
        table = self.profiler.key_averages().table(
            sort_by=sort_key,
            row_limit=40,
        )
        (self.output_dir / "operator_profile.txt").write_text(
            table,
            encoding="utf-8",
        )
        profiled_flops = sum(
            event.flops or 0 for event in self.profiler.key_averages()
        )
        if profiled_flops and mlflow.active_run() is not None:
            mlflow.log_metric(
                "profiled_operator_flops_in_profile_window",
                float(profiled_flops),
            )
