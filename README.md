# SmolLM3 SFT Lab

A configuration-driven supervised fine-tuning project for `HuggingFaceTB/SmolLM3-3B-Base`. The structure separates configuration, data preparation, model loading, callbacks, evaluation, experiment tracking, and orchestration.

The training method is **full-parameter supervised fine-tuning**, matching the original training script. It does not silently switch to LoRA.

## Project structure

```text
smollm3-sft/
├── configs/
│   └── smollm3_sft.yaml       # NeMo-style experiment configuration
├── src/smollm3_sft/
│   ├── callbacks.py           # MLflow metrics and optional PyTorch profiler
│   ├── cli.py                 # Command-line entry point
│   ├── config.py              # YAML and environment loading
│   ├── data.py                # Formatting, deduplication and data splits
│   ├── evaluation.py          # Generation and lightweight text metrics
│   ├── model.py               # Model/tokenizer loading and architecture metadata
│   ├── tracking.py            # DagsHub/MLflow and hardware metrics
│   └── train.py               # Experiment orchestration
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## 1. Configure credentials

Copy `.env.example` to `.env` and fill in your credentials:

```dotenv
DAGSHUB_REPO_OWNER=your_dagshub_username
DAGSHUB_REPO=smollm3-finetuning
DAGSHUB_TOKEN=your_dagshub_token

HF_USERNAME=your_huggingface_username
HF_TOKEN=your_huggingface_token
# Optional override:
# HF_REPO_ID=your_username/SmolLM3-Custom-SFT
```

The loader also accepts `HUGGINGFACE_USERNAME` and `HUGGINGFACE_TOKEN`. Never commit `.env`.

## 2. Install

This project uses `uv`. From the project directory:

```bash
cp .env.example .env
uv sync
```

Install a PyTorch build compatible with the CUDA/runtime available on your machine or cluster if the default installation does not match your environment. The project intentionally does not assume a particular GPU model.

## 3. Review the YAML configuration

Edit `configs/smollm3_sft.yaml` before training. Important settings include:

- `model.dtype`, `training.bf16`, and `training.fp16`: choose a precision supported by your GPU.
- `dataset.max_samples` and `dataset.max_length`: control the dataset subset and sequence truncation.
- `training.*`: batch size, accumulation, learning rate, number of steps and checkpoint interval.
- `training.early_stopping_patience`: number of validation evaluations without improvement before stopping.
- `evaluation.generation`: number of held-out prompts and generation settings.
- `profiler.enabled`: enable a short PyTorch Profiler capture with tensor shapes, memory and supported operator FLOP estimates.
- `hub.push_to_hub` and `hub.repo_id`: control Hugging Face publication.

The default split is 80% train, 10% validation and 10% test. Exact duplicate formatted conversations are removed before splitting. The validation split controls early stopping and best-checkpoint selection. The test split is not passed to training; it is used for baseline and final evaluation.

## 4. Run

```bash
uv run python -m smollm3_sft --config configs/smollm3_sft.yaml
```

The script will:

1. Load the base model and tokenizer.
2. Prepare and split the selected dataset.
3. Evaluate the base model on the held-out test split.
4. Generate baseline responses for a fixed sample of test prompts.
5. Fine-tune using the training split and validate at the configured interval.
6. Stop after the configured number of validation evaluations without improvement and restore the best checkpoint.
7. Evaluate the selected checkpoint on the test split.
8. Generate fine-tuned responses for the same test prompts and save a comparison.
9. Save the final model locally and push it to the configured Hugging Face repository.
10. Log configuration, parameters, metrics, reports and optional system metrics to DagsHub-hosted MLflow.

## 5. Outputs

Local outputs:

- `outputs/SmolLM3-Custom-SFT/`: final model and tokenizer.
- `reports/`: test metrics and base/fine-tuned generation comparisons.
- `profiler_traces/`: optional PyTorch Profiler traces and operator summary.

DagsHub/MLflow:

- Experiment parameters and model/dataset metadata.
- Training and validation metrics.
- Baseline and final test loss/perplexity.
- Trainer FLOPs estimate and average estimated FLOP/s where available.
- GPU peak memory and MLflow system metrics when supported.
- Evaluation reports and generation comparisons.

Hugging Face Hub:

- Final model and tokenizer, pushed to `hub.repo_id` or the repository derived from `HF_USERNAME`.

By default, model weights are not duplicated into MLflow artifacts because the final model is published to Hugging Face. Set `tracking.log_final_model_artifact: true` if you also want a copy in DagsHub artifact storage. This can use substantial repository storage.

## 6. Computational profiling notes

`trainer_estimated_total_flops` and `trainer_estimated_flops_per_second` use the FLOPs estimate reported by the Transformers Trainer. They are estimates, not a direct count of every hardware instruction.

Set `profiler.enabled: true` to capture a short training window using `torch.profiler` with `record_shapes`, `profile_memory`, and `with_flops`. Operator FLOP estimates are only available for operations supported by PyTorch Profiler and describe the captured window, not necessarily the entire run. Profiling adds overhead; leave it disabled for normal benchmark runs and run a separate profiling experiment.

MLflow system metrics depend on the monitoring libraries and hardware/cluster permissions. The project includes `psutil` and `nvidia-ml-py` for this purpose.

## 7. Evaluation interpretation

Perplexity is derived from token-level cross-entropy and is useful for language-modelling comparisons, but it is not a measure of instruction-following quality. The optional ROUGE-L score measures lexical overlap with the reference response; it does not establish factual correctness or semantic quality. The saved paired generations are intended for subsequent blind human evaluation or a separately configured benchmark such as IFEval. No benchmark score is claimed by this project unless that benchmark is run independently.

## Reproducibility

The YAML captures experiment settings and the script records the random seed, model metadata, dataset split sizes, token statistics, and software run metrics. For a fully reproducible report, also record the exact dataset revision, model revision, Git commit, GPU/driver details, and dependency lockfile for each run.
