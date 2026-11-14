.DEFAULT_GOAL := help
SRC := src

## ──────────────────────────────────────────────
## Setup
## ──────────────────────────────────────────────

.PHONY: loaduv
loaduv: 
	module load python/uv

.PHONY: init
init: ## Create venv, pyproject.toml, and install all deps
	uv venv --python 3.12
	uv init

.PHONY: bootstrap
bootstrap: ## Install deps into existing venv
	uv add --requirements requirements.txt --link-mode=copy

.PHONY: install
install: ## Install deps into existing venv
	uv sync

.PHONY: env
env: ## Install deps into existing venv
	cp .env.example .env

.PHONY: srun
srun:
	srun --x11 \
		--cpus-per-task=2 \
		--mem=8GB \
		--pty bash


.PHONY: cuda
cuda: ## Start interactive CUDA session on ilifu
	srun --x11 \
		--time=1:00:00 \
		--nodes=1 \
		--partition=GPU \
		--mem=64GB \
		--nodelist=gpu-[001-007] \
		--gres=gpu:1 \
		--pty bash


.PHONY: sft
sft: # Run training with config file
	trl sft --config sft_config.yaml