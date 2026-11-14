import argparse
import logging

from .train import run


def main():
    parser = argparse.ArgumentParser(description="Run config-driven SmolLM3 SFT")
    parser.add_argument(
        "--config",
        default="configs/smollm3_sft.yaml",
        help="Path to the YAML experiment configuration",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(args.config)
