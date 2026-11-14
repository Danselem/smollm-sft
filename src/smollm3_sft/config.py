from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and the project-level .env file."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = OmegaConf.load(path)
    return OmegaConf.to_container(config, resolve=True)


def load_credentials() -> dict[str, str | None]:
    """Read DagsHub and Hugging Face credentials from environment variables."""
    import os

    credentials = {
        "dagshub_owner": os.getenv("DAGSHUB_REPO_OWNER"),
        "dagshub_repo": os.getenv("DAGSHUB_REPO"),
        "dagshub_token": os.getenv("DAGSHUB_TOKEN"),
        "hf_token": (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        ),
        "hf_username": (
            os.getenv("HF_USERNAME")
            or os.getenv("HUGGINGFACE_USERNAME")
        ),
        "hf_repo_id": os.getenv("HF_REPO_ID"),
    }

    missing = [
        name for name in ("dagshub_owner", "dagshub_repo", "dagshub_token", "hf_token")
        if not credentials[name]
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables for: " + ", ".join(missing)
        )

    return credentials
