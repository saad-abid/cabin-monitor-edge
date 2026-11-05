"""Shared helpers used by every script in the project."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "settings.yaml"


MODELS_DIR = PROJECT_ROOT / "models"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config file and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_path(name: str | Path, models_dir: Path = MODELS_DIR) -> Path:
    """Return where a model file lives.

    An existing path is used as-is. A bare name like "yolov8n.pt" points into models/;
    if the file is not there yet, Ultralytics downloads it to that location.
    """
    path = Path(name)
    if path.exists():
        return path
    return models_dir / path.name


def select_device() -> str:
    """Return "cuda:0" if PyTorch can use an NVIDIA GPU, otherwise "cpu"."""
    import torch  # imported here so loading the config does not require PyTorch

    return "cuda:0" if torch.cuda.is_available() else "cpu"
