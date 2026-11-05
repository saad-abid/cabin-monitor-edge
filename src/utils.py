"""Shared helpers used by every script in the project."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels: top-left and bottom-right corners
LUGGAGE_CLASSES = frozenset({"backpack", "handbag", "suitcase"})


@dataclass(frozen=True)
class Detection:
    """One object found in one frame. track_id is None when the tracker has not assigned an ID yet."""

    track_id: int | None
    class_name: str
    box: Box


def box_center(box: Box) -> tuple[float, float]:
    """Return the (x, y) center of a box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_diagonal(box: Box) -> float:
    """Return the length of the box diagonal in pixels (a simple measure of its size)."""
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)


def distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    """Return the straight-line distance between two points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])

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
