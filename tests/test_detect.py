"""Tests for detection helpers and an end-to-end run on a tiny synthetic video."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from detect import build_output_path, get_class_ids, is_video, run
from utils import MODELS_DIR, load_config, resolve_model_path

YOLOV8N = MODELS_DIR / "yolov8n.pt"


def test_is_video_detects_extensions() -> None:
    assert is_video("clip.MP4") is True
    assert is_video("photo.jpg") is False
    with pytest.raises(ValueError):
        is_video("notes.txt")


def test_get_class_ids_returns_the_four_coco_ids() -> None:
    assert get_class_ids(load_config()) == [0, 24, 26, 28]


def test_build_output_path_combines_source_and_model() -> None:
    out = build_output_path(Path("data/sample.mp4"), "yolov8n.pt", Path("results/detections"))
    assert out == Path("results/detections/sample_yolov8n.mp4")


def test_resolve_model_path_points_bare_names_into_models_dir(tmp_path: Path) -> None:
    assert resolve_model_path("yolov8s.pt", models_dir=tmp_path) == tmp_path / "yolov8s.pt"


def test_resolve_model_path_keeps_existing_files(tmp_path: Path) -> None:
    weights = tmp_path / "custom.pt"
    weights.touch()
    assert resolve_model_path(weights) == weights


def _write_synthetic_video(path: Path, n_frames: int = 5, size: tuple[int, int] = (320, 240)) -> None:
    """Write a small video of random-noise frames."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()


@pytest.mark.skipif(not YOLOV8N.exists(), reason="models/yolov8n.pt not downloaded")
def test_run_on_video_writes_annotated_video_with_same_frame_count(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video)

    out = run(str(video), "yolov8n.pt", output_dir=str(tmp_path / "out"))

    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()
