"""Tests for the tracking pipeline helpers and an end-to-end run on a synthetic video."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from track import RED, box_style, draw_frame, run
from occupancy import OccupancyRecord
from unattended import UnattendedDetector
from utils import MODELS_DIR, Detection

YOLOV8N = MODELS_DIR / "yolov8n.pt"


def test_active_unattended_bag_is_drawn_red_with_label() -> None:
    det = UnattendedDetector(time_seconds=1.0, motion_pixels=15.0, distance_factor=3.0)
    bag = Detection(7, "suitcase", (10.0, 10.0, 60.0, 60.0))
    for i in range(20):
        det.update(i, i / 10, [bag])
    color, label = box_style(bag, det)
    assert color == RED
    assert label == "UNATTENDED suitcase #7"


def test_draw_frame_keeps_frame_size_and_does_not_modify_input() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    det = UnattendedDetector(10.0, 15.0, 3.0)
    occ = OccupancyRecord(frame=0, time_s=0.0, people=1, people_smoothed=1.0)
    out = draw_frame(frame, [Detection(1, "person", (20, 40, 80, 200))], det, occ)
    assert out.shape == frame.shape
    assert frame.sum() == 0 and out.sum() > 0


@pytest.mark.skipif(not YOLOV8N.exists(), reason="models/yolov8n.pt not downloaded")
def test_run_produces_video_and_both_csvs(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for _ in range(5):
        writer.write(np.full((240, 320, 3), 127, dtype=np.uint8))
    writer.release()

    paths = run(str(video), "yolov8n.pt", results_dir=str(tmp_path / "results"))

    assert paths["video"].exists()
    assert len(pd.read_csv(paths["occupancy"])) == 5
    assert list(pd.read_csv(paths["alerts"]).columns)[:2] == ["event_id", "track_id"]
