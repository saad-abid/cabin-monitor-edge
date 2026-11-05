"""Tests for people counting and the moving average."""

import pandas as pd
import pytest

from occupancy import OccupancyCounter, count_people
from utils import Detection

P = Detection(1, "person", (0, 0, 10, 10))
B = Detection(2, "backpack", (0, 0, 10, 10))


def test_count_people_ignores_luggage() -> None:
    assert count_people([P, P, B]) == 2
    assert count_people([]) == 0


def test_moving_average_smooths_a_missed_detection() -> None:
    counter = OccupancyCounter(window=3)
    smoothed = [counter.update(i, i / 10, dets).people_smoothed
                for i, dets in enumerate([[P, P], [P, P], [P]])]  # one person missed in frame 2
    assert smoothed == [2.0, 2.0, pytest.approx(1.67)]


def test_window_forgets_old_frames() -> None:
    counter = OccupancyCounter(window=2)
    for i, dets in enumerate([[P, P, P], [P], [P]]):
        rec = counter.update(i, i / 10, dets)
    assert rec.people_smoothed == 1.0


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError):
        OccupancyCounter(window=0)


def test_save_csv_has_one_row_per_frame(tmp_path) -> None:
    counter = OccupancyCounter(window=2)
    for i in range(4):
        counter.update(i, i / 10, [P])
    path = tmp_path / "occupancy.csv"
    counter.save_csv(path)
    df = pd.read_csv(path)
    assert list(df.columns) == ["frame", "time_s", "people", "people_smoothed"]
    assert len(df) == 4
