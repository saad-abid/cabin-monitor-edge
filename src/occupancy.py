"""Count people per frame and smooth the count over time with a moving average."""

from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from utils import Detection


def count_people(detections: list[Detection]) -> int:
    """Return how many detections are persons."""
    return sum(1 for d in detections if d.class_name == "person")


@dataclass
class OccupancyRecord:
    """People count for one frame."""

    frame: int
    time_s: float
    people: int
    people_smoothed: float


class OccupancyCounter:
    """Keep the last N counts and return their average, so one missed detection does not make the count jump."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._recent: deque[int] = deque(maxlen=window)
        self.records: list[OccupancyRecord] = []

    def update(self, frame: int, time_s: float, detections: list[Detection]) -> OccupancyRecord:
        """Add one frame's detections and return the raw and smoothed count."""
        people = count_people(detections)
        self._recent.append(people)
        smoothed = sum(self._recent) / len(self._recent)
        record = OccupancyRecord(frame, round(time_s, 3), people, round(smoothed, 2))
        self.records.append(record)
        return record

    def save_csv(self, path: Path) -> None:
        """Write all records to a CSV file."""
        columns = ["frame", "time_s", "people", "people_smoothed"]
        pd.DataFrame([asdict(r) for r in self.records], columns=columns).to_csv(path, index=False)
