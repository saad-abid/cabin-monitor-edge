"""Unattended luggage logic.

A luggage item is "unattended" when, for at least T seconds in a row:
  (a) its center stays within X pixels of where it was, and
  (b) no person is within D x (luggage box diagonal) of it.
One alert is raised per event. The event ends when the bag moves or a person comes near.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from utils import LUGGAGE_CLASSES, Box, Detection, box_center, box_diagonal, distance


def is_person_nearby(luggage: Box, persons: list[Box], distance_factor: float) -> bool:
    """True if any person's center is within distance_factor x luggage diagonal of the luggage center."""
    max_dist = distance_factor * box_diagonal(luggage)
    center = box_center(luggage)
    return any(distance(center, box_center(p)) <= max_dist for p in persons)


@dataclass
class Alert:
    """One unattended-luggage event."""

    event_id: int
    track_id: int
    class_name: str
    still_since_s: float
    alert_time_s: float
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class _LuggageState:
    """What we remember about one luggage track between frames."""

    anchor: tuple[float, float]       # center position when the bag became still
    alone_since: float | None = None  # time when "still and alone" started (None = not counting)
    alerted: bool = False             # an alert was already raised for the current event


class UnattendedDetector:
    """Follow every luggage track over time and raise alerts."""

    def __init__(self, time_seconds: float, motion_pixels: float, distance_factor: float) -> None:
        self.time_seconds = time_seconds
        self.motion_pixels = motion_pixels
        self.distance_factor = distance_factor
        self._states: dict[int, _LuggageState] = {}
        self.alerts: list[Alert] = []

    def update(self, frame: int, time_s: float, detections: list[Detection]) -> list[Alert]:
        """Process one frame. Return the alerts raised in this frame (usually none)."""
        persons = [d.box for d in detections if d.class_name == "person"]
        new_alerts = []
        for det in detections:
            if det.class_name not in LUGGAGE_CLASSES or det.track_id is None:
                continue
            if self._update_track(det, persons, time_s):
                alert = self._make_alert(det, frame, time_s)
                self.alerts.append(alert)
                new_alerts.append(alert)
        return new_alerts

    def is_active(self, track_id: int | None) -> bool:
        """True if this luggage track is currently in an unattended event (draw it red)."""
        state = self._states.get(track_id) if track_id is not None else None
        return bool(state and state.alerted)

    def _update_track(self, det: Detection, persons: list[Box], time_s: float) -> bool:
        """Update one luggage track. Return True when a new alert must be raised."""
        center = box_center(det.box)
        state = self._states.setdefault(det.track_id, _LuggageState(anchor=center))

        if is_person_nearby(det.box, persons, self.distance_factor):
            self._reset(state, center, start=None)   # attended: stop counting
        elif distance(center, state.anchor) > self.motion_pixels:
            self._reset(state, center, start=time_s)  # bag moved: restart from new place
        elif state.alone_since is None:
            state.alone_since = time_s                # just became still and alone

        due = state.alone_since is not None and time_s - state.alone_since >= self.time_seconds
        if due and not state.alerted:
            state.alerted = True
            return True
        return False

    @staticmethod
    def _reset(state: _LuggageState, center: tuple[float, float], start: float | None) -> None:
        """End any current event and restart counting from this position."""
        state.anchor = center
        state.alone_since = start
        state.alerted = False

    def _make_alert(self, det: Detection, frame: int, time_s: float) -> Alert:
        """Build the alert record for a luggage detection."""
        x1, y1, x2, y2 = (round(v, 1) for v in det.box)
        since = self._states[det.track_id].alone_since
        return Alert(len(self.alerts) + 1, det.track_id, det.class_name,
                     round(since, 3), round(time_s, 3), frame, x1, y1, x2, y2)

    def save_csv(self, path: Path) -> None:
        """Write all alerts to a CSV file (header only if there were none)."""
        columns = list(Alert.__dataclass_fields__)
        pd.DataFrame([asdict(a) for a in self.alerts], columns=columns).to_csv(path, index=False)
