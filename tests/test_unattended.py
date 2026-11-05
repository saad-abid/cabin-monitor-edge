"""Unit tests for the unattended-luggage logic, using small synthetic track data.

Scenario timing: 10 frames per second, T = 10 s, X = 15 px, D = 3 x diagonal.
The bag box is 50x50 px (diagonal ~70.7 px), so "nearby" means within ~212 px.
"""

import pytest

from unattended import UnattendedDetector, is_person_nearby
from utils import Detection

FPS = 10
BAG = (500.0, 500.0, 550.0, 550.0)
PERSON_NEAR = (560.0, 380.0, 640.0, 600.0)   # center ~75 px from the bag
PERSON_FAR = (0.0, 0.0, 80.0, 220.0)         # center ~660 px from the bag


def make_detector() -> UnattendedDetector:
    return UnattendedDetector(time_seconds=10.0, motion_pixels=15.0, distance_factor=3.0)


def bag(track_id: int = 1, dx: float = 0.0) -> Detection:
    x1, y1, x2, y2 = BAG
    return Detection(track_id, "backpack", (x1 + dx, y1, x2 + dx, y2))


def person(box=PERSON_NEAR, track_id: int = 99) -> Detection:
    return Detection(track_id, "person", box)


def run_frames(det: UnattendedDetector, frames: list[list[Detection]], start: int = 0) -> list:
    """Feed frames to the detector and return all alerts raised."""
    alerts = []
    for i, dets in enumerate(frames, start=start):
        alerts += det.update(i, i / FPS, dets)
    return alerts


# --- is_person_nearby -------------------------------------------------------

def test_person_close_to_bag_is_nearby() -> None:
    assert is_person_nearby(BAG, [PERSON_NEAR], distance_factor=3.0)


def test_person_far_from_bag_is_not_nearby() -> None:
    assert not is_person_nearby(BAG, [PERSON_FAR], distance_factor=3.0)


def test_no_person_is_not_nearby() -> None:
    assert not is_person_nearby(BAG, [], distance_factor=3.0)


def test_nearby_distance_scales_with_bag_size() -> None:
    big_bag = (500.0, 500.0, 700.0, 700.0)  # 4x larger diagonal -> larger "nearby" zone
    person_at_400px = (950.0, 550.0, 1010.0, 650.0)
    assert not is_person_nearby(BAG, [person_at_400px], 3.0)
    assert is_person_nearby(big_bag, [person_at_400px], 3.0)


# --- UnattendedDetector ------------------------------------------------------

def test_still_alone_bag_alerts_after_T_seconds_not_before() -> None:
    det = make_detector()
    assert run_frames(det, [[bag()]] * 100) == []          # 0.0 .. 9.9 s
    alerts = run_frames(det, [[bag()]], start=100)         # 10.0 s
    assert len(alerts) == 1
    assert alerts[0].track_id == 1
    assert alerts[0].alert_time_s == pytest.approx(10.0)
    assert det.is_active(1)


def test_only_one_alert_per_event() -> None:
    det = make_detector()
    alerts = run_frames(det, [[bag()]] * 300)               # 30 s still and alone
    assert len(alerts) == 1


def test_bag_with_person_nearby_never_alerts() -> None:
    det = make_detector()
    assert run_frames(det, [[bag(), person()]] * 300) == []


def test_person_far_away_does_not_count_as_attending() -> None:
    det = make_detector()
    assert len(run_frames(det, [[bag(), person(PERSON_FAR)]] * 150)) == 1


def test_small_jitter_below_X_still_alerts() -> None:
    det = make_detector()
    frames = [[bag(dx=5.0 if i % 2 else 0.0)] for i in range(150)]  # detector noise of 5 px
    assert len(run_frames(det, frames)) == 1


def test_bag_moving_more_than_X_resets_the_timer() -> None:
    det = make_detector()
    frames = [[bag()]] * 80 + [[bag(dx=40.0)]] * 80        # moves at 8 s, then still 8 s
    assert run_frames(det, frames) == []


def test_person_returning_ends_event_and_leaving_again_starts_new_one() -> None:
    det = make_detector()
    frames = [[bag()]] * 110 + [[bag(), person()]] * 20 + [[bag()]] * 110
    alerts = run_frames(det, frames)
    assert len(alerts) == 2
    assert [a.event_id for a in alerts] == [1, 2]


def test_alert_ends_when_owner_returns() -> None:
    det = make_detector()
    run_frames(det, [[bag()]] * 110)
    assert det.is_active(1)
    run_frames(det, [[bag(), person()]], start=110)
    assert not det.is_active(1)


def test_two_bags_are_tracked_independently() -> None:
    det = make_detector()
    other = Detection(2, "suitcase", (100.0, 100.0, 160.0, 160.0))
    frames = [[bag(), other]] * 110
    alerts = run_frames(det, frames)
    assert sorted(a.track_id for a in alerts) == [1, 2]


def test_detections_without_track_id_are_ignored() -> None:
    det = make_detector()
    assert run_frames(det, [[Detection(None, "backpack", BAG)]] * 150) == []


def test_save_csv_writes_header_even_without_alerts(tmp_path) -> None:
    det = make_detector()
    path = tmp_path / "alerts.csv"
    det.save_csv(path)
    assert path.read_text().startswith("event_id,track_id,class_name")
