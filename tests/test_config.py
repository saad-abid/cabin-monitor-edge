"""Smoke tests: the config file exists and contains the keys the project needs."""

from utils import load_config


def test_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg, dict)


def test_config_has_four_classes() -> None:
    classes = load_config()["detection"]["classes"]
    assert set(classes) == {"person", "backpack", "handbag", "suitcase"}


def test_unattended_thresholds_are_positive() -> None:
    u = load_config()["unattended"]
    assert u["time_seconds"] > 0
    assert u["motion_pixels"] > 0
    assert u["distance_factor"] > 0
