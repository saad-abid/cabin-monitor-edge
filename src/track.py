"""Track people and luggage in a video, count occupancy and flag unattended luggage.

Example:
    python src/track.py --source data/sample.mp4 --model yolov8n.pt

Outputs:
    results/detections/<video>_<model>_tracked.mp4   annotated video
    results/occupancy.csv                           people per frame (+ smoothed)
    results/alerts.csv                              one row per unattended-luggage event
"""

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from detect import get_class_ids
from occupancy import OccupancyCounter, OccupancyRecord
from unattended import UnattendedDetector
from utils import PROJECT_ROOT, Detection, load_config, resolve_model_path, select_device

GREEN, BLUE, RED, WHITE = (0, 200, 0), (255, 140, 0), (0, 0, 255), (255, 255, 255)  # OpenCV uses BGR


def result_to_detections(result) -> list[Detection]:
    """Convert one Ultralytics tracking result into a list of Detection records."""
    boxes = result.boxes
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
    return [
        Detection(tid, result.names[int(c)], tuple(float(v) for v in xyxy))
        for tid, c, xyxy in zip(ids, boxes.cls.tolist(), boxes.xyxy.tolist())
    ]


def track_frame(model: YOLO, frame: np.ndarray, cfg: dict[str, Any], device: str) -> list[Detection]:
    """Run detection + ByteTrack on one frame. persist=True keeps IDs between frames."""
    det = cfg["detection"]
    result = model.track(
        frame, persist=True, tracker=cfg["tracking"]["tracker"], classes=get_class_ids(cfg),
        conf=det["confidence"], iou=det["iou"], imgsz=det["imgsz"], device=device, verbose=False,
    )[0]
    return result_to_detections(result)


def box_style(d: Detection, unattended: UnattendedDetector) -> tuple[tuple[int, int, int], str]:
    """Pick the box color and label for one detection."""
    tid = "" if d.track_id is None else f" #{d.track_id}"
    if unattended.is_active(d.track_id) and d.class_name != "person":
        return RED, f"UNATTENDED {d.class_name}{tid}"
    color = GREEN if d.class_name == "person" else BLUE
    return color, f"{d.class_name}{tid}"


def draw_frame(frame: np.ndarray, detections: list[Detection], unattended: UnattendedDetector,
               occ: OccupancyRecord) -> np.ndarray:
    """Draw boxes, labels and the occupancy counter on a copy of the frame."""
    out = frame.copy()
    for d in detections:
        color, label = box_style(d, unattended)
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if color == RED else 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    text = f"People: {occ.people}  (avg {occ.people_smoothed:.1f})"
    cv2.rectangle(out, (0, 0), (330, 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    return out


def process_video(source: Path, model: YOLO, cfg: dict[str, Any], device: str,
                  out_video: Path) -> tuple[OccupancyCounter, UnattendedDetector]:
    """Read the video frame by frame, track, count, check luggage and write the annotated video."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    u = cfg["unattended"]
    occupancy = OccupancyCounter(cfg["occupancy"]["moving_average_window"])
    unattended = UnattendedDetector(u["time_seconds"], u["motion_pixels"], u["distance_factor"])

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time_s = frame_idx / fps  # timestamp comes from the video, not the wall clock
        detections = track_frame(model, frame, cfg, device)
        occ = occupancy.update(frame_idx, time_s, detections)
        for alert in unattended.update(frame_idx, time_s, detections):
            print(f"ALERT t={alert.alert_time_s:.1f}s: {alert.class_name} #{alert.track_id} unattended")
        writer.write(draw_frame(frame, detections, unattended, occ))
        frame_idx += 1
    cap.release()
    writer.release()
    print(f"Processed {frame_idx} frames at {fps:.1f} FPS (video rate)")
    return occupancy, unattended


def run(source: str, model_name: str, results_dir: str | None = None,
        config_path: str | None = None) -> dict[str, Path]:
    """Run the full pipeline on a video and return the paths of the three outputs."""
    cfg = load_config(config_path) if config_path else load_config()
    device = select_device()
    model = YOLO(str(resolve_model_path(model_name)))
    src = Path(source)
    res_dir = Path(results_dir) if results_dir else PROJECT_ROOT / cfg["paths"]["results_dir"]
    (res_dir / "detections").mkdir(parents=True, exist_ok=True)

    paths = {
        "video": res_dir / "detections" / f"{src.stem}_{Path(model_name).stem}_tracked.mp4",
        "occupancy": res_dir / "occupancy.csv",
        "alerts": res_dir / "alerts.csv",
    }
    print(f"Model: {model_name} | device: {device} | source: {src}")
    occupancy, unattended = process_video(src, model, cfg, device, paths["video"])
    occupancy.save_csv(paths["occupancy"])
    unattended.save_csv(paths["alerts"])
    print(f"Alerts raised: {len(unattended.alerts)}")
    for p in paths.values():
        print(f"Saved: {p}")
    return paths


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    p = argparse.ArgumentParser(description="Track people/luggage, count occupancy, flag unattended luggage.")
    p.add_argument("--source", required=True, help="Path to a video file")
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt or yolov8s.pt (stored in models/)")
    p.add_argument("--results-dir", default=None, help="Output folder (default: results/)")
    p.add_argument("--config", default=None, help="Path to settings.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source, args.model, args.results_dir, args.config)
