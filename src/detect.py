"""Detect people and luggage in an image or a video and save an annotated copy.

Example:
    python src/detect.py --source data/sample.mp4 --model yolov8n.pt
"""

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from utils import PROJECT_ROOT, load_config, resolve_model_path, select_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_video(path: str | Path) -> bool:
    """Return True if the file extension is a video format, False if it is an image."""
    suffix = Path(path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return True
    if suffix in IMAGE_EXTENSIONS:
        return False
    raise ValueError(f"Unsupported file type: {suffix}")


def get_class_ids(cfg: dict[str, Any]) -> list[int]:
    """Return the COCO class IDs to keep (person, backpack, handbag, suitcase)."""
    return sorted(cfg["detection"]["classes"].values())


def build_output_path(source: Path, model_name: str, output_dir: Path) -> Path:
    """Build the annotated output path, e.g. results/detections/sample_yolov8n.mp4."""
    return output_dir / f"{source.stem}_{Path(model_name).stem}{source.suffix}"


def predict_frame(model: YOLO, frame: np.ndarray, cfg: dict[str, Any], device: str):
    """Run the detector on one frame, keeping only our classes and thresholds."""
    det = cfg["detection"]
    return model.predict(
        frame,
        classes=get_class_ids(cfg),
        conf=det["confidence"],
        iou=det["iou"],
        imgsz=det["imgsz"],
        device=device,
        verbose=False,
    )[0]


def count_classes(result) -> Counter:
    """Count detections per class name in one YOLO result."""
    return Counter(result.names[int(c)] for c in result.boxes.cls)


def detect_image(model: YOLO, source: Path, out_path: Path, cfg: dict[str, Any], device: str) -> Counter:
    """Detect objects in one image, save the annotated image, return counts per class."""
    frame = cv2.imread(str(source))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {source}")
    result = predict_frame(model, frame, cfg, device)
    cv2.imwrite(str(out_path), result.plot())
    return count_classes(result)


def detect_video(model: YOLO, source: Path, out_path: Path, cfg: dict[str, Any], device: str) -> Counter:
    """Detect objects frame by frame, save the annotated video, return total counts per class."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    totals: Counter = Counter()
    n_frames, start = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = predict_frame(model, frame, cfg, device)
        writer.write(result.plot())
        totals += count_classes(result)
        n_frames += 1
    cap.release()
    writer.release()

    elapsed = time.perf_counter() - start
    print(f"Processed {n_frames} frames in {elapsed:.1f}s ({n_frames / max(elapsed, 1e-9):.1f} FPS incl. drawing)")
    return totals


def run(source: str, model_name: str, output_dir: str | None = None, config_path: str | None = None) -> Path:
    """Load config and model, run detection on the source, return the output file path."""
    cfg = load_config(config_path) if config_path else load_config()
    device = select_device()
    model = YOLO(str(resolve_model_path(model_name)))
    src = Path(source)
    out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / cfg["paths"]["results_dir"] / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = build_output_path(src, model_name, out_dir)

    print(f"Model: {model_name} | device: {device} | source: {src}")
    detect_fn = detect_video if is_video(src) else detect_image
    counts = detect_fn(model, src, out_path, cfg, device)
    print(f"Detections per class: {dict(counts)}")
    print(f"Saved: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    p = argparse.ArgumentParser(description="Detect people and luggage in an image or video.")
    p.add_argument("--source", required=True, help="Path to an image or video file")
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt or yolov8s.pt (stored in models/)")
    p.add_argument("--output-dir", default=None, help="Where to save the result (default: results/detections)")
    p.add_argument("--config", default=None, help="Path to settings.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source, args.model, args.output_dir, args.config)
