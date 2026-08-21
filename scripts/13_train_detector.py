#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.detector_training import validate_detector_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune cow instance segmentation on owned camera frames."
    )
    parser.add_argument("--data", required=True, type=Path, help="Ultralytics dataset YAML")
    parser.add_argument("--dataset-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--min-images", type=int, default=300)
    args = parser.parse_args()

    coverage = validate_detector_dataset(
        args.data, args.dataset_metadata, min_images=args.min_images
    )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for detector fine-tuning") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    result = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(args.output_dir.resolve()),
        name="cow-segmentation-v1",
        exist_ok=False,
    )
    payload = {**coverage, "model": args.model, "epochs": args.epochs, "result": str(result)}
    with (args.output_dir / "training_request.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"detector training complete: {args.output_dir}")


if __name__ == "__main__":
    main()
