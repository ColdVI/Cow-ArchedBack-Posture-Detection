#!/usr/bin/env python3
"""Run the bounded T1 zero-shot trial in a dedicated Python 3.10-3.12 env."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
        raise RuntimeError(
            "DeepLabCut officially supports Python 3.10-3.12. Run this trial in a "
            "separate environment; do not alter the project's Python 3.13 environment."
        )
    parser = argparse.ArgumentParser(description="T1 SuperAnimal-Quadruped zero-shot trial.")
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-adapt", action="store_true")
    parser.add_argument("--max-individuals", type=int, default=3)
    args = parser.parse_args()
    missing = [str(path) for path in args.videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Trial videos not found: {missing}")

    import deeplabcut

    package = Path(deeplabcut.__file__).resolve().parent
    config_path = package / "modelzoo" / "project_configs" / "superanimal_quadruped.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not inspect SuperAnimal keypoint config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bodyparts = list(config.get("bodyparts", []))
    print(f"SuperAnimal-Quadruped bodyparts ({len(bodyparts)}): {bodyparts}")
    if len(bodyparts) != 39:
        raise RuntimeError(f"Expected 39 SuperAnimal-Quadruped keypoints, found {len(bodyparts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    deeplabcut.video_inference_superanimal(
        [str(path) for path in args.videos],
        "superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        video_adapt=args.video_adapt,
        max_individuals=args.max_individuals,
        dest_folder=str(args.output_dir),
    )


if __name__ == "__main__":
    main()
