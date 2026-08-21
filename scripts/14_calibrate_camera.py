#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.camera import fit_plumb_line_calibration, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit radial distortion from clicked points on known-straight beams."
    )
    parser.add_argument("--lines", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--focal-length-px", type=float)
    args = parser.parse_args()

    payload = json.loads(args.lines.read_text(encoding="utf-8"))
    calibration = fit_plumb_line_calibration(
        payload["lines"],
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        focal_length_px=args.focal_length_px or payload.get("focal_length_px"),
    )
    if calibration.rms_line_error_after_px >= calibration.rms_line_error_before_px:
        raise RuntimeError(
            "Calibration did not reduce plumb-line error; add lines spanning more of the frame."
        )
    save_calibration(calibration, args.output)
    print(
        f"wrote {args.output}; line RMS "
        f"{calibration.rms_line_error_before_px:.3f}px -> "
        f"{calibration.rms_line_error_after_px:.3f}px"
    )


if __name__ == "__main__":
    main()
