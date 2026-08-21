#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.io import atomic_write_csv, read_manifest
from cowarch.passages import aggregate_passages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate frame-level geometry into auditable cow passages."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--min-valid-frames", type=int, default=3)
    parser.add_argument(
        "--allow-legacy-auto",
        action="store_true",
        help="Explicitly aggregate historical fixed-trim auto_sagitta data.",
    )
    args = parser.parse_args()

    frames = read_manifest(args.manifest)
    passages = aggregate_passages(
        frames,
        min_quality=args.min_quality,
        min_valid_frames=args.min_valid_frames,
        allow_legacy_auto=args.allow_legacy_auto,
    )
    atomic_write_csv(passages, args.output)
    eligible = int(passages["score_eligible"].sum())
    print(f"wrote {len(passages)} passages ({eligible} score-eligible) to {args.output}")


if __name__ == "__main__":
    main()
