#!/usr/bin/env python3
"""Build data/sources.csv rows for the public datasets under data/external/.

Run after downloading each source. Idempotent: re-running regenerates the file
from what is actually on disk, so it never drifts from the real download state.

Each CattleLameness clip becomes its own source_id/video_id, matching how the
rest of the pipeline groups by video for the leakage-safe split. The two image
collections (Mendeley side/back views, Wagyu/Angus) are each one image_dir
source; 02_prepare.py's own detector-based filtering, not this script, decides
which individual frames are usable lateral crops.

The Livestock Keypoint Detection cattle subset is intentionally NOT added here.
Its 18-point scheme and mixed-view crops don't match this project's 5-point
dorsal protocol or the side-view requirement; per LITERATURE_NOTES.md it is a
pose-pretraining asset, not an arched-back labeling input.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
OUT = ROOT / "data" / "sources.csv"

ROWS: list[dict] = []


def add(source_id, kind, local_path, license_, source_score, notes):
    # Absolute paths, matching 01_collect.py's own convention: 02_prepare.py
    # joins a relative local_path onto sources.csv's *parent directory*
    # (repo_root/data), so a path already written as "data/..." would
    # double up into "data/data/...". Writing it resolved sidesteps that.
    ROWS.append(
        {
            "source_id": source_id,
            "kind": kind,
            "url": "",
            "local_path": str(local_path.resolve()),
            "license": license_,
            "source_score": source_score,
            "notes": notes,
        }
    )


def add_cattle_lameness() -> int:
    base = EXTERNAL / "CattleLameness" / "Data"
    count = 0
    for weak_class, subdir in [("lame_weak_label", "Lame"), ("normal_weak_label", "Normal")]:
        folder = base / subdir
        if not folder.exists():
            continue
        for video in sorted(folder.glob("*.mp4")):
            clean_stem = re.sub(r"[()]", "", video.stem).strip().replace(" ", "_")
            source_id = f"cattlelameness_{subdir.lower()}_{clean_stem}"
            add(
                source_id=source_id,
                kind="video",
                local_path=video,
                license_="check-before-use",  # no LICENSE file in the upstream repo
                source_score=weak_class,
                notes=(
                    "Weak label from video title/category, NOT frame-level arched-back "
                    "ground truth. Re-label posture by hand. github.com/fahimsohan/CattleLameness"
                ),
            )
            count += 1
    return count


def add_cattle_side_back_views() -> int:
    """One row per animal, side-view folder only.

    The Mendeley API groups files under two folder_id hashes: 55b368b1 (lateral
    photos, confirmed by eye) and c54ab6a2 (rear-facing photos, confirmed by
    eye - a rear view cannot show a dorsal arch by definition and is excluded
    here). Filenames "N.png" match 1:1 across both folders, i.e. one number is
    one animal. Splitting by file - not by the whole dataset as one row - gives
    the group-wise split ~70 independent groups instead of collapsing 72
    distinct cattle into a single video_id.
    """
    side_folder = EXTERNAL / "CattleSideBackViews" / "55b368b1"
    if not side_folder.exists():
        return 0
    count = 0
    for image_path in sorted(side_folder.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        cow_number = image_path.stem
        add(
            source_id=f"mendeley_h2s22wr5py_side_{cow_number}",
            kind="image",
            local_path=image_path,
            license_="CC-BY-4.0",
            source_score="",
            notes=(
                f"Horqin cattle #{cow_number}, side view only (back-view folder excluded - "
                "a rear photo cannot show a dorsal arch). No posture label. "
                "https://doi.org/10.17632/h2s22wr5py.3"
            ),
        )
        count += 1
    return count


def add_wagyu_angus() -> int:
    folder = EXTERNAL / "CattleLamenessImages" / "extracted" / "Wagyu&Angus" / "Folder 1"
    if not folder.exists():
        return 0
    add(
        source_id="mendeley_f4j83j77ng_wagyu_angus",
        kind="image_dir",
        local_path=folder,
        license_="CC-BY-4.0",
        source_score="",
        notes=(
            "277 Wagyu/Angus body-view photos, Tau Sa farm, South Africa. Raw camera "
            "frames (cow not pre-cropped) - side-view yield depends on 02_prepare "
            "detection. No posture label. https://doi.org/10.17632/f4j83j77ng.1"
        ),
    )
    return 1


def main() -> None:
    n1 = add_cattle_lameness()
    n2 = add_cattle_side_back_views()
    n3 = add_wagyu_angus()

    if not ROWS:
        print("Nothing downloaded yet under data/external/ - nothing to write.")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "kind", "url", "local_path", "license", "source_score", "notes"])
        writer.writeheader()
        writer.writerows(ROWS)

    print(f"CattleLameness clips:       {n1}")
    print(f"Cattle side/back (Mendeley): {n2}")
    print(f"Wagyu/Angus (Mendeley):      {n3}")
    print(f"wrote {len(ROWS)} rows to {OUT}")
    if n1 < 50:
        print("WARNING: expected 50 CattleLameness clips, found fewer - check the download.")


if __name__ == "__main__":
    main()
