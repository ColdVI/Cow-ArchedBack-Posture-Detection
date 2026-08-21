"""Preflight checks for the one supervised v1 task: cow segmentation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .io import as_bool


def validate_detector_dataset(
    data_yaml: str | Path,
    metadata_csv: str | Path,
    *,
    min_images: int = 300,
) -> dict:
    """Require camera-owned, IR and behind-rail coverage before fine-tuning."""
    if min_images < 1:
        raise ValueError("min_images must be positive")
    data_yaml = Path(data_yaml)
    with data_yaml.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if missing := sorted({"train", "val", "names"} - set(config)):
        raise ValueError(f"Detector dataset YAML is missing keys: {missing}")
    names = config["names"]
    values = names.values() if isinstance(names, dict) else names
    if "cow" not in {str(value).strip().lower() for value in values}:
        raise ValueError("Detector dataset must contain a class named cow")

    metadata = pd.read_csv(metadata_csv, keep_default_na=False)
    required = {"image", "is_ir", "behind_rails"}
    if missing := sorted(required - set(metadata.columns)):
        raise ValueError(f"Detector metadata is missing columns: {missing}")
    n_images = int(metadata["image"].astype(str).str.strip().replace("", pd.NA).nunique())
    if n_images < min_images:
        raise ValueError(f"Detector dataset needs at least {min_images} labeled images; found {n_images}")
    n_ir = int(as_bool(metadata["is_ir"]).sum())
    n_rails = int(as_bool(metadata["behind_rails"]).sum())
    if n_ir == 0:
        raise ValueError("Detector dataset must include labeled IR/night frames")
    if n_rails == 0:
        raise ValueError("Detector dataset must include cows behind corridor rails")
    return {"n_images": n_images, "n_ir": n_ir, "n_behind_rails": n_rails}
