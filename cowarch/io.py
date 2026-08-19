from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "source_id",
    "video_id",
    "crop_path",
    "accepted",
}


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path, keep_default_na=False)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    return df


def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def resolve_data_path(value: str, manifest_path: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(manifest_path).resolve().parent / path


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})

