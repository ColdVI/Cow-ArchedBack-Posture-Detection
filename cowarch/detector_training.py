"""Preflight checks for the one supervised v1 task: cow segmentation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
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


def topline_vertical_mae(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    *,
    withers_x: float,
    sacrum_x: float,
) -> dict[str, float]:
    """Mean vertical topline error inside the annotated anchor interval."""
    predicted = np.asarray(predicted_mask).astype(bool)
    truth = np.asarray(ground_truth_mask).astype(bool)
    if predicted.ndim != 2 or truth.ndim != 2 or predicted.shape != truth.shape:
        raise ValueError("predicted and ground-truth masks need the same 2-D shape")
    left = max(0, int(np.ceil(min(withers_x, sacrum_x))))
    right = min(truth.shape[1] - 1, int(np.floor(max(withers_x, sacrum_x))))
    length = abs(float(sacrum_x) - float(withers_x))
    if right <= left or length <= 1e-6:
        raise ValueError("withers_x and sacrum_x must define a valid mask interval")
    differences = []
    for x in range(left, right + 1):
        pred_y = np.flatnonzero(predicted[:, x])
        true_y = np.flatnonzero(truth[:, x])
        if len(pred_y) and len(true_y):
            differences.append(abs(float(pred_y[0]) - float(true_y[0])))
    if not differences:
        return {
            "topline_mae_px": float("nan"),
            "topline_mae_body_length_fraction": float("nan"),
            "topline_columns_compared": 0,
        }
    mae = float(np.mean(differences))
    return {
        "topline_mae_px": mae,
        "topline_mae_body_length_fraction": mae / length,
        "topline_columns_compared": len(differences),
    }


def evaluate_detector_topline(model, metadata_csv: str | Path) -> tuple[dict, pd.DataFrame]:
    """Run a trained segmenter and report the task-specific topline metric."""
    import cv2

    metadata_csv = Path(metadata_csv)
    metadata = pd.read_csv(metadata_csv, keep_default_na=False)
    required = {
        "image", "ground_truth_mask", "withers_x", "sacrum_x", "is_ir", "behind_rails"
    }
    if missing := sorted(required - set(metadata.columns)):
        raise ValueError(f"Topline evaluation metadata is missing columns: {missing}")
    rows = []
    for source in metadata.itertuples(index=False):
        image_path = Path(source.image)
        mask_path = Path(source.ground_truth_mask)
        if not image_path.is_absolute():
            image_path = metadata_csv.parent / image_path
        if not mask_path.is_absolute():
            mask_path = metadata_csv.parent / mask_path
        image = cv2.imread(str(image_path))
        truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or truth is None:
            raise FileNotFoundError(f"Unreadable evaluation image/mask: {image_path}, {mask_path}")
        prediction = model.predict(image, verbose=False)
        result = prediction[0] if prediction else None
        detected = bool(result is not None and result.masks is not None and len(result.masks.data))
        metrics = {
            "topline_mae_px": float("nan"),
            "topline_mae_body_length_fraction": float("nan"),
            "topline_columns_compared": 0,
        }
        if detected:
            confidence = result.boxes.conf.detach().cpu().numpy()
            selected = int(np.argmax(confidence)) if len(confidence) else 0
            mask = result.masks.data[selected].detach().cpu().numpy()
            mask = cv2.resize(
                mask.astype(np.float32),
                (truth.shape[1], truth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
            metrics = topline_vertical_mae(
                mask,
                truth > 0,
                withers_x=float(source.withers_x),
                sacrum_x=float(source.sacrum_x),
            )
        rows.append(
            {
                "image": str(source.image),
                "detected": detected,
                "is_ir": bool(str(source.is_ir).lower() in {"1", "true", "yes", "y"}),
                "behind_rails": bool(
                    str(source.behind_rails).lower() in {"1", "true", "yes", "y"}
                ),
                **metrics,
            }
        )
    details = pd.DataFrame(rows)

    def rate(mask: pd.Series) -> float:
        subset = details.loc[mask, "detected"]
        return float(subset.mean()) if len(subset) else float("nan")

    finite = pd.to_numeric(
        details["topline_mae_body_length_fraction"], errors="coerce"
    ).dropna()
    summary = {
        "n_eval_images": int(len(details)),
        "detection_rate_all": rate(pd.Series(True, index=details.index)),
        "detection_rate_ir": rate(details["is_ir"]),
        "detection_rate_behind_rails": rate(details["behind_rails"]),
        "mean_topline_mae_px": float(details["topline_mae_px"].mean()),
        "mean_topline_mae_body_length_fraction": float(finite.mean()) if len(finite) else float("nan"),
        "topline_target_fraction": 0.005,
        "topline_target_pass": bool(len(finite) and float(finite.mean()) < 0.005),
    }
    return summary, details
