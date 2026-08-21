#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.embeddings import extract_resnet18_embeddings
from cowarch.geometry import decode_keypoints, keypoint_features
from cowarch.io import as_bool, atomic_write_csv, read_manifest, resolve_data_path
from cowarch.modeling import binary_metrics, train_with_validation
from cowarch.splits import assert_no_group_leakage, require_complete_groups


KEYPOINT_FEATURES = [
    "kp_sagitta_norm",
    "kp_mean_deviation_norm",
    # Signed, normalized deviation is what lets the baseline distinguish an
    # upward arch from an equally large downward sag. Pixel-height fields stay
    # available for inspection but are excluded here to avoid scale leakage.
    "kp_mean_signed_deviation_norm",
    "kp_quad_peak_norm",
    "kp_back_deflection_deg",
    "kp_menger_curvature_norm",
]

AUTO_FEATURES = [
    "auto_sagitta",
    "auto_chord_rmse",
    "auto_circle_curvature_norm",
]


def add_keypoint_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for value in frame.get("keypoints_json", pd.Series("", index=frame.index)):
        points = decode_keypoints(value)
        if points is None:
            rows.append({name: np.nan for name in KEYPOINT_FEATURES})
            continue
        try:
            features = keypoint_features(points)
            rows.append({name: features[name] for name in KEYPOINT_FEATURES})
        except ValueError:
            rows.append({name: np.nan for name in KEYPOINT_FEATURES})
    features_frame = pd.DataFrame(rows, index=frame.index)
    return pd.concat([frame, features_frame], axis=1)


def embeddings_for_frame(
    frame: pd.DataFrame,
    manifest: Path,
    cache_path: Path,
    batch_size: int,
    device: str,
) -> np.ndarray:
    sample_ids = frame["sample_id"].astype(str).to_numpy()
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["sample_ids"].astype(str)
        if np.array_equal(sample_ids, cached_ids):
            print(f"using embedding cache: {cache_path}")
            return cached["embeddings"].astype(np.float32)

    paths = [resolve_data_path(value, manifest) for value in frame["crop_path"].astype(str)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing crop files, first entries: {missing[:5]}")
    embeddings = extract_resnet18_embeddings(paths, batch_size=batch_size, device=device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, sample_ids=sample_ids, embeddings=embeddings)
    return embeddings


def ensure_split_classes(frame: pd.DataFrame, mask: np.ndarray, model_name: str) -> None:
    subset = frame.loc[mask]
    for split in ("train", "val", "test"):
        labels = set(subset.loc[subset["split"].eq(split), "target"].astype(int))
        if labels != {0, 1}:
            raise ValueError(
                f"{model_name}: {split} needs both classes after feature filtering; found {sorted(labels)}"
            )


def has_split_classes(frame: pd.DataFrame, mask: np.ndarray) -> bool:
    subset = frame.loc[mask]
    return all(
        set(subset.loc[subset["split"].eq(split), "target"].astype(int)) == {0, 1}
        for split in ("train", "val", "test")
    )


def train_one(
    name: str,
    frame: pd.DataFrame,
    matrix: np.ndarray,
    usable_mask: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    c_grid: list[float],
    metadata: dict,
) -> tuple[dict, pd.DataFrame]:
    ensure_split_classes(frame, usable_mask, name)
    train_mask = usable_mask & frame["split"].eq("train").to_numpy()
    val_mask = usable_mask & frame["split"].eq("val").to_numpy()
    test_mask = usable_mask & frame["split"].eq("test").to_numpy()

    trained = train_with_validation(
        matrix[train_mask],
        frame.loc[train_mask, "target"].to_numpy(dtype=int),
        matrix[val_mask],
        frame.loc[val_mask, "target"].to_numpy(dtype=int),
        c_grid=c_grid,
    )
    val_probability = trained.pipeline.predict_proba(matrix[val_mask])[:, 1]
    test_probability = trained.pipeline.predict_proba(matrix[test_mask])[:, 1]
    metrics = {
        "model": name,
        "c_value": trained.c_value,
        "threshold_selected_on_validation": trained.threshold,
        "validation": binary_metrics(
            frame.loc[val_mask, "target"].to_numpy(dtype=int),
            val_probability,
            trained.threshold,
        ),
        "test": binary_metrics(
            frame.loc[test_mask, "target"].to_numpy(dtype=int),
            test_probability,
            trained.threshold,
        ),
        "feature_count": int(matrix.shape[1]),
        "feature_names": feature_names,
        **metadata,
    }

    bundle = {
        "pipeline": trained.pipeline,
        "threshold": trained.threshold,
        "c_value": trained.c_value,
        "feature_names": feature_names,
        "positive_label": "arched",
        "negative_label": "normal",
        "metadata": metadata,
    }
    joblib.dump(bundle, output_dir / f"{name}.joblib")

    prediction_parts = []
    for split_name, split_mask, probability in [
        ("val", val_mask, val_probability),
        ("test", test_mask, test_probability),
    ]:
        part = frame.loc[
            split_mask,
            ["sample_id", "source_id", "video_id", "split", "label", "target", "crop_path"],
        ].copy()
        part["model"] = name
        part["probability_arched"] = probability
        part["prediction"] = np.where(probability >= trained.threshold, "arched", "normal")
        part["threshold"] = trained.threshold
        prediction_parts.append(part)
    return metrics, pd.concat(prediction_parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train geometry, frozen-embedding, and fusion baselines.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["geometry", "embedding", "fusion"],
        default=["geometry", "embedding", "fusion"],
    )
    parser.add_argument("--allow-auto-geometry", action="store_true")
    parser.add_argument("--group-column", default="cow_id")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    args = parser.parse_args()

    frame = read_manifest(args.manifest)
    needed = {"split", "label", args.group_column}
    if missing := sorted(needed - set(frame.columns)):
        raise ValueError(f"Manifest is missing columns: {missing}")
    accepted = as_bool(frame["accepted"])
    frame = frame.loc[accepted & frame["label"].isin(["normal", "arched"])].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No accepted normal/arched labels were found")
    require_complete_groups(frame, args.group_column)
    if set(frame["split"]) - {"train", "val", "test"}:
        raise ValueError("Every training row must have train/val/test split")
    assert_no_group_leakage(frame, args.group_column)
    frame["target"] = frame["label"].map({"normal": 0, "arched": 1}).astype(int)
    frame = add_keypoint_features(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_needed = any(name in args.models for name in ["embedding", "fusion"])
    embeddings = None
    if embedding_needed:
        embeddings = embeddings_for_frame(
            frame,
            args.manifest,
            args.output_dir / "embedding_cache.npz",
            args.batch_size,
            args.device,
        )

    keypoint_mask = frame[KEYPOINT_FEATURES].notna().all(axis=1).to_numpy()
    geometry_source = None
    geometry_features: list[str] = []
    geometry_matrix = None
    if keypoint_mask.any() and has_split_classes(frame, keypoint_mask):
        geometry_source = "five_dorsal_keypoints"
        geometry_features = KEYPOINT_FEATURES
        geometry_matrix = frame[geometry_features].to_numpy(dtype=np.float32)
        geometry_mask = keypoint_mask
    elif args.allow_auto_geometry:
        if missing := sorted(set(AUTO_FEATURES) - set(frame.columns)):
            raise ValueError(f"Auto geometry requested but columns are missing: {missing}")
        geometry_source = "experimental_fixed_trim_silhouette"
        geometry_features = AUTO_FEATURES
        geometry_matrix = frame[geometry_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        geometry_mask = np.isfinite(geometry_matrix).all(axis=1)
        if not has_split_classes(frame, geometry_mask):
            raise ValueError(
                "Experimental auto geometry does not contain both classes in every split."
            )
    else:
        geometry_mask = np.zeros(len(frame), dtype=bool)

    if any(name in args.models for name in ["geometry", "fusion"]) and geometry_source is None:
        raise ValueError(
            "Geometry/fusion requested but complete five-keypoint rows with both classes "
            "in every split do not exist. "
            "Annotate keypoints or explicitly pass --allow-auto-geometry."
        )

    all_metrics = {}
    prediction_frames = []
    for name in args.models:
        if name == "geometry":
            assert geometry_matrix is not None
            matrix = geometry_matrix
            usable = geometry_mask
            feature_names = geometry_features
            metadata = {"geometry_source": geometry_source}
        elif name == "embedding":
            assert embeddings is not None
            matrix = embeddings
            usable = np.ones(len(frame), dtype=bool)
            feature_names = [f"resnet18_{index:03d}" for index in range(embeddings.shape[1])]
            metadata = {"backbone": "torchvision_resnet18_default_frozen"}
        else:
            assert embeddings is not None and geometry_matrix is not None
            matrix = np.concatenate([geometry_matrix, embeddings], axis=1)
            usable = geometry_mask
            feature_names = geometry_features + [f"resnet18_{index:03d}" for index in range(embeddings.shape[1])]
            metadata = {
                "geometry_source": geometry_source,
                "backbone": "torchvision_resnet18_default_frozen",
            }

        metrics, predictions = train_one(
            name,
            frame,
            matrix,
            usable,
            feature_names,
            args.output_dir,
            args.c_grid,
            metadata,
        )
        all_metrics[name] = metrics
        prediction_frames.append(predictions)
        print(
            f"{name}: test PR-AUC={metrics['test']['pr_auc']:.3f}, "
            f"recall={metrics['test']['sensitivity_recall']:.3f}, "
            f"specificity={metrics['test']['specificity']:.3f}"
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    atomic_write_csv(predictions, args.output_dir / "predictions.csv")
    run_config = {
        "manifest": str(args.manifest.resolve()),
        "models": args.models,
        "group_column": args.group_column,
        "c_grid": args.c_grid,
        "allow_auto_geometry": args.allow_auto_geometry,
        "labeled_samples": int(len(frame)),
        "groups": int(frame[args.group_column].nunique()),
        "split_counts": frame["split"].value_counts().to_dict(),
        "label_counts": frame["label"].value_counts().to_dict(),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)
    print(f"saved run to {args.output_dir}")


if __name__ == "__main__":
    main()
