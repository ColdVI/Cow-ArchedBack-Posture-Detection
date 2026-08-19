#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.embeddings import extract_resnet18_embeddings
from cowarch.geometry import decode_keypoints, keypoint_features
from cowarch.io import as_bool, atomic_write_csv, read_manifest, resolve_data_path


def geometry_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    rows = []
    for _, row in frame.iterrows():
        if names and names[0].startswith("kp_"):
            points = decode_keypoints(row.get("keypoints_json", ""))
            if points is None:
                rows.append([np.nan] * len(names))
                continue
            try:
                features = keypoint_features(points)
                rows.append([features[name] for name in names])
            except ValueError:
                rows.append([np.nan] * len(names))
        else:
            rows.append([pd.to_numeric(row.get(name, np.nan), errors="coerce") for name in names])
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained posture model on a newly prepared manifest.")
    parser.add_argument("--model", required=True, type=Path, help="A joblib bundle produced by 05_train.py")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Load only model bundles created by this project; joblib/pickle is not safe for untrusted files.
    bundle = joblib.load(args.model)
    frame = read_manifest(args.manifest)
    frame = frame.loc[as_bool(frame["accepted"])].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No accepted samples in inference manifest")

    feature_names = list(bundle["feature_names"])
    geometry_names = [name for name in feature_names if name.startswith(("kp_", "auto_"))]
    embedding_names = [name for name in feature_names if name.startswith("resnet18_")]
    blocks = []
    valid = np.ones(len(frame), dtype=bool)

    if geometry_names:
        geometry = geometry_matrix(frame, geometry_names)
        valid &= np.isfinite(geometry).all(axis=1)
        blocks.append(geometry)
    if embedding_names:
        paths = [resolve_data_path(value, args.manifest) for value in frame["crop_path"].astype(str)]
        embedding = extract_resnet18_embeddings(paths, batch_size=args.batch_size, device=args.device)
        if embedding.shape[1] != len(embedding_names):
            raise ValueError("Embedding dimension does not match the saved model contract")
        blocks.append(embedding)
    if not blocks:
        raise ValueError("Saved model contains no recognized feature block")

    matrix = np.concatenate(blocks, axis=1)
    if matrix.shape[1] != len(feature_names):
        raise ValueError("Feature order/dimension does not match the saved model contract")
    frame = frame.loc[valid].copy()
    matrix = matrix[valid]
    if frame.empty:
        raise ValueError("No samples have all features required by the model")

    probability = bundle["pipeline"].predict_proba(matrix)[:, 1]
    threshold = float(bundle["threshold"])
    frame["probability_arched"] = probability
    frame["prediction"] = np.where(probability >= threshold, "arched", "normal")
    frame["threshold"] = threshold
    atomic_write_csv(frame, args.output)

    summary = (
        frame.groupby("video_id", as_index=False)
        .agg(
            frames=("sample_id", "size"),
            median_probability_arched=("probability_arched", "median"),
            mean_probability_arched=("probability_arched", "mean"),
            positive_frame_fraction=("prediction", lambda values: float((values == "arched").mean())),
        )
    )
    summary["clip_prediction"] = np.where(
        summary["median_probability_arched"] >= threshold,
        "arched",
        "normal",
    )
    summary_path = args.output.with_name(args.output.stem + "_clips.csv")
    atomic_write_csv(summary, summary_path)
    print(f"wrote frame predictions: {args.output}")
    print(f"wrote clip summaries: {summary_path}")


if __name__ == "__main__":
    main()

