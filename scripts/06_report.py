#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cow_arch_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.geometry import DORSAL_KEYPOINTS, decode_keypoints, keypoint_features
from cowarch.io import as_bool, read_manifest, resolve_data_path
from cowarch.modeling import (
    aggregate_group_predictions,
    evaluate_frame_and_group_metrics,
)


def safe_float(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(number) else f"{number:.3f}"


def safe_interval(payload: dict) -> str:
    return f"[{safe_float(payload.get('lower'))}, {safe_float(payload.get('upper'))}]"


def attach_evaluation_groups(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Resolve the selected evaluation group from the source manifest."""

    if "sample_id" not in predictions.columns:
        raise ValueError("predictions.csv is missing sample_id")
    if group_column == "sample_id":
        return predictions.copy()
    if group_column not in manifest.columns:
        if group_column in predictions.columns:
            return predictions.copy()
        raise ValueError(f"Manifest is missing evaluation group column: {group_column}")

    mapping = manifest[["sample_id", group_column]].copy()
    conflicting = mapping.groupby("sample_id", dropna=False)[group_column].nunique(
        dropna=False
    )
    if bool((conflicting > 1).any()):
        sample_ids = conflicting[conflicting > 1].index.astype(str).tolist()[:5]
        raise ValueError(
            f"Manifest maps sample_id to conflicting {group_column} values: {sample_ids}"
        )
    mapping = mapping.drop_duplicates("sample_id", keep="first")
    base = predictions.drop(columns=[group_column], errors="ignore")
    joined = base.merge(mapping, on="sample_id", how="left", validate="many_to_one")
    return joined


def grouped_evaluation(
    predictions: pd.DataFrame,
    metrics: dict,
    group_column: str,
    aggregation: str,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Build structured frame/group metrics and group-level prediction rows."""

    evaluation: dict = {}
    group_rows: list[pd.DataFrame] = []
    for model in metrics:
        test = predictions[
            (predictions["model"] == model) & (predictions["split"] == "test")
        ].copy()
        if test.empty:
            raise ValueError(f"No test predictions found for model: {model}")
        thresholds = pd.to_numeric(test["threshold"], errors="raise").unique()
        if len(thresholds) != 1:
            raise ValueError(
                f"Model {model} has multiple test thresholds: {thresholds.tolist()}"
            )
        threshold = float(thresholds[0])
        summary = evaluate_frame_and_group_metrics(
            test["target"].to_numpy(dtype=int),
            test["probability_arched"].to_numpy(dtype=float),
            test[group_column].to_numpy(),
            threshold,
            aggregation=aggregation,
            n_bootstrap=n_bootstrap,
            random_state=bootstrap_seed,
        )
        summary["group_column"] = group_column
        evaluation[model] = summary

        grouped = aggregate_group_predictions(
            test["target"].to_numpy(dtype=int),
            test["probability_arched"].to_numpy(dtype=float),
            test[group_column].to_numpy(),
            aggregation=aggregation,
        )
        group_frame = pd.DataFrame(
            {
                "model": model,
                group_column: grouped.group_ids,
                "target": grouped.y_true,
                "label": np.where(grouped.y_true == 1, "arched", "normal"),
                "probability_arched": grouped.probability,
                "prediction": np.where(
                    grouped.probability >= threshold, "arched", "normal"
                ),
                "threshold": threshold,
                "frame_count": grouped.frame_counts,
                "probability_aggregation": aggregation,
            }
        )
        group_rows.append(group_frame)
    return evaluation, pd.concat(group_rows, ignore_index=True)


def save_curves(predictions: pd.DataFrame, model: str, figures: Path) -> None:
    test = predictions[(predictions["model"] == model) & (predictions["split"] == "test")]
    y_true = test["target"].astype(int).to_numpy()
    probability = test["probability_arched"].astype(float).to_numpy()
    prediction = test["prediction"].map({"normal": 0, "arched": 1}).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    if len(np.unique(y_true)) == 2:
        PrecisionRecallDisplay.from_predictions(y_true, probability, ax=axes[0], name=model)
        RocCurveDisplay.from_predictions(y_true, probability, ax=axes[1], name=model)
    else:
        axes[0].text(0.5, 0.5, "PR curve unavailable\n(single test class)", ha="center", va="center")
        axes[1].text(0.5, 0.5, "ROC curve unavailable\n(single test class)", ha="center", va="center")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        prediction,
        labels=[0, 1],
        display_labels=["normal", "arched"],
        colorbar=False,
        ax=axes[2],
    )
    fig.suptitle(f"{model} — frame-level independent test split")
    fig.tight_layout()
    fig.savefig(figures / f"{model}_evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_geometry_histogram(manifest: pd.DataFrame, figures: Path) -> str | None:
    rows = []
    for _, row in manifest.iterrows():
        if row.get("label", "") not in {"normal", "arched"}:
            continue
        points = decode_keypoints(row.get("keypoints_json", ""))
        if points is None:
            continue
        try:
            feature = keypoint_features(points)["kp_sagitta_norm"]
        except ValueError:
            continue
        rows.append({"label": row["label"], "kp_sagitta_norm": feature})
    if not rows:
        return None
    values = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(values, x="kp_sagitta_norm", hue="label", element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title("Five-keypoint normalized sagitta")
    fig.tight_layout()
    name = "keypoint_sagitta_histogram.png"
    fig.savefig(figures / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return name


def save_keypoint_overlays(
    manifest: pd.DataFrame,
    manifest_path: Path,
    figures: Path,
    limit: int = 12,
) -> list[str]:
    overlay_dir = figures / "keypoint_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    names = []
    if "keypoints_json" not in manifest.columns:
        return []
    eligible = manifest[manifest["keypoints_json"].astype(str).str.len() > 2]
    for _, row in eligible.head(limit).iterrows():
        points = decode_keypoints(row.get("keypoints_json", ""))
        if points is None:
            continue
        path = resolve_data_path(str(row["crop_path"]), manifest_path)
        if not path.exists():
            continue
        with Image.open(path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.line([tuple(point) for point in points], fill=(255, 215, 0), width=max(2, image.width // 250))
        radius = max(4, image.width // 160)
        for index, (name, point) in enumerate(zip(DORSAL_KEYPOINTS, points), start=1):
            x, y = float(point[0]), float(point[1])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 80, 40))
            draw.text((x + radius + 2, y - radius), f"{index}:{name}", fill=(255, 255, 255))
        filename = f"{row['sample_id']}.jpg"
        image.save(overlay_dir / filename, quality=92)
        names.append(f"keypoint_overlays/{filename}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an honest Markdown report and evaluation figures."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--group-column",
        default=None,
        help=(
            "Evaluation cluster column, typically video_id, cow_id, or passage_id "
            "(default: run_config group_column, then video_id)."
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median", "max"],
        default="mean",
        help="How to aggregate frame probabilities inside each group.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    with (args.run_dir / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (args.run_dir / "run_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    predictions = pd.read_csv(args.run_dir / "predictions.csv")
    manifest = read_manifest(args.manifest)
    accepted = as_bool(manifest["accepted"])
    group_column = args.group_column or config.get("group_column") or "video_id"
    predictions = attach_evaluation_groups(predictions, manifest, group_column)
    evaluation, group_predictions = grouped_evaluation(
        predictions,
        metrics,
        group_column,
        args.aggregation,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    with (args.run_dir / "group_evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, ensure_ascii=False)
    group_predictions.to_csv(args.run_dir / "test_group_predictions.csv", index=False)

    figures = args.run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for model in metrics:
        save_curves(predictions, model, figures)
    histogram = save_geometry_histogram(manifest.loc[accepted], figures)
    overlays = save_keypoint_overlays(manifest.loc[accepted], args.manifest, figures)

    test_predictions = predictions[predictions["split"] == "test"].copy()
    test_predictions["error"] = test_predictions["label"] != test_predictions["prediction"]
    test_predictions[test_predictions["error"]].to_csv(args.run_dir / "test_errors.csv", index=False)

    lines = [
        "# Arched-Back Posture PoC Report",
        "",
        "> This system detects a human-defined visual arched-back posture label. It does not diagnose lameness or disease.",
        "",
        "## Dataset contract",
        "",
        f"- Accepted prepared samples: {int(accepted.sum())}",
        f"- Labeled binary samples used: {config['labeled_samples']}",
        f"- Training split groups (`{config.get('group_column', 'video_id')}`): {config['groups']}",
        f"- Evaluation group column: `{group_column}`",
        f"- Group probability aggregation: `{args.aggregation}`",
        f"- Confidence intervals: {args.bootstrap_samples} cluster-bootstrap draws resampling `{group_column}` groups, never individual frames.",
        f"- Split counts: `{config['split_counts']}`",
        f"- Label counts: `{config['label_counts']}`",
        f"- Training split unit: `{config.get('group_column', 'video_id')}`; cross-split group leakage is rejected by the training script.",
        "",
        "## Final test results — frame level",
        "",
        "Frame metrics are retained for error analysis, but their confidence intervals resample whole groups.",
        "",
        "| Model | Frames | Groups | PR-AUC | Cluster 95% CI | ROC-AUC | Recall | Specificity | Precision | F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in metrics:
        test = evaluation[model]["frame"]
        interval = evaluation[model]["cluster_bootstrap"]["frame"]
        lines.append(
            f"| {model} | {test['n']} | {test['n_groups']} | {safe_float(test['pr_auc'])} | "
            f"{safe_interval(interval)} | {safe_float(test['roc_auc'])} | "
            f"{safe_float(test['sensitivity_recall'])} | {safe_float(test['specificity'])} | "
            f"{safe_float(test['precision'])} | {safe_float(test['f1'])} | "
            f"{safe_float(test['balanced_accuracy'])} |"
        )
    lines += [
        "",
        f"## Final test results — `{group_column}` level",
        "",
        f"Each group contributes one `{args.aggregation}` probability and must have exactly one ground-truth label.",
        "",
        "| Model | Groups | Frames | PR-AUC | Cluster 95% CI | ROC-AUC | Recall | Specificity | Precision | F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in metrics:
        test = evaluation[model]["group"]
        interval = evaluation[model]["cluster_bootstrap"]["group"]
        lines.append(
            f"| {model} | {test['n_groups']} | {test['n_frames']} | {safe_float(test['pr_auc'])} | "
            f"{safe_interval(interval)} | {safe_float(test['roc_auc'])} | "
            f"{safe_float(test['sensitivity_recall'])} | {safe_float(test['specificity'])} | "
            f"{safe_float(test['precision'])} | {safe_float(test['f1'])} | "
            f"{safe_float(test['balanced_accuracy'])} |"
        )
    lines += ["", "## Evaluation figures", ""]
    for model in metrics:
        lines.append(f"![{model} evaluation](figures/{model}_evaluation.png)")
        lines.append("")
    if histogram:
        lines += ["## Geometry sanity check", "", f"![Sagitta histogram](figures/{histogram})", ""]
    if overlays:
        lines += ["## Keypoint overlay samples", ""]
        for overlay in overlays[:6]:
            lines.append(f"![Keypoint overlay](figures/{overlay})")
            lines.append("")
    lines += [
        "## Mandatory interpretation limits",
        "",
        "- Ground truth is the project's visual posture annotation unless a veterinarian independently labels the data.",
        "- Back arch is associated with lameness but is neither specific nor sufficient for a clinical diagnosis.",
        "- Internet or educational-video sampling does not represent natural herd prevalence.",
        "- Source/camera/breed/background shift has not been eliminated by one internal test split.",
        "- Fixed-percentage silhouette geometry, when enabled, is experimental and not an anatomical Poursaberi replication.",
        "- `source_score` is auxiliary metadata; it must not be presented as frame-level clinical ground truth.",
        "",
        "## Next evidence needed",
        "",
        "1. Second rater or veterinarian agreement on a blinded subset.",
        "2. External test footage from a new camera/farm.",
        "3. Head and hoof trajectories for true temporal lameness modelling.",
        "4. Temporal validation and false alarms per passage/time unit.",
    ]
    (args.run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report: {args.run_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
