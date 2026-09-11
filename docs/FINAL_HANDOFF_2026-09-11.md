# Final Handoff — Cow Arched-Back / Posture Track

**Date:** 2026-09-11

This repository is the posture/geometry research track of the wider İnek-Kambur project. It should be handed over together with the private `ColdVI/cow-reid-pipeline` repository, which owns the identity/tracklet/gallery layer.

## Scope

Current v1 posture direction is not a generic “arched vs normal” image classifier. The main path measures dorsal geometry from a fixed lateral measurement camera and converts robust passage-level geometry into an absolute posture triage score after observer calibration.

Main flow:

```text
raw 30 fps side video
 -> camera distortion mitigation / center band
 -> cow detection + segmentation
 -> withers + sacrum + head keypoints
 -> head-down / quality rejection
 -> anchored dorsal profile
 -> anchored_sagitta_signed_norm
 -> robust passage aggregation
 -> observer-calibrated posture band
 -> retrospective treatment/calving association
```

The repository does **not** diagnose lameness or disease.

## What was implemented

- inspection-first image workflow and labeling utilities;
- multi-cow cropping and Label Studio bridge;
- external source manifest support;
- longitudinal posture monitoring scaffolding;
- absolute posture scoring v3;
- full-pose annotation workflow and improved annotation UI;
- measurement camera role/gating;
- plumb-line undistortion and center-band mitigation;
- `withers,sacrum,head` production keypoint contract;
- 101-point anchored dorsal profile;
- robust passage aggregation;
- observer calibration with weighted kappa, Spearman and isotonic bands;
- treatment/calving retrospective validation path;
- detector/keypoint metric gates and IR/rail slices;
- future baseline/CUSUM code retained but intentionally outside v1.

## Important files

- `README.md` — complete current pipeline and commands.
- `docs/IMPLEMENTATION_STATUS_V3.md` — code-complete vs external-evidence-required matrix.
- `docs/ANNOTATION_GUIDE.md` — labeling protocol.
- `docs/KEYPOINT_MODEL_DECISION.md` — keypoint model decision state.
- `docs/SCORE_CALIBRATION.md` — score calibration contract.
- `docs/FUTURE_SYSTEM.md` — explicitly unvalidated future work.
- `cowarch/anchors.py` — framework-neutral keypoint adapter contract.
- `cowarch/baseline.py` — deferred individual baseline logic.
- `scripts/02_prepare.py` — measurement data preparation.
- `scripts/10_aggregate_passages.py` — robust passage aggregation.
- `scripts/11_variance_study.py` — before/after camera mitigation study.
- `scripts/15_calibrate_scores.py` — observer-based absolute score calibration.
- `scripts/12_validate_retrospective.py` — treatment/calving retrospective validation.
- `scripts/13_train_detector.py` — detector fine-tuning and topline evaluation.
- `scripts/16_label_pose.py` — full-pose local labeling tool.
- `scripts/17_evaluate_keypoints.py` — three-point PCKh evaluation.
- `scripts/19_ingest_pose_media.py` — local media ingestion for pose labeling.

## Key research decisions

1. A posture score should not depend on one frame. Passage-level robust aggregation is the main unit.
2. Geometry should be anchored by anatomically meaningful points, not a fixed image crop.
3. Head position is a quality/filter signal, not part of the dorsal curvature metric.
4. Low-quality observations are retained with explicit rejection reasons instead of silently discarded.
5. Sagitta-to-score thresholds are not hardcoded; they are derived from two-observer ordinal labels.
6. Personal baseline/CUSUM and temporal BiLSTM/TCN are future research directions, not validated capabilities.
7. ReID belongs to the separate identity repository and should be used to connect passage scores across days.

## External evidence still missing

The code path existing does not mean the field claim is validated. Remaining real-world requirements include:

- dedicated/fixed measurement camera installation;
- real raw 30 fps lateral capture;
- camera straight-line calibration clicks and before/after variance study;
- farm keypoint labels including IR/rail/black-cow hard cases;
- trained production three-keypoint checkpoint;
- about 100–150 passages scored independently by two observers;
- treatment and calving records;
- multi-week repeated cow measurements;
- prospective validation before any alert/clinical claim.

See `docs/IMPLEMENTATION_STATUS_V3.md` for the formal task matrix.

## Keypoint model status

The repository's own T1 decision remains **pending zero-shot run**. Do not fabricate a winner. The documented decision tree is:

1. stable SuperAnimal landmarks -> transfer learning/new three-point decoder;
2. close but drifting -> use proposals, correct, train YOLO pose;
3. implausible -> directly label and train YOLO pose.

Separately, the private ReID repository contains an adapter for an external 15-point DeepLabCut model. That is a different integration path and must not be confused with this repo's T1 decision.

## Relationship to ReID

Longitudinal monitoring requires identity. The intended join is:

```text
video -> tracklet_id -> confirmed/predicted cow_id
                     -> keypoints -> passage posture score
cow_id + timestamp + score -> longitudinal history
```

The private ReID repo owns the identity confidence, UNKNOWN/AMBIGUOUS policy, debug review, gallery and cross-day identity evaluation.

## First tasks for the next team

1. Reproduce tests and smoke checks.
2. Decide/validate the keypoint model on real farm media.
3. Acquire proper side-view measurement data at original frame rate.
4. Run camera mitigation / variance study.
5. Complete observer calibration.
6. Integrate confirmed cow identity from the ReID pipeline.
7. Only then evaluate longitudinal baseline/change detection.

Do not present future-system items as completed features and do not present posture score as a clinical diagnosis.
