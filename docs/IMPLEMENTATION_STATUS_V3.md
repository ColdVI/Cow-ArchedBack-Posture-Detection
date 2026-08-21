# Implementation Plan v3 Status

| Task | Code status | External evidence still required |
|---|---|---|
| T0 camera | `camera_role` schema and measurement-only gate implemented | Dedicated camera purchase/install |
| T1 keypoint choice | Reproducible SuperAnimal trial and framework-neutral adapter implemented | Python 3.12 zero-shot run and visual decision |
| T2 protocol | Three-point annotation, confidence and head-down filters implemented | Annotated IR/rail coverage |
| T3 passage path | Anchored frame measurement and robust passage table implemented | Trained three-keypoint checkpoint; raw 30 fps field capture |
| T4 variance | Both readings and before/after camera comparison implemented | About 50 cows × 2–3 weeks measurement-camera data |
| T5 baseline/CUSUM | Historical code retained | Deferred outside v1 |
| T6 validation | Absolute score↔observed treatment correlation and calving suppression implemented | Farm treatment/calving records |
| T7 detector/keypoint metrics | YOLO11m-seg fine-tuning, anchored topline MAE, IR/rail slices and three-point PCKh gate implemented | Farm labels and training run |
| T8 report | Prevalence-reweighted PPV formula and mandatory rows implemented | Farm prevalence supplied to report |
| T9 calibration | Weighted kappa, correlation and data-derived bands implemented | Two observers scoring about 100–150 passages |
| Camera mitigation | Plumb-line fit, undistortion, centre-band gate and T4 comparison implemented | Clicked straight-line points and before/after field run |

No metric in this table is marked complete merely because its code path exists;
data- and hardware-dependent acceptance gates remain external work.
