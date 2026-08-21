# Keypoint Model Decision (T1)

Status: **pending zero-shot run; no model decision has been fabricated.**

The five field videos required by the plan are present under
`/Users/anil/Downloads/kambur ve topal (1..5).mp4`. The repository environment
is Python 3.13, while the current DeepLabCut installation guide supports Python
3.10–3.12. DeepLabCut is consequently not installed into the main environment.

Run the bounded trial in a separate Python 3.12 environment:

```bash
python3.12 -m venv .venv-dlc
source .venv-dlc/bin/activate
pip install -r requirements-superanimal.txt
python scripts/09_superanimal_trial.py \
  "/Users/anil/Downloads/kambur ve topal (1).mp4" \
  "/Users/anil/Downloads/kambur ve topal (2).mp4" \
  "/Users/anil/Downloads/kambur ve topal (3).mp4" \
  "/Users/anil/Downloads/kambur ve topal (4).mp4" \
  "/Users/anil/Downloads/kambur ve topal (5).mp4" \
  --output-dir outputs/keypoint_trial
```

The script prints and validates the 39-keypoint model configuration before
inference. Review the labeled videos specifically at the withers, sacrum and
head regions, then record exactly one outcome:

1. Anatomically stable: use SuperAnimal transfer learning with a new three-point
   decoder.
2. Close but drifting: use the zero-shot output only as annotation proposals,
   correct them, and train `YOLO11m-pose` in `withers,sacrum,head` order.
3. Implausible: label directly and train `YOLO11m-pose`.

Until that review is completed, `cowarch.anchors.predict_anchors()` provides the
framework-neutral contract and refuses to invent a default model. A production
YOLO checkpoint is also rejected unless it has exactly three keypoints.

References: [DeepLabCut SuperAnimal example](https://deeplabcut.github.io/DeepLabCut/examples/COLAB/COLAB_YOURDATA_SuperAnimal.html),
[DeepLabCut installation support](https://deeplabcut.github.io/DeepLabCut/docs/installation.html).
