# Score Calibration

Status: **waiting for real dual-observer passage scores**.

No `score_observations.csv` is present in the repository, so the following
values cannot be honestly reported yet:

| Required result | Current value |
|---|---|
| Jointly scored passages | pending (target: about 100–150) |
| Quadratic weighted kappa | pending |
| Sagitta–score Spearman correlation | pending |
| Data-derived band boundaries | pending |

The implementation is `cowarch/calibration.py` and
`scripts/15_calibrate_scores.py`. Input must contain exactly two independent
observers for every included passage:

```csv
passage_id,observer_id,score
passage_0001,observer_a,1
passage_0001,observer_b,1
```

Run:

```bash
python scripts/15_calibrate_scores.py \
  --passages data/passages_after.csv \
  --observations data/score_observations.csv \
  --output-dir outputs/score_calibration \
  --report docs/SCORE_CALIBRATION.md
```

The command replaces this status page with measured kappa, correlation and
isotonic data-derived boundaries. It fails instead of inventing thresholds when
observer coverage, score variation or sagitta–score direction is insufficient.
