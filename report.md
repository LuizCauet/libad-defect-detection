# Defect inspection: findings

## Experiment 1: full-dataset PatchCore reference

Luiz ran the existing LIBAD implementation with a frozen DINOv2-small feature extractor. The model built its normal memory bank from 252 normal samples across all eleven categories. Evaluation used 492 samples: 109 normal and 383 defective.

Configuration: visible-light sides A/B, max-side map combination, 640 × 512 input, float32 extraction, batch size one, 5% farthest-point-sampling coreset, and single-modality weighted-sum scoring. The bank contains 40,824 vectors of 384 numbers each, selected from 816,480 patch features. The image score averages the highest-scoring 1% of the combined map values.

Official split: `LIBAD_normal70train_noval_seed7345.csv`. Reference revision: `8f98f2ebde63cd45660e58d6413f5ccd638966f4`. DINOv2-small differs from the default gated DINOv3 backbone; this is not a reproduction of the paper's headline result.

### Ranking results

- Image area under the receiver operating characteristic curve (AUROC): **0.8491**.
- Average precision, labeled image AUPR by the reference: **0.9535**.
- Evaluation defect prevalence: **77.85%**. Precision on this defect-heavy set does not establish precision on a production stream with rare defects.

### Threshold experiment

We re-evaluated the saved predictions using higher thresholds. We did not retrain DINOv2, rebuild the bank, or rerun inference. Only the rule for converting a score to a flag changed: **flag when score is greater than or equal to the threshold**.

The reference selected the first threshold to maximize F1 on these evaluation labels. F1 combines precision and recall. The additional fixed thresholds of 11 through 15 are illustrative score cutoffs, not approved operating settings. The 95%-recall row is calculated from the evaluation receiver operating characteristic curve.

| Setting | Threshold | Defects caught | Defects missed | Good falsely flagged | Good correctly passed | Recall | FPR | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference best-F1 threshold | 10.281132 | 374 | 9 | 82 | 27 | 97.65% | 75.23% | 82.02% |
| At least 95% recall (diagnostic) | 10.465415 | 366 | 17 | 77 | 32 | 95.56% | 70.64% | 82.62% |
| Fixed threshold 11 | 11.000000 | 343 | 40 | 54 | 55 | 89.56% | 49.54% | 86.40% |
| Fixed threshold 12 | 12.000000 | 298 | 85 | 24 | 85 | 77.81% | 22.02% | 92.55% |
| Fixed threshold 13 | 13.000000 | 235 | 148 | 12 | 97 | 61.36% | 11.01% | 95.14% |
| Fixed threshold 14 | 14.000000 | 190 | 193 | 5 | 104 | 49.61% | 4.59% | 97.44% |
| Fixed threshold 15 | 15.000000 | 141 | 242 | 1 | 108 | 36.81% | 0.92% | 99.30% |

The table displays rounded thresholds; CSV outputs retain full numerical precision. Recall is the fraction of 383 defects caught. False-positive rate (FPR) is the fraction of 109 normal samples falsely flagged. Precision is the fraction of alerts that are real defects.

### Interpretation

Increasing the threshold reduces false alarms but also misses more defects. The threshold changes decisions; it does not improve the score ranking, so AUROC and average precision remain unchanged.

At the reference threshold, the system catches 374 defects and misses 9, but falsely flags 82 good samples. That is 97.65% recall with 75.23% FPR. At the diagnostic threshold reaching at least 95% recall, the measured FPR is **70.64%** and realized recall is **95.56%**.

A suitable operating point requires a stated limit on missed defects and false alarms. No such factory requirement has been supplied. None of these thresholds has been selected on independent validation data.

### Runtime from Luiz's supplied console output

The reference timed 491 samples after excluding its first batch. Mean feature-extraction/scoring time was 156.287 milliseconds per sample, or 6.398 samples per second. This excludes image loading, resizing, and heatmap saving; it is not full service latency or a p95 measurement. Runtime varies with system load.

### Earlier wrinkling-only experiment

The earlier run used 27 normal training samples and 49 evaluation samples from wrinkling only. AUROC was 0.6962 and FPR at at least 95% recall was 100%. The new aggregate AUROC is higher, but both training coverage and evaluation composition changed. This is not a controlled demonstration that adding normal data improved every defect category.

Visual inspection of the good sample A32 showed a heatmap response along a rectangular texture boundary. This supports investigating sensitivity to harmless texture boundaries. It does not establish why the response occurred. Each saved PNG has its own color normalization, so colors cannot be compared as absolute scores between images.

### Limits and next decisions

- These are development findings from one official split. We have inspected the evaluation results; do not describe this set as untouched final evaluation after tuning against it.
- The supplied alternative validation protocol has only normal samples. We must agree on labeled defect development data and an independent final evaluation plan before selecting our trainable model and threshold.
- Per-defect evaluation remains to be completed. Aggregate performance can hide failures in particular defect categories.
- Pixel-level metrics are unavailable because this evaluation path supplied no ground-truth masks. Heatmaps are qualitative evidence only.
- No change to the configured threshold or model files was made by this analysis. The selected table rows are comparisons, not a deployment recommendation.

## Reproduce the threshold analysis

From the project root:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_thresholds.py
```

This command regenerates this report and the analysis tables from the saved run. Add later experiment narratives to separate reports or update the generator to preserve them.

Files:

- `reports/threshold_tradeoffs.csv`: the selected table rows.
- `reports/threshold_sweep.csv`: counts and rates at every distinct saved score, plus an above-maximum threshold that flags nothing.
- `reports/threshold_analysis.json`: input score-file checksum, sample counts, computed ranking metrics, and diagnostic FPR95.
- `artifacts/patchcore_vis_all_seed7345/`: original run settings, scores, bank, and heatmaps.

Validation: checked unique sample identifiers, finite scores, matching reference AUROC and best-F1 threshold, and matching original confusion counts. Values are computed from saved scores at the float32 precision used by the reference.
