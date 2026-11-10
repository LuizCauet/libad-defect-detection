# First PatchCore-style baseline — completed

## Configuration

- Existing LIBAD reference revision: `8f98f2ebde63cd45660e58d6413f5ccd638966f4`; upstream source unchanged.
- Public backbone: `facebook/dinov2-small`, cached revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- Wrinkling category, official seed 7345 no-validation split assignments; category-filtered memory bank and evaluation.
- Training: 27 normal samples. Evaluation: 49 samples (11 normal, 38 anomalous).
- Visible A/B sides processed independently, max-side aggregation, top-1%-patch average score, 640 × 512 input, fp32, batch size one.
- FPS coreset: 87,480 normal patch features reduced to 4,374 vectors of dimension 384; scalar normalization and L2 distance.
- Single-modality weighted-sum output, without OCSVM calibration.
- Python 3.12.10; PyTorch 2.7.1+cu126; torchvision 0.22.1+cu126; RTX 2060 6 GB. Exact package versions: `requirements-baseline-lock.txt`.

## Measured results

| Measurement | Result |
| --- | ---: |
| Image AUROC | 0.696172 |
| FPR at first ROC operating point reaching at least 95% recall | 100% |
| Realized recall at that point | 100% (38/38 defects) |
| False positives at that point | 11/11 normal samples |
| Threshold at that diagnostic point | 11.006611 |
| Reference inference-loop mean, excluding first batch | 261.632 ms/sample |
| Reference inference-loop throughput | 3.822 samples/second |
| Completed-run elapsed time | 49.35 seconds |
| Peak PyTorch allocated GPU memory | 325,029,376 bytes (~310 MiB) |
| Saved heatmap PNGs | 49 |

AUROC and the high-recall operating point were independently recomputed from the saved raw-score CSV using scikit-learn and agree with the reference. Verified 49 unique sample names, finite scores, and 49 heatmap files. Dependency consistency check passed.

## Interpretation and limitations

This configuration ranks defects above normal samples imperfectly and cannot reach 95% recall on this evaluation set without flagging every normal sample. That establishes an initial reference result, not evidence that all PatchCore configurations fail or that our future model will improve it.

Only eleven normal evaluation samples are available: each false positive changes FPR by about 9.09 percentage points. This one-category, one-split development run does not establish general performance across LIBAD.

This is not a reproduction of the paper's headline result: DINOv3 default weights were inaccessible without authentication, so we selected the supported public DINOv2-small backbone; category scope and decision fusion also differ. No model settings were changed in response to evaluation quality during this setup run.

The reference selects its best-F1 threshold using evaluation labels; that metric is diagnostic, not a validation-selected deployment result. Here it coincides with the selected high-recall threshold. A development/final-evaluation protocol must be defined before comparing improvements, especially because the supplied validation protocol contains only normal examples.

The reported inference timing includes feature extraction/scoring but excludes data loading, resizing, and heatmap saving. It is neither end-to-end service latency nor a p95 statistic. PyTorch peak allocation excludes driver/context allocations, other processes, and some external memory usage.

Heatmap PNGs are normalized independently per image and are qualitative; their colors cannot be compared as absolute anomaly magnitude. No pixel-level accuracy claim is supported by this run.

## Artifacts and reproduction

- Runner: `scripts/run_baseline.py`.
- Outputs: `artifacts/patchcore_vis_wrinkling_seed7345/` (metadata, raw CSV, metric logs, memory bank, PNG maps).
- Console: `reports/baseline-console.log`.
- Offline repeat after caching: set `HF_HUB_OFFLINE=1`, then run `.venv\Scripts\python -u scripts/run_baseline.py` from the project root.
- Setup initially stopped on a missing matplotlib dependency, then completed after installing it; the dependency file now includes it. An interrupted sandboxed retry performed unsuccessful online metadata checks; the successful run used cached weights offline.

Next: inspect the wrinkling examples and heatmaps, check how the chosen features preserve defect evidence, and agree on a fair development protocol before building the trainable alternative. Do not tune repeatedly against this category and later call it an untouched final test.
