# Visible CNN validation results

## Scope

These are development results from the fixed 149-sample validation partition: 72 normal and 77 defective samples. The sealed final test partition was not scored. Each run used the same architecture and data split with a different initialization and training-shuffle seed.

The selected operating point maximizes validation recall while keeping validation false-positive rate (FPR) at or below 10%. Each false positive changes validation FPR by approximately 1.39 percentage points because there are 72 normal validation samples.

## Per-seed results

| Seed | Best epoch | AUROC | Selected threshold | Recall | FPR | Precision | FPR at ≥90% recall | FPR at ≥95% recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 39 | 0.9493 | 0.5956 | 84.42% | 8.33% | 91.55% | 16.67% | 19.44% |
| 42 | 44 | 0.9210 | 0.7426 | 74.03% | 8.33% | 90.48% | 26.39% | 31.94% |
| 73 | 35 | 0.9006 | 0.4810 | 72.73% | 5.56% | 93.33% | 45.83% | 51.39% |

## Aggregate across three seeds

| Measurement | Mean | Population standard deviation | Range |
| --- | ---: | ---: | ---: |
| AUROC | 0.9236 | 0.0200 | 0.9006–0.9493 |
| Recall under 10% FPR constraint | 77.06% | 5.23% | 72.73%–84.42% |
| Selected FPR | 7.41% | 1.31% | 5.56%–8.33% |
| FPR at ≥90% recall | 29.63% | 12.13% | 16.67%–45.83% |
| FPR at ≥95% recall | 34.26% | 13.14% | 19.44%–51.39% |

## Recall by defect category at each seed's selected threshold

| Category | Validation defects | Mean recall | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| 10_streak | 7 | 100.00% | 100.00% | 100.00% |
| 11_pinhole | 11 | 81.82% | 72.73% | 90.91% |
| 1_wrinkling | 8 | 83.33% | 75.00% | 87.50% |
| 2_particle | 7 | 95.24% | 85.71% | 100.00% |
| 3_pit | 7 | 85.71% | 71.43% | 100.00% |
| 4_unevenness | 6 | 55.56% | 16.67% | 83.33% |
| 5_barefoil | 6 | 100.00% | 100.00% | 100.00% |
| 6_scratch | 7 | 76.19% | 42.86% | 100.00% |
| 7_polarity | 7 | 0.00% | 0.00% | 0.00% |
| 8_debonding | 4 | 75.00% | 75.00% | 75.00% |
| 9_crack | 7 | 90.48% | 85.71% | 100.00% |

## Findings

- Mean validation AUROC was **0.9236**, with a range of 0.9006 to 0.9493. Initialization materially affected the result.
- Under the 10% FPR constraint, mean recall was **77.06%** and individual-seed recall ranged from 72.73% to 84.42%.
- Reaching at least 95% recall required mean FPR of **34.26%**, ranging from 19.44% to 51.39%.
- Polarity recall was 0% for all three seeds at the selected operating thresholds. This is the clearest category-level failure to investigate.
- Streak and bare-foil defects reached 100% recall for every seed at the selected thresholds, although their validation sample counts remain small.
- The three runs took 26.5 minutes in total. Peak PyTorch CUDA allocation was approximately 1288 MiB.

## Limits

- These are validation results used for development; they are not final test estimates.
- Thresholds differ by seed because score calibration differs across trained models.
- Per-category estimates use only 4 to 11 defective validation samples per category.
- The earlier PatchCore result used a different split and cannot support a direct superiority claim.
- No model change has yet been tested against this initial configuration.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\train_visible_cnn.py
.\.venv\Scripts\python.exe scripts\summarize_visible_cnn_validation.py
```
