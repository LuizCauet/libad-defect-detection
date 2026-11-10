"""Re-evaluate saved anomaly scores; do not fit a model or change its artifacts."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts' / 'patchcore_vis_all_seed7345'


def evaluate(labels, scores, threshold):
    flagged = scores >= threshold
    tp = int(np.sum(flagged & (labels == 1)))
    fp = int(np.sum(flagged & (labels == 0)))
    tn = int(np.sum(~flagged & (labels == 0)))
    fn = int(np.sum(~flagged & (labels == 1)))
    return dict(threshold=float(threshold), defects_caught=tp, defects_missed=fn,
                good_falsely_flagged=fp, good_correctly_passed=tn,
                recall=tp / (tp + fn), fpr=fp / (fp + tn),
                precision=tp / (tp + fp) if tp + fp else 0.0)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows):
    result = ['| Setting | Threshold | Defects caught | Defects missed | Good falsely flagged | Good correctly passed | Recall | FPR | Precision |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        result.append(
            f"| {row['setting']} | {row['threshold']:.6f} | {row['defects_caught']} | {row['defects_missed']} "
            f"| {row['good_falsely_flagged']} | {row['good_correctly_passed']} | {row['recall']:.2%} "
            f"| {row['fpr']:.2%} | {row['precision']:.2%} |")
    return '\n'.join(result)


def main():
    source = next(RUN.glob('raw_scores_*.csv'))
    with source.open(encoding='utf-8-sig', newline='') as handle:
        records = list(csv.DictReader(handle))
    labels = np.array([int(row['image_label']) for row in records])
    # The reference calculates its metrics on float32 scores. Preserve that precision.
    scores = np.array([float(row['image_score']) for row in records], dtype=np.float32)
    if len(records) != 492 or len({row['sample_name'] for row in records}) != 492:
        raise ValueError('Expected 492 unique full-dataset evaluation samples.')
    if not np.isfinite(scores).all() or set(labels) != {0, 1}:
        raise ValueError('Scores must be finite and labels must include normal and anomalous samples.')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = float(thresholds[int(np.argmax(f1))])
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    target_index = int(np.flatnonzero(tpr >= .95)[0])
    target = float(roc_thresholds[target_index])
    selected = [('Reference best-F1 threshold', best), ('At least 95% recall (diagnostic)', target)]
    selected += [(f'Fixed threshold {value}', float(value)) for value in [11, 12, 13, 14, 15]]
    selected.sort(key=lambda item: item[1])
    rows = [dict(setting=name, **evaluate(labels, scores, value)) for name, value in selected]
    baseline = evaluate(labels, scores, best)
    with (RUN / 'experiment_log.csv').open(encoding='utf-8-sig', newline='') as handle:
        reference = list(csv.DictReader(handle))[-1]
    auroc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    if not np.isclose(auroc, float(reference['image_auroc'])):
        raise ValueError('Recomputed AUROC does not match saved reference metrics.')
    if not np.isclose(best, float(reference['image_best_f1_threshold'])):
        raise ValueError('Recomputed best-F1 threshold does not match reference.')
    if (baseline['defects_caught'], baseline['good_falsely_flagged'], baseline['defects_missed']) != (374, 82, 9):
        raise ValueError('Reference confusion counts differ from supplied run.')
    report_dir = ROOT / 'reports'
    report_dir.mkdir(exist_ok=True)
    write_csv(report_dir / 'threshold_tradeoffs.csv', rows)
    sweep = [evaluate(labels, scores, float(value)) for value in sorted(set(scores.tolist()))]
    # Include a threshold above every score to represent flagging nothing.
    sweep.append(evaluate(labels, scores, float(np.nextafter(scores.max(), np.float32(np.inf)))))
    write_csv(report_dir / 'threshold_sweep.csv', sweep)
    metadata = json.loads((RUN / 'run_metadata.json').read_text(encoding='utf-8'))
    provenance = dict(source=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      comparison='image_score >= threshold', samples=len(records),
                      normal=int(np.sum(labels == 0)), anomaly=int(np.sum(labels == 1)),
                      image_auroc=auroc, average_precision=ap, fpr95=float(fpr[target_index]),
                      threshold95=target, recall_at_threshold95=float(tpr[target_index]),
                      reference_revision=metadata['reference_revision'])
    (report_dir / 'threshold_analysis.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    text = f'''# Defect inspection: findings

## Experiment 1: full-dataset PatchCore reference

Luiz ran the existing LIBAD implementation with a frozen DINOv2-small feature extractor. The model built its normal memory bank from 252 normal samples across all eleven categories. Evaluation used 492 samples: 109 normal and 383 defective.

Configuration: visible-light sides A/B, max-side map combination, 640 × 512 input, float32 extraction, batch size one, 5% farthest-point-sampling coreset, and single-modality weighted-sum scoring. The bank contains 40,824 vectors of 384 numbers each, selected from 816,480 patch features. The image score averages the highest-scoring 1% of the combined map values.

Official split: `LIBAD_normal70train_noval_seed7345.csv`. Reference revision: `{metadata['reference_revision']}`. DINOv2-small differs from the default gated DINOv3 backbone; this is not a reproduction of the paper's headline result.

### Ranking results

- Image area under the receiver operating characteristic curve (AUROC): **{auroc:.4f}**.
- Average precision, labeled image AUPR by the reference: **{ap:.4f}**.
- Evaluation defect prevalence: **{np.mean(labels):.2%}**. Precision on this defect-heavy set does not establish precision on a production stream with rare defects.

### Threshold experiment

We re-evaluated the saved predictions using higher thresholds. We did not retrain DINOv2, rebuild the bank, or rerun inference. Only the rule for converting a score to a flag changed: **flag when score is greater than or equal to the threshold**.

The reference selected the first threshold to maximize F1 on these evaluation labels. F1 combines precision and recall. The additional fixed thresholds of 11 through 15 are illustrative score cutoffs, not approved operating settings. The 95%-recall row is calculated from the evaluation receiver operating characteristic curve.

{table(rows)}

The table displays rounded thresholds; CSV outputs retain full numerical precision. Recall is the fraction of 383 defects caught. False-positive rate (FPR) is the fraction of 109 normal samples falsely flagged. Precision is the fraction of alerts that are real defects.

### Interpretation

Increasing the threshold reduces false alarms but also misses more defects. The threshold changes decisions; it does not improve the score ranking, so AUROC and average precision remain unchanged.

At the reference threshold, the system catches 374 defects and misses 9, but falsely flags 82 good samples. That is 97.65% recall with 75.23% FPR. At the diagnostic threshold reaching at least 95% recall, the measured FPR is **{fpr[target_index]:.2%}** and realized recall is **{tpr[target_index]:.2%}**.

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
.\\.venv\\Scripts\\python.exe scripts\\analyze_thresholds.py
```

This command regenerates this report and the analysis tables from the saved run. Add later experiment narratives to separate reports or update the generator to preserve them.

Files:

- `reports/threshold_tradeoffs.csv`: the selected table rows.
- `reports/threshold_sweep.csv`: counts and rates at every distinct saved score, plus an above-maximum threshold that flags nothing.
- `reports/threshold_analysis.json`: input score-file checksum, sample counts, computed ranking metrics, and diagnostic FPR95.
- `artifacts/patchcore_vis_all_seed7345/`: original run settings, scores, bank, and heatmaps.

Validation: checked unique sample identifiers, finite scores, matching reference AUROC and best-F1 threshold, and matching original confusion counts. Values are computed from saved scores at the float32 precision used by the reference.
'''
    (ROOT / 'report.md').write_text(text, encoding='utf-8')
    print(table(rows))
    print('Saved report.md and reports/threshold_* analysis files.')


if __name__ == '__main__':
    main()
