# Battery electrode defect inspection

Employer-focused applied computer vision study using LIBAD. See [plan.md](plan.md) for the roadmap.

## First baseline

The existing LIBAD reference supplies PatchCore-style memory-bank scoring with FPS coreset selection. Our first configuration uses public DINOv2-small features because the reference default DINOv3 weights require separate authenticated access; it is not a reproduction of the paper's headline result.

- Visible-light sides A and B; reference max-side aggregation.
- Category: wrinkling, with original assignments from official seed 7345, no-validation protocol.
- Reference revision: `8f98f2ebde63cd45660e58d6413f5ccd638966f4`.
- Input: 640 high × 512 wide; float32 extraction; batch size 1; 5% FPS coreset.
- Single-modality weighted sum bypasses unnecessary OCSVM fusion; this differs from the reference default.
- This is development evaluation. Inspected samples/results are not untouched final test evidence.

## Windows setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements-baseline.txt
git clone https://github.com/evenrose/LIBAD.git third_party/LIBAD
git -C third_party/LIBAD checkout 8f98f2ebde63cd45660e58d6413f5ccd638966f4
```

Use the CUDA 12.6 build for the inspected RTX 2060 environment; the upstream Linux dependency export includes nonportable entries and is not installed wholesale. Exact resolved packages are recorded in `requirements-baseline-lock.txt` after setup.

Place authorized dataset files at `data/LIBAD/` and official CSVs at `data/splits/`. Original archives, extracted data, downloaded weights, and generated artifacts are excluded from Git.

```powershell
.venv\Scripts\python -u scripts/run_baseline.py
```

After the first model download, repeat without network checks:

```powershell
$env:HF_HUB_OFFLINE = '1'
.venv\Scripts\python -u scripts/run_baseline.py
```

Outputs are under `artifacts/patchcore_vis_wrinkling_seed7345/`: run metadata, reference metrics, raw scores, saved memory bank, and score maps. The adapter filters dataset items by category without changing upstream source files; it does not shuffle official assignments.

The reference's saved heatmaps are PNGs normalized separately for each image; their colors are not comparable absolute anomaly scores across images. Use raw image scores for numerical comparisons.

## Attribution and limits

- Reference code: [evenrose/LIBAD](https://github.com/evenrose/LIBAD), BSD-3-Clause; retain its license when redistributing code.
- Dataset: [Evenrose/LIBAD](https://huggingface.co/datasets/Evenrose/LIBAD), labeled CC-BY-4.0 on the public card; file access requires accepting the provider's conditions. Do not commit the dataset or credentials.
- Cite Sui et al., *LIBAD: A Multimodal Anomaly Detection Benchmark for Li-Ion Battery Electrode Manufacturing*, https://arxiv.org/abs/2608.07958.
- PatchCore: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, https://arxiv.org/abs/2106.08265.
- Timing from the reference loop excludes data loading and resizing; it is not end-to-end service latency. Metrics chosen from evaluation labels are diagnostic rather than a validation-selected production threshold.
