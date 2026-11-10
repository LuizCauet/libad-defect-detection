# Initial feasibility check

- Local archives: LIBAD.zip and splits.zip are present and readable.
- Inventory: 2,976 TIFF files; split tables identify 744 sample keys across eleven categories, with four modality files per referenced sample.
- All modality paths referenced by the twenty split CSVs exist in the image archive.
- Expanded image-file size: approximately 7.80 GB (decimal); approximately 48.8 GB disk space was available at inspection.
- Hardware: NVIDIA GeForce RTX 2060, 6,144 MiB VRAM; Python 3.12 is available. PyTorch and uv were not found in the inspected shell environment.
- Example headers only: visible sides A/B are 8-bit grayscale, low-resolution X-ray L is 32-bit floating-point, and high-resolution X-ray X is 16-bit; inspected examples are 1,024 × 1,280. This is not a full image decode/integrity check.
- Important evaluation constraint: inspected normal-validation protocol has only normal validation samples (197 normal training, 55 normal validation, 109 normal test, 383 anomaly test for seed 1245). It cannot directly select a threshold targeting defect recall using validation anomalies; inspect the reference evaluation and define a disclosed development protocol before tuning models.
- Still to verify: dataset license/access terms, provenance/grouping metadata, full image integrity, reference dependencies and GPU compatibility, all protocol variants, and sample visual inspection.

Reproduce inventory: `python scripts/audit_data.py` (requires Pillow). Full counts and split checks are in `data_inventory.json`; no image pixel arrays were inspected visually or used for model selection here.

Next: inspect reference code and licensing, create an isolated dependency environment compatible with the RTX 2060, and prepare a documented visual inspection subset without treating inspected examples as untouched final evaluation.
