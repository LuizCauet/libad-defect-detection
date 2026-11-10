"""Run LIBAD reference scoring on all categories with official split assignments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'third_party' / 'LIBAD'
OUTPUT = ROOT / 'artifacts' / 'patchcore_vis_all_seed7345'


def readable_metrics(formatter):
    """Give console counts clear labels without changing stored metric keys."""
    labels = {
        'image_tn': 'Good samples correctly passed (TN)',
        'image_fp': 'Good samples incorrectly flagged (FP)',
        'image_fn': 'Defective samples missed (FN)',
        'image_tp': 'Defective samples correctly flagged (TP)',
    }

    def format_table(metrics, digits=4):
        table = formatter(metrics, digits=digits)
        table['metric'] = table['metric'].replace(labels)
        return table

    return format_table


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    # Optional offline repeat runs: set HF_HUB_OFFLINE=1 after weights are cached.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(REFERENCE))
    import torch
    import dataset
    import run
    from utils import io_utils

    io_utils.format_metrics = readable_metrics(io_utils.format_metrics)

    original = dataset.CoatingDatasetSplit
    counts = {}

    class RecordedDataset(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            counts[self.split] = len(self.items)
            print(f'All categories; {self.split}={len(self.items)}', flush=True)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            if not self.return_meta:
                return result
            sample, meta = result
            # Sample names repeat across categories and normal/anomaly folders.
            # Use the full key to prevent heatmaps overwriting one another.
            meta['sample_name'] = meta['sample_key'].replace('/', '__')
            return sample, meta

    run.CoatingDatasetSplit = RecordedDataset
    arguments = [
        '--root_path', str(ROOT / 'data' / 'LIBAD'),
        '--split_dir', str(ROOT / 'data' / 'splits'),
        '--output_dir', str(OUTPUT),
        '--results_csv', str(OUTPUT / 'experiment_log.csv'),
        '--results_md', str(OUTPUT / 'experiment_log.md'),
        '--modality', 'vis', '--dataset_seed', '7345',
        '--backbone_family', 'vit', '--dino_version', 'v2', '--backbone_variant', 'small',
        '--extractor_precision', 'fp32', '--coreset_selection_method', 'fps',
        '--f_coreset', '0.05', '--resize_h', '640', '--resize_w', '512',
        '--batch_size', '1', '--test_batch_size', '1', '--num_workers', '0',
        '--distance_chunk_size', '512', '--decision_fusion', 'weighted_sum', '--no-use_ocsvm',
        '--save_raw_scores', '--save_pixel_maps', '--save_bank', '--measure_inference_time',
        '--notes', 'development_all_dinov2_small_fp32',
    ]
    revision = subprocess.check_output([
        'git', '-c', f'safe.directory={REFERENCE.as_posix()}', '-C', str(REFERENCE),
        'rev-parse', 'HEAD',
    ], text=True).strip()
    metadata = {
        'reference_revision': revision, 'arguments': arguments,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'caveat': 'Development evaluation; DINOv2-small differs from gated DINOv3 default. No paper reproduction claim.',
        'category_filter': None,
        'sample_identifier': 'Full sample_key with / replaced by __; prevents output collisions',
    }
    (OUTPUT / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    sys.argv = [str(REFERENCE / 'run.py'), *arguments]
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        run.main()
        metadata['status'] = 'completed'
    except Exception as error:
        metadata['status'] = 'failed'
        metadata['error'] = repr(error)
        raise
    finally:
        metadata['elapsed_seconds'] = time.perf_counter() - start
        metadata['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        metadata['counts'] = counts
        (OUTPUT / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
