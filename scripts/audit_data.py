"""Inventory local LIBAD archives without extracting or altering them."""
import csv
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    report = {}
    with ZipFile(root / 'LIBAD.zip') as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        names = {item.filename for item in files}
        report['image_files'] = len(files)
        report['uncompressed_bytes'] = sum(item.file_size for item in files)
        report['counts_by_category_and_label'] = dict(sorted(Counter(
            '/'.join(Path(item.filename).parts[1:3]) for item in files
        ).items()))
        report['example_image_headers'] = []
        for suffix in ['A', 'B', 'L', 'X']:
            name = next(item.filename for item in files if item.filename.endswith(suffix + '.tiff'))
            with Image.open(io.BytesIO(archive.read(name))) as image:
                report['example_image_headers'].append({
                    'path': name, 'size': image.size, 'mode': image.mode,
                    'bits_per_sample': image.tag_v2.get(258),
                })
    with ZipFile(root / 'splits.zip') as archive:
        report['splits'] = []
        for name in sorted(archive.namelist()):
            if not name.endswith('.csv'):
                continue
            rows = list(csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig'))))
            keys = [row['sample_key'] for row in rows]
            missing = [f"LIBAD/{key}{suffix}.tiff" for key in keys for suffix in 'ABLX'
                       if f"LIBAD/{key}{suffix}.tiff" not in names]
            report['splits'].append({
                'file': name, 'rows': len(rows),
                'duplicate_sample_keys': len(keys) - len(set(keys)),
                'missing_modality_files': missing,
                'counts': dict(sorted(Counter(
                    f"{row['split']}/{row['source']}" for row in rows
                ).items())),
            })
    destination = root / 'reports' / 'data_inventory.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'splits'}, indent=2))
    print('Split summaries:', json.dumps(report['splits'][:2], indent=2))
    print('All split references present:', all(not row['missing_modality_files'] for row in report['splits']))
    print('Saved:', destination)


if __name__ == '__main__':
    main()
