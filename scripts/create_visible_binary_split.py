"""Create and validate the fixed visible-light binary-classification split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split


MANIFEST_VERSION = "visible_binary_split_v1"
DEFAULT_SEED = 42
FIELDNAMES = (
    "manifest_version",
    "split_seed",
    "split",
    "sample_key",
    "sample_name",
    "category",
    "source",
    "label",
    "side_a_path",
    "side_b_path",
)
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create the deterministic 60/20/20 LIBAD visible-light split."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root / "data" / "splits" / "LIBAD_normal70train_noval_seed7345.csv",
        help="An official LIBAD CSV used only as a complete sample catalog.",
    )
    parser.add_argument("--data-root", type=Path, default=root / "data" / "LIBAD")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "manifests" / f"{MANIFEST_VERSION}.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "manifests" / f"{MANIFEST_VERSION}.json",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs if their content differs.",
    )
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"sample_key", "sample_name", "class_name", "source"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")
    if len(rows) != 744:
        raise ValueError(f"Expected 744 physical samples, found {len(rows)}")

    keys = [row["sample_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Catalog contains duplicate sample keys")
    if set(row["source"] for row in rows) != {"normal", "anomaly"}:
        raise ValueError("Catalog source values must be exactly normal and anomaly")
    return rows


def assign_splits(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    keys = [row["sample_key"] for row in rows]
    strata = [f'{row["class_name"]}|{row["source"]}' for row in rows]

    train_keys, held_out_keys = train_test_split(
        keys,
        train_size=0.60,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    row_by_key = {row["sample_key"]: row for row in rows}
    held_out_strata = [
        f'{row_by_key[key]["class_name"]}|{row_by_key[key]["source"]}'
        for key in held_out_keys
    ]
    validation_keys, test_keys = train_test_split(
        held_out_keys,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=held_out_strata,
    )

    assignments = {key: "train" for key in train_keys}
    assignments.update({key: "validation" for key in validation_keys})
    assignments.update({key: "test" for key in test_keys})
    return assignments


def create_manifest_rows(
    rows: list[dict[str, str]], assignments: dict[str, str], data_root: Path, seed: int
) -> list[dict[str, str | int]]:
    manifest_rows: list[dict[str, str | int]] = []
    project_root = Path(__file__).resolve().parents[1]
    for row in rows:
        sample_key = row["sample_key"]
        side_a = data_root / f"{sample_key}A.tiff"
        side_b = data_root / f"{sample_key}B.tiff"
        try:
            side_a_relative = side_a.relative_to(project_root).as_posix()
            side_b_relative = side_b.relative_to(project_root).as_posix()
        except ValueError as error:
            raise ValueError("Data root must be inside the project directory") from error

        manifest_rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "split_seed": seed,
                "split": assignments[sample_key],
                "sample_key": sample_key,
                "sample_name": row["sample_name"],
                "category": row["class_name"],
                "source": row["source"],
                "label": 1 if row["source"] == "anomaly" else 0,
                "side_a_path": side_a_relative,
                "side_b_path": side_b_relative,
            }
        )
    manifest_rows.sort(key=lambda row: (SPLIT_ORDER[str(row["split"])], str(row["sample_key"])))
    return manifest_rows


def validate_manifest(rows: list[dict[str, str | int]], project_root: Path) -> dict[str, object]:
    if len(rows) != 744:
        raise ValueError(f"Manifest must contain 744 rows, found {len(rows)}")

    keys = [str(row["sample_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("A physical sample appears more than once in the manifest")

    expected_counts = {"train": 446, "validation": 149, "test": 149}
    split_counts = Counter(str(row["split"]) for row in rows)
    if dict(split_counts) != expected_counts:
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")

    missing_files: list[str] = []
    for row in rows:
        expected_label = 1 if row["source"] == "anomaly" else 0
        if int(row["label"]) != expected_label:
            raise ValueError(f"Label/source mismatch for {row['sample_key']}")
        for field in ("side_a_path", "side_b_path"):
            path = project_root / str(row[field])
            if not path.is_file():
                missing_files.append(str(path))
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} visible files: {missing_files[:5]}")

    stratum_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        stratum = f'{row["category"]}|{row["source"]}'
        stratum_counts[str(row["split"])][stratum] += 1
    all_strata = set().union(*(counts.keys() for counts in stratum_counts.values()))
    for split_name in expected_counts:
        absent = sorted(all_strata.difference(stratum_counts[split_name]))
        if absent:
            raise ValueError(f"{split_name} is missing strata: {absent}")

    source_counts = {
        split_name: dict(
            sorted(Counter(str(row["source"]) for row in rows if row["split"] == split_name).items())
        )
        for split_name in expected_counts
    }
    category_source_counts = {
        split_name: dict(sorted(stratum_counts[split_name].items()))
        for split_name in expected_counts
    }
    return {
        "rows": len(rows),
        "unique_sample_keys": len(set(keys)),
        "split_counts": expected_counts,
        "source_counts": source_counts,
        "category_source_counts": category_source_counts,
        "missing_visible_files": 0,
        "all_22_category_source_strata_present_in_each_split": True,
    }


def manifest_bytes(rows: list[dict[str, str | int]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_checked(path: Path, content: bytes, overwrite: bool) -> None:
    if path.exists() and path.read_bytes() != content and not overwrite:
        raise FileExistsError(f"{path} exists with different content; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    catalog_rows = load_catalog(args.catalog)
    assignments = assign_splits(catalog_rows, args.seed)
    rows = create_manifest_rows(catalog_rows, assignments, args.data_root, args.seed)
    validation = validate_manifest(rows, project_root)

    csv_content = manifest_bytes(rows)
    metadata = {
        "manifest_version": MANIFEST_VERSION,
        "split_seed": args.seed,
        "strategy": "60/20/20 stratified by category and source",
        "source_catalog": args.catalog.relative_to(project_root).as_posix(),
        "source_catalog_sha256": sha256_file(args.catalog),
        "manifest_sha256": sha256_bytes(csv_content),
        "known_limitation": (
            "No roll-level or sheet-level grouping metadata has been established; "
            "the physical sample is the current grouping unit."
        ),
        "validation": validation,
    }
    json_content = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")

    write_checked(args.manifest, csv_content, args.overwrite)
    write_checked(args.metadata, json_content, args.overwrite)

    print(json.dumps(validation, indent=2, sort_keys=True))
    print(f"Manifest: {args.manifest.resolve()}")
    print(f"Metadata: {args.metadata.resolve()}")
    print(f"Manifest SHA-256: {metadata['manifest_sha256']}")


if __name__ == "__main__":
    main()
