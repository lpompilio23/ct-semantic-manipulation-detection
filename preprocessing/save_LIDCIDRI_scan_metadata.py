import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from LIDCIDRI_dataset_utils import (
    compute_volume_stats,
    compute_scan_vmin_export,
    load_orig_ids,
    read_dicom_volume,
    resolve_lidc_scan_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-scan statistics and DICOM rescale parameters for original LIDC CT volumes listed in the sample catalog."
    )
    parser.add_argument("--data-csv", type=Path, default=None, help="M3DSynth sample catalog CSV containing the orig_id values to process.")
    parser.add_argument("--lidc-root", type=Path, default=None, help="Root directory containing one DICOM scan directory per orig_id.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Root directory where per-scan metadata CSV files are saved under metadata/.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of unique orig_id values to process starting at --index; process all values when omitted.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based start index in the list of unique orig_id values when --chunk-size is set.",
    )
    parser.add_argument(
        "--chunk-name",
        type=str,
        default=None,
        help="Name used for the failures CSV; defaults to the start index.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir

    orig_ids = load_orig_ids(args.data_csv)
    if args.chunk_size is not None:
        chunk_orig_ids = orig_ids[args.index : args.index + args.chunk_size]
    else:
        chunk_orig_ids = orig_ids

    chunk_name = args.chunk_name if args.chunk_name is not None else str(args.index)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    failures = []

    for orig_id in tqdm(chunk_orig_ids, desc="Scanning LIDC volumes", unit="scan"):
        try:
            scan_dir = resolve_lidc_scan_path(orig_id, args.lidc_root)
            volume, slope, intercept = read_dicom_volume(scan_dir)
            stats = compute_volume_stats(volume, slope, intercept)
            stats["min_export"] = compute_scan_vmin_export(volume)

            stats.update(
                {
                    "orig_id": orig_id,
                    "scan_dir": str(scan_dir),
                }
            )
            sample_csv = metadata_dir / f"{orig_id}.csv"
            if sample_csv.exists():
                tqdm.write(f"[SKIP] {sample_csv.name} già esistente")
                continue
            pd.DataFrame([stats]).to_csv(sample_csv, index=False)
        except Exception as exc:
            failures.append({"orig_id": orig_id, "error": str(exc)})
            tqdm.write(f"[SKIP] {orig_id}: {exc}")

    if failures:
        failures_df = pd.DataFrame(failures)
        failures_df.to_csv(metadata_dir / f"failures_{chunk_name}.csv", index=False)


if __name__ == "__main__":
    main()
