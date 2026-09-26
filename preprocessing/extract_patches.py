import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from M3DSynth_dataset_utils import CTPatchExtractor, get_table_date


@torch.no_grad()
def save_cubes(
    loader: Iterable,
    output_dir: Path,
    device: torch.device,
    set_name: str,
    ty: str,
    sample_format: str,
) -> None:
    samples_dirname = "samples" if sample_format == "npy" else "samples_nii"
    sample_suffix = ".npy" if sample_format == "npy" else ".nii.gz"

    for batch in tqdm(loader, desc="Saving cubes", unit="vol"):
        cubes = batch["cube"].to(device, non_blocking=True)
        names = [f"{img}" for img in batch["img_id"]]
        mods = batch["mod"]
        orig_ids = batch["orig_id"]
        tys = batch["ty"]
        patch_mins = batch["patch_min"]
        patch_maxs = batch["patch_max"]
        for i, (name, mod, orig_id, ty, patch_min, patch_max) in enumerate(
            zip(names, mods, orig_ids, tys, patch_mins, patch_maxs)
        ):
            save_dir = output_dir / set_name / ty / mod
            samples_dir = save_dir / samples_dirname
            metadata_dir = save_dir / "metadata"
            samples_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            sample_path = samples_dir / f"{name}{sample_suffix}"
            metadata_path = metadata_dir / f"{name}.csv"

            cube_np = cubes[i].cpu().numpy()
            if cube_np.ndim == 4 and cube_np.shape[0] == 1:
                cube_np = cube_np[0]

            shape_value = tuple(cube_np.shape)

            if sample_path.exists() and metadata_path.exists():
                tqdm.write(f"[SKIP] {name} già salvato")
                continue

            if not sample_path.exists():
                if sample_format == "npy":
                    np.save(sample_path, cube_np.astype(np.float32))
                else:
                    nifti_img = nib.Nifti1Image(cube_np.astype(np.float32), affine=np.eye(4))
                    nib.save(nifti_img, sample_path)

            if not metadata_path.exists():
                metadata_df = pd.DataFrame([
                    {
                        "img_id": name,
                        "orig_id": orig_id,
                        "mod": mod,
                        "ty": ty,
                        "shape": str(shape_value),
                        "patch_min": float(patch_min),
                        "patch_max": float(patch_max),
                        "path": str(sample_path),
                    }
                ])
                metadata_df.to_csv(metadata_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract real and manipulated M3DSynth CT patches and save them as .npy or .nii.gz files."
    )
    parser.add_argument("--data-root", type=Path, required=False, help="Default root for both real and manipulated CT scans unless separate roots are supplied.")
    parser.add_argument("--data-root-real", type=Path, required=False, help="Root containing real CT scans under <orig_id>/.")
    parser.add_argument("--data-root-fake", type=Path, required=False, help="Root containing manipulated scans under <mod>/scan/<img_id>/ and masks under <mod>/label/<img_id>/.")
    parser.add_argument("--data-csv", type=Path, required=True, help="Sample catalog CSV containing image IDs, original scan IDs, generator names, and manipulation types.")
    parser.add_argument("--sets-csv", type=Path, required=True, help="CSV mapping each original scan ID to a train, valid, or test split.")
    parser.add_argument("--mods", type=str, required=True, help="Comma-separated generator names to include, such as 'real,cycle,diffusion'.")
    parser.add_argument("--ty", type=str, required=True, help="Manipulation type to select from the ty column, such as inj or rem.")
    parser.add_argument("--set-name", type=str, default="train", help="Dataset split to process: train, valid, or test.")
    parser.add_argument(
        "--value-space",
        type=str,
        choices=["stored-value", "hu"],
        default="hu",
        help="Input intensity space: stored TIFF values or Hounsfield units reconstructed from LIDC metadata.",
    )
    parser.add_argument(
        "--value-normalization",
        type=str,
        choices=["none", "sample-wise", "hu-clip-01"],
        default="none",
        help="Intensity normalization: none; sample-wise percentile scaling; or HU clipping to [-1000, 1000] followed by scaling to [0, 1].",
    )
    parser.add_argument(
        "--lidc-stats-csv",
        type=Path,
        help="LIDC metadata CSV with min_export, rescale_slope, and rescale_intercept for each orig_id; required when --value-space is hu.",
    )
    parser.add_argument("--sample-format", type=str, choices=["npy", "nii.gz"], default="npy", help="Output patch format: NumPy .npy or NIfTI .nii.gz.")
    parser.add_argument("--output-dir", type=Path, default=Path("./preprocessed_cubes"), help="Root directory for saved patches and per-patch metadata.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of patches loaded per batch.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Number of sample IDs to process starting at --index; process all samples when omitted.")
    parser.add_argument("--index", type=int, default=0, help="Zero-based start index in the filtered sample ID list when --chunk-size is set.")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_root = args.data_root_real or args.data_root
    fake_root = args.data_root_fake or args.data_root
    if real_root is None or fake_root is None:
        raise ValueError("Specifica almeno --data-root oppure entrambe --data-root-real e --data-root-fake.")

    full_tab = get_table_date(
        ty=args.ty,
        mods=args.mods,
        set_name=args.set_name,
        data_path=args.data_csv,
        set_path=args.sets_csv,
    )

    if args.chunk_size is not None:
        sample_ids = full_tab["img_id"].tolist()
        selected = sample_ids[args.index : args.index + args.chunk_size]
        tab = full_tab[full_tab["img_id"].isin(selected)].reset_index(drop=True)
    else:
        tab = full_tab

    dataset = CTPatchExtractor(
        real_root,
        fake_root,
        tab,
        value_space=args.value_space,
        value_normalization=args.value_normalization,
        lidc_stats_csv=args.lidc_stats_csv,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    save_cubes(
        loader,
        args.output_dir,
        device=device,
        set_name=args.set_name,
        ty=args.ty,
        sample_format=args.sample_format,
    )


if __name__ == "__main__":
    main()
