import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch
import numpy as np
import nibabel as nib
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.spectre import MODEL_CONFIGS, SpectreImageFeatureExtractor
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataloader import get_FFTorIFFTdataloader, get_dataloader

def save_config_json(output_dir: Path, args: argparse.Namespace) -> None:

    config = {
        "script": Path(__file__).name,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

def parse_grid_size(raw: str) -> Tuple[int, int, int]:
    cleaned = raw.replace("x", ",")
    parts = [p for p in cleaned.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Grid size non valida: {raw}")
    return tuple(int(p) for p in parts)


def parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Valore booleano non valido: {raw}")


def split_volume_into_windows(volumes: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
    if volumes.ndim != 5:
        raise ValueError(f"Atteso tensore 5D (B,1,D,H,W), ricevuto shape {volumes.shape}")
    g_h, g_w, g_d = grid_size

    volumes_hwd = volumes.permute(0, 1, 3, 4, 2).contiguous()
    _, _, H, W, D = volumes_hwd.shape

    if (H % g_h) or (W % g_w) or (D % g_d):
        raise ValueError(
            f"Shape {H}x{W}x{D} non divisibile per la griglia {grid_size} "
            "(assicurati che i volumi siano già resample/resize correttamente)."
        )

    h_size, w_size, d_size = H // g_h, W // g_w, D // g_d
    windows = volumes_hwd.view(-1, 1, g_h, h_size, g_w, w_size, g_d, d_size)
    windows = windows.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    windows = windows.view(-1, g_h * g_w * g_d, 1, h_size, w_size, d_size)
    return windows


@torch.no_grad()
def extract_embeddings(
    model: SpectreImageFeatureExtractor,
    loader: Iterable,
    output_dir: Path,
    device: torch.device,
    grid_size: Tuple[int, int, int],
    set_name: str,
    ty: str,
):
    def _python_value(value):
        if torch.is_tensor(value):
            return value.item()
        return value

    out_of_range_rows = []

    for batch in tqdm(loader, desc="Extracting", unit="vol"):
        cubes = batch["cube"].to(device, non_blocking=True)
        names = [f"{img}" for img in batch["img_id"]]
        mods = batch["mod"]
        orig_ids = batch["orig_id"]
        labels = batch["label"]
        splits = batch["split"]
        tys = batch["ty"]
        source_paths = batch["path"]
        cube_mins = batch["patch_min"]
        cube_maxs = batch["patch_max"]

        windows = split_volume_into_windows(cubes, grid_size)
        windows = windows.to(dtype=torch.float32)
        _, features = model(windows, grid_size=grid_size)
        features = features.cpu()

        for idx, feat in enumerate(features):
            name = names[idx]
            mod = mods[idx]
            save_dir = output_dir / set_name / ty / mod
            samples_dir = save_dir / "samples"
            metadata_dir = save_dir / "metadata"
            samples_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)

            save_path = samples_dir / f"{name}.pt"
            metadata_path = metadata_dir / f"{name}.csv"

            if feat.ndim >= 2:
                feat = feat[0].contiguous().clone()
            if not save_path.exists():
                torch.save(feat, save_path)
            else:
                tqdm.write(f"[SKIP] {save_path} già presente")

            if not metadata_path.exists():
                metadata_row = {
                    "img_id": name,
                    "orig_id": orig_ids[idx],
                    "mod": mod,
                    "ty": tys[idx],
                    "split": splits[idx],
                    "label": _python_value(labels[idx]),
                    "source_path": source_paths[idx],
                    "path": str(save_path),
                }
                pd.DataFrame([metadata_row]).to_csv(metadata_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract pretrained SPECTRE embeddings from real and manipulated M3DSynth CT patches.")
    parser.add_argument("--data-root", type=Path, required=False, help="Root directory containing precomputed patch or IFFT samples and their metadata.")
    parser.add_argument("--mods", type=str, required=True, help="Comma-separated generator names to process, such as 'real,cycle,diffusion'.")
    parser.add_argument("--ty", type=str, required=True, help="Manipulation type to select from the ty column, such as inj or rem.")
    parser.add_argument("--set-name", type=str, default="train", help="Dataset split to process: train, valid, or test.")
    parser.add_argument("--grid-size", type=parse_grid_size, default="3,3,4", help="Number of non-overlapping windows along the H, W, and D axes; use 2,2,2 for 32-voxel patches.")
    parser.add_argument("--output-dir", type=Path, default=Path("./embeddings_spectre"), help="Root directory for saved .pt embeddings, per-sample metadata, and the configuration JSON.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of patches processed per inference batch.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")
    parser.add_argument(
        "--source-representation",
        type=str,
        default="img",
        choices=["img", "ifft"],
        help="Input representation: 'img' loads spatial patches; 'ifft' loads reconstructed frequency patches.",
    )
    parser.add_argument("--sample_format", type=str, default="npy", choices=["npy", "nii.gz"],
                        help="Format of the input patches: NumPy .npy or NIfTI .nii.gz.")
    parser.add_argument(
        "--norm_mode_ifft",
        type=parse_bool,
        default=True,
        help="For IFFT inputs, apply global min-max scaling from all metadata under the input root when true; otherwise preserve stored values.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = tuple(args.grid_size)

    save_config_json(args.output_dir / args.set_name / args.ty, args)

    mods = [m.strip() for m in args.mods.split(",") if m.strip()]
    if args.source_representation == "img":
        loader = get_dataloader(
            root=args.data_root,
            split=args.set_name,
            ty=args.ty,
            mods=mods,
            sample_format=args.sample_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )
    elif args.source_representation == "ifft":
        loader = get_FFTorIFFTdataloader(
            root=args.data_root,
            split=args.set_name,
            ty=args.ty,
            mods=mods,
            sample_format=args.sample_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            norm_mode=args.norm_mode_ifft
        )
    else:
        raise ValueError(
            f"Unsupported source-representation={args.source_representation!r}. "
            "Expected 'img' or 'ifft'."
        )


    config = MODEL_CONFIGS["spectre-large-pretrained"]
    model = SpectreImageFeatureExtractor.from_config(config)
    model.eval().to(device)

    extract_embeddings(
        model,
        loader,
        args.output_dir,
        device=device,
        grid_size=grid_size,
        set_name=args.set_name,
        ty=args.ty,
    )


if __name__ == "__main__":
    main()
