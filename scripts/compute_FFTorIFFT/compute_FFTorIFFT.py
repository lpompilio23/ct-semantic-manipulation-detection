import argparse
import sys
from pathlib import Path
from typing import Iterable
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataloader import get_dataloader

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


def make_radial_mask_3d(shape_zyx: Tuple[int, int, int], cutoff: float, kind: str) -> np.ndarray:


    Z, Y, X = shape_zyx
    zz = np.arange(Z) - (Z // 2)
    yy = np.arange(Y) - (Y // 2)
    xx = np.arange(X) - (X // 2)
    Zg, Yg, Xg = np.meshgrid(zz, yy, xx, indexing="ij")

    r = np.sqrt(Zg**2 + Yg**2 + Xg**2)


    r_max = np.sqrt((Z//2)**2 + (Y//2)**2 + (X//2)**2)
    r0 = cutoff * r_max

    if kind == "lowpass":
        mask = (r <= r0).astype(np.float32)
    elif kind == "highpass":
        mask = (r >= r0).astype(np.float32)
    else:
        raise ValueError("kind must be 'lowpass' or 'highpass'")
    return mask


def compute_FFTorIFFT(
    volume_zyx: np.ndarray,
    cutoff: float,
    kind: str,
) -> Tuple[np.ndarray, np.ndarray]:


    spectrum = np.fft.fftn(volume_zyx)
    spectrum_shifted = np.fft.fftshift(spectrum)
    mask = make_radial_mask_3d(volume_zyx.shape, cutoff=cutoff, kind=kind)
    spectrum_filt_shifted = spectrum_shifted * mask
    magnitude = np.abs(spectrum_filt_shifted).astype(np.float32)
    spectrum_filt = np.fft.ifftshift(spectrum_filt_shifted)
    reconstructed = np.real(np.fft.ifftn(spectrum_filt)).astype(np.float32)

    return magnitude, reconstructed


def minmax_normalize_volume(volume: np.ndarray) -> np.ndarray:


    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32)


def save_fft_central_slice(
    fft_volume: np.ndarray,
    out_dir: Path,
    name: str,
    cmap: str,
    interpolation: str,
    dpi: int,
    figsize: tuple,
) -> None:


    out_dir.mkdir(parents=True, exist_ok=True)

    fft_no_ch = fft_volume[0]

    zc = fft_no_ch.shape[0] // 2
    yc = fft_no_ch.shape[1] // 2
    xc = fft_no_ch.shape[2] // 2

    slice_xy = fft_no_ch[zc, :, :]
    slice_xz = fft_no_ch[:, yc, :]
    slice_yz = fft_no_ch[:, :, xc]

    for plane, data in [("xy", slice_xy), ("xz", slice_xz), ("yz", slice_yz)]:
        fig, ax = plt.subplots(figsize=figsize)
        vmin = np.min(data)
        vmax = np.max(data)
        if vmax > vmin:
            data_norm = (data - vmin) / (vmax - vmin)
        else:
            data_norm = np.zeros_like(data)
        im = ax.imshow(data_norm, cmap=cmap, interpolation=interpolation)
        ax.set_title(f"{name} FFT {plane}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_fft_{plane}.png", dpi=dpi)
        plt.close(fig)


def get_mods_list(mods: str) -> Iterable[str]:
    return [m.strip() for m in mods.split(",") if m.strip()]


def save_FFTorIFFT_cubes(
    loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    set_name: str,
    ty: str,
) -> None:
    do_hf = args.freq in ("hf", "both")
    do_lf = args.freq in ("lf", "both")
    mode_fft = args.mode == "fft"
    norm_mode = args.norm_mode
    samples_dirname = "samples" if args.sample_format == "npy" else "samples_nii"
    sample_suffix = ".npy" if args.sample_format == "npy" else ".nii.gz"

    for batch in tqdm(loader, desc="Extracting", unit="vol"):
        cubes = batch["cube"].to(device, non_blocking=True)
        names = [f"{img}" for img in batch["img_id"]]
        orig_ids = batch["orig_id"]
        mods = batch["mod"]


        cubes_np = cubes.detach().cpu().numpy()

        for cube_np, name, orig_id, mod in zip(cubes_np, names, orig_ids, mods):
            save_dir = output_dir / set_name / ty / mod
            samples_dir = save_dir / samples_dirname
            metadata_dir = save_dir / "metadata"
            samples_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)


            vol = cube_np[0].astype(np.float32)

            if do_hf:
                hf_fft_mag, hf_spatial = compute_FFTorIFFT(vol, cutoff=args.cutoff, kind="highpass")
                if mode_fft:
                    hf_data = np.log1p(hf_fft_mag).astype(np.float32)
                    component_name = f"{name}"
                    representation = "fft_log"
                    freq_label = "hf"
                else:
                    hf_data = (hf_spatial if hf_spatial.ndim == 3 else hf_spatial[0]).astype(np.float32)
                    if norm_mode == "sample-wise":
                        hf_data = minmax_normalize_volume(hf_data)
                    component_name = f"{name}"
                    representation = "ifft"
                    freq_label = "hf"

                sample_path = samples_dir / f"{component_name}{sample_suffix}"
                metadata_path = metadata_dir / f"{component_name}.csv"
                if sample_path.exists() and metadata_path.exists():
                    tqdm.write(f"[SKIP] {component_name} già presente")
                else:
                    if not sample_path.exists():
                        sample_path.parent.mkdir(parents=True, exist_ok=True)
                        if args.sample_format == "npy":
                            np.save(sample_path, hf_data.astype(np.float32))
                        else:
                            nib.save(nib.Nifti1Image(hf_data.astype(np.float32), np.eye(4)), sample_path)
                    if not metadata_path.exists():
                        metadata_path.parent.mkdir(parents=True, exist_ok=True)
                        hf_min = float(np.min(hf_data))
                        hf_max = float(np.max(hf_data))
                        metadata_df = pd.DataFrame([
                            {
                                "img_id": name,
                                "orig_id": orig_id,
                                "mod": mod,
                                "ty": ty,
                                "shape": str(hf_data.shape),
                                "representation": representation,
                                "freq": freq_label,
                                "cutoff": float(args.cutoff),
                                "min": hf_min,
                                "max": hf_max,
                                "path": str(sample_path),
                            }
                        ])
                        metadata_df.to_csv(metadata_path, index=False)

            if do_lf:
                lf_fft_mag, lf_spatial = compute_FFTorIFFT(vol, cutoff=args.cutoff, kind="lowpass")
                if mode_fft:
                    lf_data = np.log1p(lf_fft_mag).astype(np.float32)
                    component_name = f"{name}_lf_fft_log"
                    representation = "fft_log"
                    freq_label = "lf"
                else:
                    lf_data = lf_spatial.astype(np.float32)
                    component_name = f"{name}_lf_ifft"
                    representation = "ifft"
                    freq_label = "lf"

                sample_path = samples_dir / f"{component_name}{sample_suffix}"
                metadata_path = metadata_dir / f"{component_name}.csv"
                if sample_path.exists() and metadata_path.exists():
                    tqdm.write(f"[SKIP] {component_name} già presente")
                else:
                    if not sample_path.exists():
                        sample_path.parent.mkdir(parents=True, exist_ok=True)
                        if args.sample_format == "npy":
                            np.save(sample_path, lf_data.astype(np.float32))
                        else:
                            nib.save(nib.Nifti1Image(lf_data.astype(np.float32), np.eye(4)), sample_path)
                    if not metadata_path.exists():
                        metadata_path.parent.mkdir(parents=True, exist_ok=True)
                        lf_min = float(np.min(lf_data))
                        lf_max = float(np.max(lf_data))
                        metadata_df = pd.DataFrame([
                            {
                                "img_id": name,
                                "orig_id": orig_id,
                                "mod": mod,
                                "ty": ty,
                                "shape": str(lf_data.shape),
                                "representation": representation,
                                "freq": freq_label,
                                "cutoff": float(args.cutoff),
                                "min": lf_min,
                                "max": lf_max,
                                "path": str(sample_path),
                            }
                        ])
                        metadata_df.to_csv(metadata_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a radial frequency filter to 3D CT patches and save the filtered spectrum or spatial reconstruction."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing precomputed CT patches and their metadata.")
    parser.add_argument("--mods", default="real", help="Comma-separated generator names to process, such as real,cycle,diffusion.")
    parser.add_argument("--ty", default="train", help="Manipulation type to select from the ty column, such as inj or rem.")
    parser.add_argument("--set-name", default="train", help="Dataset split to process: train, valid, or test.")
    parser.add_argument("--sample-format", choices=["npy", "nii.gz"], default="npy", help="Input and output patch format: NumPy .npy or NIfTI .nii.gz.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of patches loaded per batch.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes.")
    parser.add_argument("--output-dir",type=Path, default="./fft_outputs", help="Root directory for filtered patches, per-patch metadata, and the configuration JSON.")
    parser.add_argument("--images-dir",type=Path, default="./fft3d_images", help="Reserved directory for FFT slice images; currently unused by the processing flow.")
    parser.add_argument("--img-cmap", default="magma", help="Reserved colormap for FFT slice images; currently unused by the processing flow.")
    parser.add_argument(
    "--cutoff",
    type=float,
    default=0.15,
    help="Radial cutoff as a fraction of the maximum radius in the centered 3D spectrum.",
)
    parser.add_argument(
        "--freq",
        choices=["hf", "lf", "both"],
        default="both",
        help="Frequency components to save: high-pass (hf), low-pass (lf), or both.",
    )
    parser.add_argument(
        "--mode",
        choices=["fft", "ifft"],
        default="fft",
        help="Save the log-scaled filtered FFT magnitude (fft) or the spatial reconstruction after inverse FFT (ifft).",
    )
    parser.add_argument(
        "--norm-mode",
        choices=["raw", "sample-wise"],
        default="raw",
        help="For high-pass IFFT output, preserve raw reconstructed values or apply per-sample min-max scaling to [0, 1].",
    )


    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_config_json(output_dir / args.set_name / args.ty, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = [m.strip() for m in args.mods.split(",") if m.strip()]

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

    save_FFTorIFFT_cubes(
        loader,
        args,
        output_dir,
        device=device,
        set_name=args.set_name,
        ty=args.ty,
    )


if __name__ == "__main__":
    main()
