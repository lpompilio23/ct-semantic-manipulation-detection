from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import nibabel as nib
except ImportError:
    nib = None


SUPPORTED_SAMPLE_FORMATS = {"npy", "nii.gz"}
SUPPORTED_LABEL_MODES = {"binary", "multiclass"}

MULTICLASS_LABELS = {
    "real": 0,
    "cycle": 1,
    "diffusion": 2,
    "pix2pix": 3,
}

CLASS_NAMES = ["real", "cycle", "diffusion", "pix2pix"]


def _normalize_sample_format(sample_format: str) -> str:
    sample_format = sample_format.lower().strip()
    if sample_format not in SUPPORTED_SAMPLE_FORMATS:
        raise ValueError(
            f"Unsupported sample_format={sample_format!r}. "
            f"Expected one of {sorted(SUPPORTED_SAMPLE_FORMATS)}"
        )
    return sample_format


def _sample_dirname(sample_format: str) -> str:
    return "samples" if sample_format == "npy" else "samples_nii"


def _sample_suffix(sample_format: str) -> str:
    return ".npy" if sample_format == "npy" else ".nii.gz"


def _normalize_label_mode(label_mode: str) -> str:
    label_mode = label_mode.lower().strip()
    if label_mode not in SUPPORTED_LABEL_MODES:
        raise ValueError(
            f"Unsupported label_mode={label_mode!r}. "
            f"Expected one of {sorted(SUPPORTED_LABEL_MODES)}"
        )
    return label_mode


def _label_from_mod(mod: str, label_mode: str, real_mod: str) -> int:
    if label_mode == "binary":
        return 0 if mod == real_mod else 1

    if mod not in MULTICLASS_LABELS:
        raise ValueError(
            f"Unsupported mod={mod!r} for multiclass labels. "
            f"Expected one of {sorted(MULTICLASS_LABELS)}"
        )
    return MULTICLASS_LABELS[mod]


def _load_sample(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.name.endswith(".nii.gz"):
        if nib is None:
            raise ImportError("nibabel is required to load .nii.gz samples")
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    raise ValueError(f"Unsupported sample file: {path}")


def _load_embedding(path: Path) -> torch.Tensor:
    if path.suffix != ".pt":
        raise ValueError(f"Unsupported embedding file: {path}")
    return torch.load(path, map_location="cpu")


def _read_metadata_files(metadata_files: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in metadata_files:
        frame = pd.read_csv(csv_path)
        frame["metadata_csv"] = str(csv_path)


        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_metadata_files(
    root: Path,
    split: Optional[str] = None,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
) -> List[Path]:
    root = Path(root)
    search_root = root if split is None else root / split
    if not search_root.exists():
        raise FileNotFoundError(f"Root not found: {search_root}")

    mods_set = set(mods) if mods is not None else None
    metadata_files: List[Path] = []
    for csv_path in search_root.rglob("metadata/*.csv"):


        parts = csv_path.relative_to(root).parts


        if split is not None and (len(parts) < 4 or parts[0] != split):
            continue
        if ty is not None and (len(parts) < 4 or parts[1] != ty):
            continue
        if mods_set is not None and (len(parts) < 4 or parts[2] not in mods_set):
            continue
        metadata_files.append(csv_path)
    return sorted(metadata_files)


def _select_subset_indices(
    metadata: pd.DataFrame,
    mods: Optional[Iterable[str]],
    real_mod: str,
    max_samples_real: Optional[int] = None,
    max_samples_fake: Optional[int] = None,
) -> List[int]:


    mods_list = list(mods) if mods is not None else sorted(metadata["mod"].dropna().unique().tolist())
    selected: List[int] = []
    for mod in mods_list:

        mod_indices = metadata.index[metadata["mod"] == mod].tolist()
        if not mod_indices:
            continue


        limit = max_samples_real if mod == real_mod else max_samples_fake
        if limit is not None:
            mod_indices = mod_indices[:limit]

        selected.extend(mod_indices)
    return selected

def _resolve_subset_item(dataset, idx: int):
    if isinstance(dataset, Subset):
        return _resolve_subset_item(dataset.dataset, dataset.indices[idx])
    return dataset[idx]


def _metadata_global_min_max(metadata: pd.DataFrame) -> Tuple[float, float]:


    if {"min", "max"}.issubset(metadata.columns):
        min_col, max_col = "min", "max"
    elif {"patch_min", "patch_max"}.issubset(metadata.columns):
        min_col, max_col = "patch_min", "patch_max"
    else:
        raise ValueError("Metadata must contain either min/max or patch_min/patch_max columns")

    mins = pd.to_numeric(metadata[min_col], errors="coerce")
    maxs = pd.to_numeric(metadata[max_col], errors="coerce")

    global_min = float(mins.min())
    global_max = float(maxs.max())

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("Unable to compute finite global min/max from metadata")
    return global_min, global_max


class CubeDataset(Dataset):


    def __init__(
        self,
        root: str | Path,
        split: str,
        ty: Optional[str] = None,
        mods: Optional[Iterable[str]] = None,
        sample_format: str = "npy",
        label_mode: str = "binary",
        real_mod: str = "real",
    ):
        self.root = Path(root)
        self.split = split
        self.ty = ty
        self.mods = list(mods) if mods is not None else None
        self.sample_format = _normalize_sample_format(sample_format)
        self.label_mode = _normalize_label_mode(label_mode)
        self.real_mod = real_mod
        self.samples_dirname = _sample_dirname(self.sample_format)
        self.sample_suffix = _sample_suffix(self.sample_format)


        metadata_files = collect_metadata_files(
            self.root,
            split=self.split,
            ty=self.ty,
            mods=self.mods,
        )
        if not metadata_files:
            raise FileNotFoundError(
                f"No metadata CSV files found for split={split!r}, ty={ty!r}, mods={self.mods!r}"
            )


        self.metadata = _read_metadata_files(metadata_files).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def _sample_path(self, metadata_csv: Path, sample_name: str) -> Path:
        save_dir = metadata_csv.parent.parent


        return save_dir / self.samples_dirname / f"{sample_name}{self.sample_suffix}"

    def __getitem__(self, idx: int):

        row = self.metadata.iloc[idx]

        metadata_csv = Path(row["metadata_csv"])

        sample_name = metadata_csv.stem
        sample_path = self._sample_path(metadata_csv, sample_name)

        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")


        cube_np = _load_sample(sample_path)


        cube_min = float(np.min(cube_np))
        cube_max = float(np.max(cube_np))
        eps = 1e-5
        if cube_min < -eps or cube_max > 1.0 + eps:
            raise ValueError(
                f"Sample outside [0, 1] range: path={sample_path}, min={cube_min:.6f}, max={cube_max:.6f}"
            )

        cube = torch.from_numpy(cube_np).float()
        if cube.ndim == 3:
            cube = cube.unsqueeze(0)

        label = _label_from_mod(str(row["mod"]), self.label_mode, self.real_mod)

        patch_min_col = "patch_min" if "patch_min" in row else "min"
        patch_max_col = "patch_max" if "patch_max" in row else "max"

        return {
            "cube": cube,
            "label": label,
            "img_id": row["img_id"],
            "orig_id": row["orig_id"],
            "split": self.split,
            "mod": row["mod"],
            "ty": row["ty"],
            "shape": row["shape"],
            "patch_min": float(row[patch_min_col]),
            "patch_max": float(row[patch_max_col]),
            "path": str(sample_path),
        }


def get_dataloader(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    sample_format: str = "npy",
    label_mode: str = "binary",
    real_mod: str = "real",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):
    dataset = CubeDataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
        sample_format=sample_format,
        label_mode=label_mode,
        real_mod=real_mod,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )

def get_subset_dataset(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    real_mod: str = "real",
    sample_format: str = "npy",
    label_mode: str = "binary",
    max_samples_real: Optional[int] = None,
    max_samples_fake: Optional[int] = None,
) -> Subset:


    dataset = CubeDataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
        sample_format=sample_format,
        label_mode=label_mode,
        real_mod=real_mod,
    )

    indices = _select_subset_indices(
        dataset.metadata,
        mods=mods,
        real_mod=real_mod,
        max_samples_real=max_samples_real,
        max_samples_fake=max_samples_fake,
    )
    if not indices:
        raise FileNotFoundError(
        f"No subset samples found for split={split!r}, ty={ty!r}, mods={list(mods) if mods is not None else None!r}"
        )
    return Subset(dataset, indices)

def get_subset_dataloader(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    real_mod: str = "real",
    sample_format: str = "npy",
    label_mode: str = "binary",
    max_samples_real: Optional[int] = None,
    max_samples_fake: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):


    dataset = get_subset_dataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
        real_mod=real_mod,
        sample_format=sample_format,
        label_mode=label_mode,
        max_samples_real=max_samples_real,
        max_samples_fake=max_samples_fake,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


class FFTorIFFTCubeDataset(CubeDataset):


    def __init__(
        self,
        root: str | Path,
        split: str,
        ty: Optional[str] = None,
        mods: Optional[Iterable[str]] = None,
        sample_format: str = "npy",
        label_mode: str = "binary",
        real_mod: str = "real",
        norm_mode: bool = True,
    ):


        super().__init__(
            root=root,
            split=split,
            ty=ty,
            mods=mods,
            sample_format=sample_format,
            label_mode=label_mode,
            real_mod=real_mod,
        )
        self.norm_mode = norm_mode

        if self.norm_mode:


            global_metadata_files = collect_metadata_files(
                self.root,
                split=None,
                ty=None,
                mods=None,
            )
            if not global_metadata_files:
                raise FileNotFoundError(
                    f"No FFT metadata CSV files found under root={self.root!r}"
                )
            global_metadata = _read_metadata_files(global_metadata_files).reset_index(drop=True)
            self.global_min, self.global_max = _metadata_global_min_max(global_metadata)
        else:
            self.global_min = None
            self.global_max = None

    def _normalize_fft_volume(self, volume: np.ndarray) -> np.ndarray:
        if not self.norm_mode or self.global_min is None or self.global_max is None:
            return volume.astype(np.float32)
        if self.global_max <= self.global_min:
            return np.zeros_like(volume, dtype=np.float32)
        normalized = (volume.astype(np.float32) - self.global_min) / (
            self.global_max - self.global_min
        )
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        metadata_csv = Path(row["metadata_csv"])
        sample_name = metadata_csv.stem
        sample_path = self._sample_path(metadata_csv, sample_name)

        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")

        cube_np = _load_sample(sample_path)
        if self.norm_mode:
            cube_np = self._normalize_fft_volume(cube_np)

        cube = torch.from_numpy(cube_np).float()
        if cube.ndim == 3:
            cube = cube.unsqueeze(0)

        label = _label_from_mod(str(row["mod"]), self.label_mode, self.real_mod)

        patch_min_col = "patch_min" if "patch_min" in row else "min"
        patch_max_col = "patch_max" if "patch_max" in row else "max"

        return {
            "cube": cube,
            "label": label,
            "img_id": row["img_id"],
            "orig_id": row["orig_id"],
            "split": self.split,
            "mod": row["mod"],
            "ty": row["ty"],
            "shape": row["shape"],
            "patch_min": float(row[patch_min_col]),
            "patch_max": float(row[patch_max_col]),
            "path": str(sample_path),
        }


def get_FFTorIFFTdataloader(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    sample_format: str = "npy",
    label_mode: str = "binary",
    real_mod: str = "real",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
    norm_mode: bool = True,
):
    dataset = FFTorIFFTCubeDataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
        sample_format=sample_format,
        label_mode=label_mode,
        real_mod=real_mod,
        norm_mode=norm_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


def get_subset_FFTorIFFT_dataset(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    real_mod: str = "real",
    sample_format: str = "npy",
    label_mode: str = "binary",
    max_samples_real: Optional[int] = None,
    max_samples_fake: Optional[int] = None,
    norm_mode: bool = True,
) -> Subset:


    dataset = FFTorIFFTCubeDataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
        sample_format=sample_format,
        label_mode=label_mode,
        real_mod=real_mod,
        norm_mode=norm_mode,
    )
    indices = _select_subset_indices(
        dataset.metadata,
        mods=mods,
        real_mod=real_mod,
        max_samples_real=max_samples_real,
        max_samples_fake=max_samples_fake,
    )
    if not indices:
        raise FileNotFoundError(
            f"No subset samples found for split={split!r}, ty={ty!r}, mods={list(mods) if mods is not None else None!r}"
        )
    return Subset(dataset, indices)

def get_subset_FFTorIFFTdataloader(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    real_mod: str = "real",
    sample_format: str = "npy",
    label_mode: str = "binary",
    max_samples_real: Optional[int] = None,
    max_samples_fake: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
    norm_mode: bool = True,
):


    dataset = get_subset_FFTorIFFT_dataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
        real_mod=real_mod,
        sample_format=sample_format,
        label_mode=label_mode,
        max_samples_real=max_samples_real,
        max_samples_fake=max_samples_fake,
        norm_mode=norm_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )



class EmbeddingDataset(Dataset):


    def __init__(
        self,
        root: str | Path,
        split: str,
        ty: Optional[str] = None,
        mods: Optional[Iterable[str]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.ty = ty
        self.mods = list(mods) if mods is not None else None


        metadata_files = collect_metadata_files(
            self.root,
            split=self.split,
            ty=self.ty,
            mods=self.mods,
        )
        if not metadata_files:
            raise FileNotFoundError(
                f"No metadata CSV files found for split={split!r}, ty={ty!r}, mods={self.mods!r}"
            )


        self.metadata = _read_metadata_files(metadata_files).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def _embedding_path(self, metadata_csv: Path, sample_name: str) -> Path:
        save_dir = metadata_csv.parent.parent
        return save_dir / "samples" / f"{sample_name}.pt"

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        metadata_csv = Path(row["metadata_csv"])
        sample_name = metadata_csv.stem
        embedding_path = self._embedding_path(metadata_csv, sample_name)


        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

        embedding = _load_embedding(embedding_path)
        if embedding.ndim == 0:
            embedding = embedding.unsqueeze(0)

        label = int(row["label"]) if "label" in row else (0 if row["mod"] == "real" else 1)

        return {
            "embedding": embedding.float(),
            "label": label,
            "img_id": row["img_id"],
            "orig_id": row["orig_id"],
            "split": row["split"] if "split" in row else self.split,
            "mod": row["mod"],
            "ty": row["ty"],
            "path": str(embedding_path),
        }


def get_embedding_dataloader(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):
    dataset = EmbeddingDataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


def get_subset_embedding_dataset(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    real_mod: str = "real",
    max_samples_real: Optional[int] = None,
    max_samples_fake: Optional[int] = None,
) -> Subset:
    dataset = EmbeddingDataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
    )
    indices = _select_subset_indices(
        dataset.metadata,
        mods=mods,
        real_mod=real_mod,
        max_samples_real=max_samples_real,
        max_samples_fake=max_samples_fake,
    )
    if not indices:
        raise FileNotFoundError(
            f"No subset samples found for split={split!r}, ty={ty!r}, mods={list(mods) if mods is not None else None!r}"
        )
    return Subset(dataset, indices)


def get_subset_embedding_dataloader(
    root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    real_mod: str = "real",
    max_samples_real: Optional[int] = None,
    max_samples_fake: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):
    dataset = get_subset_embedding_dataset(
        root=root,
        split=split,
        ty=ty,
        mods=mods,
        real_mod=real_mod,
        max_samples_real=max_samples_real,
        max_samples_fake=max_samples_fake,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


class PairedEmbeddingSubsetDataset(Dataset):

    def __init__(
        self,
        image_root: str | Path,
        ifft_root: str | Path,
        split: str,
        ty: Optional[str] = None,
        mods: Optional[Iterable[str]] = None,
        real_mod: str = "real",
        max_samples_real: Optional[int] = None,
        max_samples_fake: Optional[int] = None,
    ):
        image_subset = get_subset_embedding_dataset(
            root=image_root,
            split=split,
            ty=ty,
            mods=mods,
            real_mod=real_mod,
            max_samples_real=max_samples_real,
            max_samples_fake=max_samples_fake,
        )
        ifft_subset = get_subset_embedding_dataset(
            root=ifft_root,
            split=split,
            ty=ty,
            mods=mods,
            real_mod=real_mod,
            max_samples_real=max_samples_real,
            max_samples_fake=max_samples_fake,
        )


        image_metadata = image_subset.dataset.metadata.iloc[image_subset.indices].reset_index(drop=True)
        ifft_metadata = ifft_subset.dataset.metadata.iloc[ifft_subset.indices].reset_index(drop=True)


        def _row_key(row: pd.Series) -> Tuple[str, str, str, str, str]:
            return (
                str(row["img_id"]),
                str(row["orig_id"]),
                str(row["split"] if "split" in row else split),
                str(row["mod"]),
                str(row["ty"]),
            )


        image_map = { _row_key(image_metadata.iloc[i]): image_subset[i] for i in range(len(image_subset)) }
        ifft_map = { _row_key(ifft_metadata.iloc[i]): ifft_subset[i] for i in range(len(ifft_subset)) }


        image_keys = set(image_map)
        ifft_keys = set(ifft_map)
        if image_keys != ifft_keys:
            only_image = sorted(image_keys - ifft_keys)
            only_ifft = sorted(ifft_keys - image_keys)
            raise ValueError(
                "Mismatch tra i campioni dei due branch paired embedding. "
                f"Solo image: {only_image[:5]}, solo ifft: {only_ifft[:5]}"
            )


        self.records: List[dict] = []
        for key in sorted(image_keys):
            image_rec = image_map[key]
            ifft_rec = ifft_map[key]
            if image_rec["label"] != ifft_rec["label"]:
                raise ValueError(
                    f"Label mismatch for sample {key}: image={image_rec['label']} ifft={ifft_rec['label']}"
                )


            self.records.append(
                {
                    "img_id": image_rec["img_id"],
                    "orig_id": image_rec["orig_id"],
                    "split": image_rec["split"],
                    "mod": image_rec["mod"],
                    "ty": image_rec["ty"],
                    "label": int(image_rec["label"]),
                    "image_embedding": image_rec["embedding"].detach().cpu().float().view(-1),
                    "ifft_embedding": ifft_rec["embedding"].detach().cpu().float().view(-1),
                    "image_path": image_rec["path"],
                    "ifft_path": ifft_rec["path"],
                }
            )

        if not self.records:
            raise FileNotFoundError(
                f"No paired samples found for split={split!r}, ty={ty!r}, mods={list(mods) if mods is not None else None!r}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        return {
            "image_embedding": record["image_embedding"],
            "ifft_embedding": record["ifft_embedding"],
            "label": torch.tensor(record["label"], dtype=torch.float32),
            "img_id": record["img_id"],
            "orig_id": record["orig_id"],
            "split": record["split"],
            "mod": record["mod"],
            "ty": record["ty"],
            "image_path": record["image_path"],
            "ifft_path": record["ifft_path"],
        }


def get_paired_embedding_dataloader(
    image_root: str | Path,
    ifft_root: str | Path,
    split: str,
    ty: Optional[str] = None,
    mods: Optional[Iterable[str]] = None,
    real_mod: str = "real",
    max_samples_real: Optional[int] = None,
    max_samples_fake: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
):
    dataset = PairedEmbeddingSubsetDataset(
        image_root=image_root,
        ifft_root=ifft_root,
        split=split,
        ty=ty,
        mods=mods,
        real_mod=real_mod,
        max_samples_real=max_samples_real,
        max_samples_fake=max_samples_fake,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
