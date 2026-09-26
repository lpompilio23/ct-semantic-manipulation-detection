from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError


def load_orig_ids(data_csv: Path) -> List[str]:

    df = pd.read_csv(data_csv)
    if "orig_id" not in df.columns:
        raise ValueError(f"Colonna 'orig_id' mancante in {data_csv}")
    return df["orig_id"].dropna().astype(str).drop_duplicates().tolist()


def resolve_lidc_scan_path(orig_id: str, lidc_root: Path) -> Path:

    scan_dir = lidc_root / str(orig_id)
    if not scan_dir.exists():
        raise FileNotFoundError(f"Cartella DICOM non trovata per orig_id={orig_id}: {scan_dir}")
    if not scan_dir.is_dir():
        raise NotADirectoryError(f"Path non directory per orig_id={orig_id}: {scan_dir}")
    return scan_dir


def find_dicom_files(scan_dir: Path) -> List[Path]:

    dicom_files = sorted(scan_dir.rglob("*.dcm"))
    if dicom_files:
        return dicom_files

    dicom_files = [p for p in sorted(scan_dir.rglob("*")) if p.is_file()]
    if not dicom_files:
        raise FileNotFoundError(f"Nessun file DICOM trovato in: {scan_dir}")
    return dicom_files


def _slice_sort_key(ds: pydicom.dataset.FileDataset, fallback_index: int) -> Tuple[float, int]:
    if hasattr(ds, "ImagePositionPatient"):
        try:
            return float(ds.ImagePositionPatient[2]), fallback_index
        except Exception:
            pass
    if hasattr(ds, "InstanceNumber"):
        try:
            return float(ds.InstanceNumber), fallback_index
        except Exception:
            pass
    return float(fallback_index), fallback_index


def read_dicom_volume(scan_dir: Path) -> Tuple[np.ndarray, float, float]:


    dicom_files = find_dicom_files(scan_dir)
    slices = []
    for idx, dicom_path in enumerate(dicom_files):
        try:
            ds = pydicom.dcmread(str(dicom_path), force=True)
        except InvalidDicomError as exc:
            raise InvalidDicomError(f"File non leggibile come DICOM: {dicom_path}") from exc
        slices.append((ds, dicom_path, idx))

    slices.sort(key=lambda item: _slice_sort_key(item[0], item[2]))
    datasets = [item[0] for item in slices]

    first_ds = datasets[0]
    slope = float(getattr(first_ds, "RescaleSlope", 1.0))
    intercept = float(getattr(first_ds, "RescaleIntercept", 0.0))

    pixel_arrays = []
    for ds in datasets:
        arr = ds.pixel_array
        pixel_arrays.append(arr)

    volume = np.stack(pixel_arrays, axis=0)
    return volume, slope, intercept


def compute_volume_stats(volume: np.ndarray, slope: float, intercept: float) -> dict:

    volume_f = volume.astype(np.float32, copy=False)
    hu_volume = volume_f * np.float32(slope) + np.float32(intercept)
    return {
        "shape": str(tuple(volume.shape)),
        "raw_min": float(np.min(volume_f)),
        "raw_max": float(np.max(volume_f)),
        "hu_min": float(np.min(hu_volume)),
        "hu_max": float(np.max(hu_volume)),
        "rescale_slope": float(slope),
        "rescale_intercept": float(intercept),
    }

def compute_scan_vmin_export(scan):


    if scan.dtype == np.uint16:
        return 0
    assert scan.dtype == np.int16
    list_min = [0, 512, 1024, 2000, 2048]
    vmin = -np.min(scan)
    if vmin not in list_min:
        f = min(5, scan.shape[0])
        o = np.stack(
            (
                scan[:f, :5, :5],
                scan[:f, :5, -5:],
                scan[:f, -5:, :5],
                scan[:f, -5:, -5:],
                scan[-f:, :5, :5],
                scan[-f:, :5, -5:],
                scan[-f:, -5:, :5],
                scan[-f:, -5:, -5:],
            ),
            0,
        )
        vmin = -np.median(o)
        if vmin not in list_min:
            vmin = list_min[np.argmax([np.count_nonzero(o == -1 * _) for _ in list_min])]

    assert vmin in list_min
    return int(vmin)

