import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch import from_numpy as numpy2torch

from preprocessing.tiff_utils import (
    apply_percentile,
    get_percentile_tiff_scan,
    get_shape_tiff_scan,
    load_tiff_scan,
)


def get_table_date(ty, mods, set_name="train", data_path="./data/data.csv", set_path="./data/sets.csv"):


    if isinstance(mods, str):
        mods = [m.strip() for m in mods.split(",") if m.strip()]
    if isinstance(ty, str):
        ty = [t.strip() for t in ty.split(",") if t.strip()]
    if isinstance(set_name, str):
        set_name = [s.strip() for s in set_name.split(",") if s.strip()]

    tab = pd.read_csv(data_path)
    tab = tab[tab["mod"].isin(mods)]
    tab = tab[tab["ty"].isin(ty)]

    sets = pd.read_csv(set_path)
    selected_sets = sets[sets["set"].isin(set_name)][["orig_id", "set"]]
    tab = tab.merge(selected_sets, on="orig_id", how="inner").reset_index(drop=True)
    return tab

def resample_to_spacing(img_data, original_spacing, target_spacing=(1.0, 1.0, 1.0)):
    resize_factors = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(np.array(img_data.shape) * resize_factors).astype(int)
    img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resampled_tensor = F.interpolate(img_tensor, size=tuple(new_shape), mode="trilinear", align_corners=False)
    resampled = resampled_tensor.squeeze(0).squeeze(0).numpy()
    return resampled.astype(np.float32), resize_factors


def resize_to_shape(img_data, target_shape):
    img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(img_tensor, size=tuple(target_shape), mode="trilinear", align_corners=False)
    resized = resized_tensor.squeeze(0).squeeze(0).numpy()
    return resized.astype(np.float32)


def pad_to_shape(arr, target_shape):
    pads = []
    for dim, targ in zip(arr.shape, target_shape):
        if dim < targ:
            total = targ - dim
            before = total // 2
            after = total - before
        else:
            before = after = 0
        pads.append((before, after))
    return np.pad(arr, pads, mode="constant", constant_values=0)


def center_crop(arr, target_shape):
    slices = []
    for dim, targ in zip(arr.shape, target_shape):
        if dim > targ:
            start = (dim - targ) // 2
            end = start + targ
        else:
            start = 0
            end = dim
        slices.append(slice(start, end))
    return arr[tuple(slices)]


def resolve_scan_path(row, real_path_scan, fake_path_scan):
    if row["mod"] == "real":
        return real_path_scan % (row["orig_id"])
    return fake_path_scan % (row["mod"], row["img_id"])


def load_dicom_rescale_table(tags_csv):
    tab = pd.read_csv(tags_csv)
    required_cols = {"scan_id", "rescale_slope", "rescale_intercept"}
    missing = required_cols.difference(tab.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV dei tag DICOM: {sorted(missing)}")
    return tab[["scan_id", "rescale_slope", "rescale_intercept"]].copy()


def apply_global_hu_normalization(scan, orig_id, dicom_rescale_tab):
    row = dicom_rescale_tab.loc[dicom_rescale_tab["scan_id"] == orig_id]
    if row.empty:
        raise KeyError(f"Tag DICOM non trovati per orig_id={orig_id}")

    slope = float(row["rescale_slope"].iloc[0])
    intercept = float(row["rescale_intercept"].iloc[0])
    scan_hu = np.float32(scan) * slope + intercept
    scan_hu = np.clip(scan_hu, -1000.0, 1000.0)
    scan_norm = (scan_hu + 1000.0) / 2000.0
    return scan_norm.astype(np.float32)


def load_lidc_scan_stats_table(stats_csv):
    tab = pd.read_csv(stats_csv)
    required_cols = {"orig_id", "min_export", "rescale_slope", "rescale_intercept"}
    missing = required_cols.difference(tab.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV LIDC scan stats: {sorted(missing)}")
    return tab[["orig_id", "min_export", "rescale_slope", "rescale_intercept"]].copy()


def _find_contiguous_true_blocks(mask_1d):


    active = np.flatnonzero(np.asarray(mask_1d, dtype=bool))
    if active.size == 0:
        return []


    blocks = []
    start = int(active[0])
    prev = int(active[0])


    for idx in active[1:]:
        idx = int(idx)


        if idx == prev + 1:
            prev = idx
            continue


        blocks.append((start, prev + 1))


        start = idx
        prev = idx

    blocks.append((start, prev + 1))
    return blocks


def _select_slice_z_from_mask(mask):


    if mask.ndim != 3:
        raise ValueError(f"Mask attesa 3D, ricevuto shape={mask.shape}")


    active_z = np.any(mask.astype(bool), axis=(1, 2))


    blocks = _find_contiguous_true_blocks(active_z)
    if not blocks:
        raise ValueError("Nessuna slice attiva trovata nella mask")


    start, end = max(blocks, key=lambda b: (b[1] - b[0], -b[0]))
    return int((start + end - 1) // 2)


def load_lidc_spacing_table(spacing_csv):
    tab = pd.read_csv(spacing_csv)
    required_cols = {"orig_id", "spacing_z", "spacing_y", "spacing_x"}
    missing = required_cols.difference(tab.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV spacing LIDC: {sorted(missing)}")
    return tab[["orig_id", "spacing_z", "spacing_y", "spacing_x"]].copy()


def restore_sv_and_convert_to_hu(scan, orig_id, lidc_stats_tab):

    row = lidc_stats_tab.loc[lidc_stats_tab["orig_id"] == orig_id]
    if row.empty:
        raise KeyError(f"Statistiche LIDC non trovate per orig_id={orig_id}")

    min_export = float(row["min_export"].iloc[0])
    slope = float(row["rescale_slope"].iloc[0])
    intercept = float(row["rescale_intercept"].iloc[0])


    scan_sv = np.float32(scan) - min_export

    scan_hu = scan_sv * slope + intercept
    return scan_hu.astype(np.float32)


def clip_hu_and_normalize_01(scan_hu, min_hu=-1000.0, max_hu=1000.0):
    scan_hu = np.clip(np.float32(scan_hu), min_hu, max_hu)
    scan_norm = (scan_hu - min_hu) / (max_hu - min_hu)
    return scan_norm.astype(np.float32)

class CTPatchExtractor(Dataset):


    def __init__(
        self,
        real_root,
        fake_root,
        tab,
        fallback_label_mods=None,
        value_space="stored-value",
        value_normalization="sample-wise",
        lidc_stats_csv=None,
    ):
        self.tab = tab.copy()
        self.real_path_scan = f"{real_root}/%s"
        self.fake_path_scan = f"{fake_root}/%s/scan/%s"
        self.fake_path_label = f"{fake_root}/%s/label/%s"

        self.value_space = value_space
        self.value_normalization = value_normalization
        if self.value_space not in {"stored-value", "hu"}:
            raise ValueError("value_space must be one of: 'stored-value', 'hu'")
        if self.value_normalization not in {"none", "sample-wise", "hu-clip-01"}:
            raise ValueError("value_normalization must be one of: 'none', 'sample-wise', 'hu-clip-01'")
        self.lidc_stats_csv = lidc_stats_csv
        if self.value_space == "hu" and self.lidc_stats_csv is None:
            raise ValueError("lidc_stats_csv is required when value_space='hu'")
        self.lidc_stats_tab = None


        self.target_shape = np.array([32,32,32], dtype=int)

        self.fallback_label_mods = fallback_label_mods or ["diffusion", "pix2pix", "cycle"]


        self.available_fake_mods = (
            self.tab[self.tab["mod"] != "real"]
            .groupby("img_id")["mod"]
            .apply(list)
            .to_dict()
        )

    def __len__(self):
        return len(self.tab)

    def _scan_dir(self, row):
        scan_dir = resolve_scan_path(row, self.real_path_scan, self.fake_path_scan)
        if row["mod"] == "real":
            mods = self.available_fake_mods.get(row["img_id"], [])
            if not mods:

                for candidate_mod in self.fallback_label_mods:
                    candidate_label = self.fake_path_label % (candidate_mod, row["img_id"])
                    if os.path.exists(candidate_label):
                        return scan_dir, candidate_label
                raise FileNotFoundError(f"Nessuna label fake trovata per img_id={row['img_id']}")
            chosen_mod = mods[0]
            return scan_dir, self.fake_path_label % (chosen_mod, row["img_id"])
        return scan_dir, self.fake_path_label % (row["mod"], row["img_id"])

    def __getitem__(self, idx):
        _data = self.tab.iloc[idx]
        scan_dirname,label_dirname = self._scan_dir(_data)
        coord = (_data["coord_z"], _data["coord_y"], _data["coord_x"])
        pid = _data["orig_id"]


        ct_scan = load_tiff_scan(scan_dirname, np.uint16)


        if self.value_space == "stored-value" and self.value_normalization == "none":
            norm_ct_scan = np.float32(ct_scan)


        elif self.value_space == "stored-value" and self.value_normalization == "sample-wise":
            ct_scan_perc = get_percentile_tiff_scan(scan_dirname, np.uint16)
            norm_ct_scan = apply_percentile(np.float32(ct_scan), *ct_scan_perc)


        elif self.value_space == "hu" and self.value_normalization == "none":

            if self.lidc_stats_tab is None:

                self.lidc_stats_tab = load_lidc_scan_stats_table(self.lidc_stats_csv)
            norm_ct_scan = restore_sv_and_convert_to_hu(ct_scan, pid, self.lidc_stats_tab)


        elif self.value_space == "hu" and self.value_normalization == "hu-clip-01":
            if self.lidc_stats_tab is None:

                self.lidc_stats_tab = load_lidc_scan_stats_table(self.lidc_stats_csv)
            scan_hu = restore_sv_and_convert_to_hu(ct_scan, pid, self.lidc_stats_tab)
            norm_ct_scan = clip_hu_and_normalize_01(scan_hu, min_hu=-1000.0, max_hu=1000.0)
        else:
            raise ValueError(
                f"Combinazione non supportata: value_space={self.value_space}, "
                f"value_normalization={self.value_normalization}"
            )


        ct_label = load_tiff_scan(label_dirname, np.bool_)

        if ct_label.shape[0] > norm_ct_scan.shape[0]:
            ct_label = ct_label[:-1, ...]


        if ct_label.shape != norm_ct_scan.shape:
            raise ValueError(
                f"Shape mismatch scan {norm_ct_scan.shape} vs label {ct_label.shape} "
                f"for img_id={_data['img_id']} orig_id={_data['orig_id']} mod={_data['mod']}"
            )


        masked_scan = norm_ct_scan * ct_label.astype(np.float32)

        nz = np.nonzero(masked_scan)

        if len(nz[0]) == 0:
            raise ValueError(f"Nessun voxel marcato nella label per img_id={_data['img_id']}")
        z_min, z_max = nz[0].min(), nz[0].max() + 1
        y_min, y_max = nz[1].min(), nz[1].max() + 1
        x_min, x_max = nz[2].min(), nz[2].max() + 1
        cropped = masked_scan[z_min:z_max, y_min:y_max, x_min:x_max]


        patch_min = float(cropped.min())
        patch_max = float(cropped.max())

        spacing_tab = pd.read_csv("./data/LIDC.csv")
        spacing_row = spacing_tab.loc[spacing_tab["orig_id"] == pid]
        original_spacing = (
            float(spacing_row["spacing_z"].values[0]),
            float(spacing_row["spacing_y"].values[0]),
            float(spacing_row["spacing_x"].values[0]),
        )
        scaled_cropped, scaling_factors = resample_to_spacing(cropped, original_spacing=original_spacing, target_spacing= (1.0,1.0,1.0))


        resized = pad_to_shape(scaled_cropped, self.target_shape)
        resized = center_crop(resized, self.target_shape)

        cube = numpy2torch(resized[None, ...]).float()


        return dict(
            cube=cube,
            orig_id=_data["orig_id"],
            img_id=_data["img_id"],
            mod=_data["mod"],
            ty=_data["ty"],
            set=_data["set"],
            shape=cube.shape,
            patch_min=patch_min,
            patch_max=patch_max,
        )
