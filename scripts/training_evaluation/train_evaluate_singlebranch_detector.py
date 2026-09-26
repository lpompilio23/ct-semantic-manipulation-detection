import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict
import csv
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader, Subset
import json
import matplotlib.pyplot as plt
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataloader import get_subset_embedding_dataloader


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def best_key_mode(best_key: str) -> str:
    return "min" if best_key == "val_loss" else "max"


def initial_best_value(best_key: str) -> float:
    return float("inf") if best_key_mode(best_key) == "min" else 0.0


def is_better_value(best_key: str, current_value: float, best_value: float, min_delta: float) -> bool:
    if best_key_mode(best_key) == "min":
        return (best_value - current_value) > min_delta
    return (current_value - best_value) > min_delta

@torch.no_grad()
def evaluate_fc(model: nn.Module,
                loader: DataLoader,
                device: torch.device,
                threshold: float = 0.5,
                return_rows: bool = False):

    model.eval()
    y_true, y_pred = [], []
    rows: List[dict] = []
    total_loss = 0.0
    num_batches = 0
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            feats = batch["embedding"]
            labels = batch["label"]
            ids = batch.get("img_id")
            splits = batch.get("split")
            mods = batch.get("mod")
            tys = batch.get("ty")

            feats = feats.to(device)
            labels = labels.to(device).float()
            logits = model(feats).squeeze(1)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits)
            probs_np = probs.detach().cpu().numpy().tolist()
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(probs_np)

            if return_rows and ids is not None:
                preds = (probs > threshold).long()
                for sample_id, sample_split, sample_mod, sample_ty, true_label, pred_label, prob_value in zip(
                    ids, splits, mods, tys, labels, preds, probs_np
                ):
                    true_int = int(true_label.item())
                    pred_int = int(pred_label.item())
                    prob_int = float(prob_value)
                    if true_int == 0 and pred_int == 0:
                        tag = "C0"
                    elif true_int == 1 and pred_int == 1:
                        tag = "C1"
                    elif true_int == 1 and pred_int == 0:
                        tag = "S0"
                    else:
                        tag = "S1"
                    rows.append({
                        "id": str(sample_id),
                        "split": str(sample_split),
                        "mod": str(sample_mod),
                        "ty": str(sample_ty),
                        "prob": prob_int,
                        "pred": pred_int,
                        "image_tag": tag
                    })

    val_loss = total_loss / max(num_batches, 1)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    ap = average_precision_score(y_true_np, y_pred_np)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true_np, y_pred_np, threshold)
    if return_rows:
        return ap, r_acc0, f_acc0, acc0, val_loss, rows
    return ap, r_acc0, f_acc0, acc0, val_loss


def save_predictions_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "mod", "ty", "prob", "pred", "image_tag"])
        writer.writeheader()
        writer.writerows(rows)


def _resolve_subset_item(dataset, idx: int):

    if isinstance(dataset, Subset):
        return _resolve_subset_item(dataset.dataset, dataset.indices[idx])
    return dataset[idx]


def collect_loaded_samples(loader: DataLoader, source_split: str) -> pd.DataFrame:

    rows: List[dict] = []
    dataset = loader.dataset
    for idx in range(len(dataset)):
        sample = _resolve_subset_item(dataset, idx)
        rows.append(
            {
                "img_id": str(sample.get("img_id")),
                "source_split": source_split,
                "dataset_split": str(sample.get("split")),
                "mod": str(sample.get("mod")),
                "ty": str(sample.get("ty")),
            }
        )
    return pd.DataFrame(rows, columns=["img_id", "source_split", "dataset_split", "mod", "ty"])


def save_loaded_samples_csv(output_path: Path, loader_map: List[Tuple[str, DataLoader]]) -> None:

    frames = [collect_loaded_samples(loader, source_split) for source_split, loader in loader_map]
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["img_id", "source_split", "dataset_split", "mod", "ty"])
    if not df.empty:
        df = df.sort_values(["source_split", "dataset_split", "mod", "img_id"], kind="stable").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _normalize_loaded_samples_df(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = ["img_id", "source_split", "dataset_split", "mod", "ty"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"loaded_samples.csv missing columns: {missing}")
    normalized = df.copy()
    for col in expected_cols:
        normalized[col] = normalized[col].astype(str)
    normalized = normalized[expected_cols]
    return normalized.sort_values(expected_cols, kind="stable").reset_index(drop=True)


def compare_loaded_samples_csv(current_path: Path, reference_path: Path, report_path: Optional[Path] = None) -> None:
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference loaded_samples CSV not found: {reference_path}. "
            f"Create it before running the leave1out experiment."
        )

    current_df = _normalize_loaded_samples_df(pd.read_csv(current_path))
    reference_df = _normalize_loaded_samples_df(pd.read_csv(reference_path))

    if current_df.equals(reference_df):
        print(f"[loaded_samples] Match confirmed for {current_path.name} vs {reference_path.name}")
        return

    diff_lines: List[str] = []
    diff_lines.append(f"Current CSV:    {current_path}")
    diff_lines.append(f"Reference CSV:  {reference_path}")
    diff_lines.append(f"Current rows:   {len(current_df)}")
    diff_lines.append(f"Reference rows: {len(reference_df)}")

    current_only = current_df.merge(reference_df, on=["img_id", "source_split", "dataset_split", "mod", "ty"], how="outer", indicator=True)
    current_only = current_only[current_only["_merge"] != "both"]

    diff_lines.append(f"Different rows: {len(current_only)}")
    if not current_only.empty:
        diff_lines.append("First differences:")
        diff_lines.extend(current_only.head(20).to_string(index=False).splitlines())

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(diff_lines) + "\n")

    raise ValueError(
        f"loaded_samples mismatch for {current_path.name} against {reference_path.name}. "
        f"See {report_path if report_path is not None else 'no report generated'} for details."
    )


def parse_mod_list(mod_list: str) -> List[str]:
    mods = [m.strip() for m in mod_list.split(",") if m.strip()]
    if not mods:
        raise ValueError("Specificare almeno un valore in --train_mods o --extra_val_pairs")
    return mods


def pick_max_samples_for_set_ty(set_name: str,
                                ty: str,
                                args,
                                is_ood: bool) -> Tuple[Optional[int], Optional[int]]:

    name = set_name.lower()
    if name.startswith("train"):
        split = "train"
    elif name.startswith("val") or name.startswith("valid"):
        split = "val"
    elif name.startswith("test"):
        split = "test"
    else:
        split = None
    if split is None:
        return None, None

    prefix = f"max_samples_{split}_{ty}"
    if is_ood:
        max_real = getattr(args, f"{prefix}_ood_real", None)
        max_fake = getattr(args, f"{prefix}_ood_fake", None)
    else:
        max_real = getattr(args, f"{prefix}_real", None)
        max_fake = getattr(args, f"{prefix}_fake", None)
    return max_real, max_fake


def _mlp_block(in_dim: int, out_dim: int, use_bn: bool, dropout: float) -> List[nn.Module]:
    layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if use_bn:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(nn.ReLU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


def build_classifier(feature_dim: int, arch: str, hidden_dim: int, dropout: float) -> nn.Module:
    arch = arch.lower()
    if arch == "linear":
        return nn.Linear(feature_dim, 1)
    if arch == "mlp3":
        layers: List[nn.Module] = []
        layers += _mlp_block(feature_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers += _mlp_block(hidden_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)
    if arch == "mlp5":
        layers = []
        layers += _mlp_block(feature_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers += _mlp_block(hidden_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers += _mlp_block(hidden_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers += _mlp_block(hidden_dim, hidden_dim, use_bn=True, dropout=dropout)
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)
    raise ValueError("fc_arch deve essere in [linear, mlp3, mlp5]")


def train_fc_head(train_feats: DataLoader,
                  val_feats: DataLoader,
                  args,
                  min_lr: float,
                  device: torch.device,
                  output_dir: Path):

    set_seed(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    first_batch = next(iter(train_feats))
    if isinstance(first_batch, dict):
        feature_dim = int(first_batch["embedding"].shape[1])
    else:
        feature_dim = int(first_batch[0].shape[1])
    model = build_classifier(feature_dim,
                             arch=args.fc_arch,
                             hidden_dim=args.fc_hidden_dim,
                             dropout=args.fc_dropout).to(device)

    if args.optim == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr,
                                      betas=(args.beta1, 0.999),
                                      weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=0.0,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError("optim deve essere in [adam, sgd]")

    loss_fn = nn.BCEWithLogitsLoss()


    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=min_lr,
        )


    best_val_loss = float("inf")
    best_value = initial_best_value(args.best_key)
    best_epoch = -1
    best_loss_epoch = -1
    val_loss_no_improve = 0


    train_losses: List[float] = []
    val_losses: List[float] = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    (train_line,) = ax.plot([], [], label="train_loss")
    (val_line,) = ax.plot([], [], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    for epoch in range(args.niter):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch in train_feats:

            feats = batch["embedding"]
            labels = batch["label"]

            feats = feats.to(device)
            labels = labels.to(device).float()
            logits = model(feats).squeeze(1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        train_loss_epoch = epoch_train_loss / num_batches


        ap, r_acc, f_acc, acc, val_loss = evaluate_fc(model, val_feats, device)
        val_metrics = {
            "val_loss": val_loss,
            "val_ap": ap,
            "val_r_acc": r_acc,
            "val_f_acc": f_acc,
            "val_acc": acc,
        }
        current_best_metric = val_metrics[args.best_key]

        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss)

        val_loss_improved = (epoch == 0) or ((best_val_loss - val_loss) > args.es_min_delta)
        if val_loss_improved:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            val_loss_no_improve = 0
        else:
            val_loss_no_improve += 1

        improved = (epoch == 0) or is_better_value(
            args.best_key,
            current_best_metric,
            best_value,
            args.es_min_delta,
        )

        if improved:
            best_value = current_best_metric
            best_epoch = epoch
            best_metadata = {
                "best_key": args.best_key,
                "best_value": float(best_value),
                "best_epoch": int(best_epoch),
            }
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "arch": args.fc_arch,
                    "best_key": args.best_key,
                    "best_value": best_value,
                    "best_epoch": best_epoch,
                },
                output_dir / "best_model.pt",
            )
            with open(output_dir / "best_params_model.json", "w") as f:
                json.dump(best_metadata, f, indent=2)

        if scheduler is not None:
            scheduler.step(val_loss)

        train_line.set_data(range(len(train_losses)), train_losses)
        val_line.set_data(range(len(val_losses)), val_losses)
        ax.relim()
        ax.autoscale_view()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_live.png", dpi=150)

        if (epoch % args.save_epoch_freq == 0) or (epoch == args.niter - 1):
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "arch": args.fc_arch,
                },
                output_dir / f"model_{epoch}.pt",
            )

        if args.early_stop and val_loss_no_improve >= args.es_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch {best_loss_epoch} (val_loss={best_val_loss:.6f}).")
            break


    return model

def parse_gpu_ids(gpu_ids: str) -> List[int]:
    ids: List[int] = []
    for token in gpu_ids.split(','):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value >= 0:
            ids.append(value)
    return ids


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


def add_ty_limit_args(parser: argparse.ArgumentParser, ty: str) -> None:
    parser.add_argument(f"--max_samples_train_{ty}_real", type=int, default=1217 if ty == "inj" else None, help="Maximum real samples in the training split; default: 1217 for inj, unlimited otherwise.")
    parser.add_argument(f"--max_samples_train_{ty}_fake", type=int, default=609 if ty == "inj" else None, help="Maximum fake samples per generator in the training split; default: 609 for inj, unlimited otherwise.")
    parser.add_argument(f"--max_samples_val_{ty}_real", type=int, default=437 if ty == "inj" else None, help="Maximum real samples in the validation split; default: 437 for inj, unlimited otherwise.")
    parser.add_argument(f"--max_samples_val_{ty}_fake", type=int, default=219 if ty == "inj" else None, help="Maximum fake samples per generator in the validation split; default: 219 for inj, unlimited otherwise.")
    parser.add_argument(f"--max_samples_val_{ty}_ood_real", type=int, default=345 if ty == "inj" else None, help="Maximum real samples in the OOD validation split; default: 345 for inj, unlimited otherwise.")
    parser.add_argument(f"--max_samples_val_{ty}_ood_fake", type=int, default=345 if ty == "inj" else None, help="Maximum fake samples for the held-out generator in the OOD validation split; default: 345 for inj, unlimited otherwise.")
    parser.add_argument(f"--max_samples_test_{ty}_real", type=int, default=479 if ty == "inj" else None, help="Maximum real samples in the ID test split; default: 479 for inj, unlimited otherwise.")
    parser.add_argument(f"--max_samples_test_{ty}_fake", type=int, default=240 if ty == "inj" else None, help="Maximum fake samples per generator in the ID test split; default: 240 for inj, unlimited otherwise.")
    parser.add_argument(f"--max_samples_test_{ty}_ood_real", type=int, default=421 if ty == "inj" else None, help="Maximum real samples in the OOD test split; default: 421 for inj, unlimited otherwise.")
    parser.add_argument(f"--max_samples_test_{ty}_ood_fake", type=int, default=421 if ty == "inj" else None, help="Maximum fake samples for the held-out generator in the OOD test split; default: 421 for inj, unlimited otherwise.")


def mean_std_dict(values: Dict[str, List[float]]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    stats = {}
    for k, v in values.items():
        if len(v) == 0:
            stats[k] = (None, None)
        else:
            arr = np.array(v, dtype=float)
            stats[k] = (float(np.mean(arr)), float(np.std(arr)))
    return stats


SUMMARY_METRICS = [
    ("loss", "LOSS"),
    ("ap", "AP"),
    ("r_acc", "R_ACC"),
    ("f_acc", "F_ACC"),
    ("acc", "ACC"),
]


def build_detailed_summary_row(
    run_name: str,
    training_mode: str,
    train_mods: Sequence[str],
    ood_mods: Sequence[str],
    id_metrics: dict,
    ood_metrics: Optional[dict],
) -> dict:

    row: Dict[str, object] = {
        "run_name": run_name,
        "training_mode": training_mode,
        "train_mods": "__".join(train_mods),
        "ood_mods": "__".join(ood_mods) if ood_mods else "",
    }

    for key, _label in SUMMARY_METRICS:
        row[f"test_id_{key}"] = id_metrics.get(f"test_{key}")

    if ood_metrics:
        for fake_mod, vals in ood_metrics.items():
            for key, _label in SUMMARY_METRICS:
                row[f"test_ood_{fake_mod}_{key}"] = vals.get(f"test_{key}")


    return row


def _build_detailed_sheet(detailed_rows: List[dict]) -> pd.DataFrame:


    metric_labels = [label for _, label in SUMMARY_METRICS]


    ordered_columns: List[Tuple[str, str]] = []
    data: Dict[Tuple[str, str], List[object]] = {}

    for row in detailed_rows:
        run_label = str(row.get("train_mods", ""))
        ood_mods = [m for m in str(row.get("ood_mods", "")).split("__") if m]


        block_columns: List[Tuple[str, str]] = [(run_label, "ID")]
        block_columns.extend((run_label, f"OOD {ood_mod}") for ood_mod in ood_mods)


        for col in block_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)


        data[(run_label, "ID")] = [row.get(f"test_id_{key}") for key, _label in SUMMARY_METRICS]


        for key, label in SUMMARY_METRICS:
            for ood_mod in ood_mods:
                col_key = (run_label, f"OOD {ood_mod}")
                if col_key not in data:
                    data[col_key] = [None] * len(metric_labels)

                data[col_key][metric_labels.index(label)] = row.get(f"test_ood_{ood_mod}_{key}")


    table = pd.DataFrame(data, index=metric_labels)


    if ordered_columns:
        table = table.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))
    table.index.name = "metric"
    return table


def _build_flat_summary_df(detailed_rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(detailed_rows)
    if df.empty:
        return pd.DataFrame(columns=["training_mode", "run_name", "train_mods", "ood_mods"])

    base_cols = ["training_mode", "run_name", "train_mods", "ood_mods"]
    metric_cols = [f"test_id_{key}" for key, _label in SUMMARY_METRICS]
    ood_cols = sorted([c for c in df.columns if c.startswith("test_ood_")])
    ordered_cols = [c for c in base_cols + metric_cols + ood_cols if c in df.columns]
    return df.reindex(columns=ordered_cols)


def save_summary_excel(root_dir: Path, detailed_rows: List[dict]) -> None:
    detailed_df = _build_detailed_sheet(detailed_rows)
    output_path = root_dir / "summary_metrics.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        detailed_df.to_excel(writer, sheet_name="detailed")


def save_summary_csv(root_dir: Path, detailed_rows: List[dict]) -> None:
    flat_df = _build_flat_summary_df(detailed_rows)
    output_path = root_dir / "summary_metrics.csv"
    flat_df.to_csv(output_path, index=False)

def build_training_runs(
    train_mods_all: List[str],
    real_mod: str,
    training_mode: str,
) -> List[Dict[str, object]]:
    fake_mods = [m for m in train_mods_all if m != real_mod]
    runs: List[Dict[str, object]] = []

    if training_mode == "allmods":
        runs.append(
            {
                "name": "allmods",
                "train_mods": list(train_mods_all),
                "ood_mods": [],
            }
        )
        return runs

    n_ood = 1 if training_mode == "leave1out" else 2
    if len(fake_mods) < n_ood:
        raise ValueError(f"training_mode={training_mode!r} richiede almeno {n_ood} mod fake diverse da real_mod.")

    for ood_mods in combinations(fake_mods, n_ood):


        ood_mods_list = list(ood_mods)
        train_mods_current = [m for m in train_mods_all if m not in ood_mods_list]
        runs.append(
            {
                "name": f"{training_mode}_{'__'.join(ood_mods_list)}",
                "train_mods": train_mods_current,
                "ood_mods": ood_mods_list,
            }
        )
    return runs

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a binary classifier on precomputed foundation-model embeddings.")

    parser.add_argument("--base_dir", type=str, required=True,
                        help="Root directory containing embedding samples and metadata under split/type/generator subdirectories.")
    parser.add_argument("--tys", type=str, default="inj,rem",
                        help="Comma-separated manipulation types; exactly one type is supported per run.")
    parser.add_argument("--train_set", type=str, default="train",
                        help="Dataset split used for training.")
    parser.add_argument("--val_set", type=str, default="valid",
                        help="Dataset split used for validation and optional OOD validation.")
    parser.add_argument("--test_set", type=str, default="valid",
                        help="Dataset split used for ID and OOD testing.")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="leave1out",
        choices=["leave1out", "allmods", "leave2out"],
        help="Generator protocol: hold out one, use all, or hold out two fake generators."
    )
    parser.add_argument("--train_mods", type=str, default="real,cycle,diffusion,pix2pix",
                        help="Comma-separated classes available for training, such as real,cycle,diffusion,pix2pix.")
    parser.add_argument("--real_mod", type=str, default="real",
                        help="Class name identifying authentic patches with label 0.")
    parser.add_argument('--batch_size', type=int, default=16, help="Number of embedding samples per batch.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader worker processes.")
    parser.add_argument('--gpu_ids', type=str, default='0', help="Comma-separated CUDA device IDs; use -1 to select the CPU.")


    parser.add_argument("--fc_arch", type=str, default="linear",
                        choices=["linear", "mlp3", "mlp5"],
                        help="Classifier head architecture: linear, three-layer MLP, or five-layer MLP.")
    parser.add_argument("--fc_hidden_dim", type=int, default=512,
                        help="Hidden layer width for MLP classifier heads.")
    parser.add_argument("--fc_dropout", type=float, default=0.0,
                        help="Dropout probability in MLP classifier heads.")
    parser.add_argument(
        "--best_key",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_ap", "val_acc", "val_r_acc", "val_f_acc"],
        help="Validation metric used to select the best model checkpoint.",
    )


    parser.add_argument("--seed", type=int, default=42, help="Random seed for training operations.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Initial learning rate for the selected optimizer.")
    parser.add_argument("--fc_min_lr", type=float, default=1e-6,
                        help="Minimum learning rate for the plateau scheduler.")
    parser.add_argument("--lr_scheduler", type=str, default="plateau", choices=["none", "plateau"],
                        help="Learning rate scheduler: none or ReduceLROnPlateau using validation loss.")
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help="Learning rate reduction factor for the plateau scheduler.")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="Validation epochs without improvement before the plateau scheduler reduces the learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay coefficient for the selected optimizer.")
    parser.add_argument("--early_stop", action="store_true", help="Enable early stopping based on validation loss.")
    parser.add_argument("--es_patience", type=int, default=10, help="Validation epochs without loss improvement before early stopping.")
    parser.add_argument("--es_min_delta", type=float, default=0.0, help="Minimum validation loss decrease counted as an improvement for early stopping.")
    parser.add_argument('--optim', type=str, default='adam', help="Optimizer to use: AdamW or SGD.")
    parser.add_argument('--beta1', type=float, default=0.9, help="First momentum coefficient for AdamW; ignored when using SGD.")
    parser.add_argument('--save_epoch_freq', type=int, default=100,help="Interval between periodic model checkpoints, measured in epochs.")
    parser.add_argument('--niter', type=int, default=1000, help="Maximum number of training epochs.")


    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help="Root directory for checkpoints, predictions, and run summaries.")
    parser.add_argument('--name', type=str, default='fc_generalization', help="Name of the experiment subdirectory under --checkpoints_dir.")
    parser.add_argument(
        '--loaded_samples_reference_dir',
        type=str,
        default='',
        help="Root directory containing <training_mode>/loaded_samples_<run_name>.csv reference manifests; defaults to <checkpoints_dir>/loaded_samples_id.",
    )


    partial_args, _ = parser.parse_known_args()
    tys = parse_mod_list(partial_args.tys) if partial_args.tys else []
    if not tys:
        raise ValueError("Specificare almeno un ty tramite --tys (es. 'inj' oppure 'inj,rem')")
    if len(tys) != 1:
        raise ValueError("Per ora il training supporta un solo ty alla volta. Usa --tys 'inj' oppure --tys 'rem'.")
    for ty in tys:
        add_ty_limit_args(parser, ty)


    args = parser.parse_args()


    train_mods_all = parse_mod_list(args.train_mods)


    run_configs = build_training_runs(train_mods_all, args.real_mod, args.training_mode)


    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_ids and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")


    base_dir = Path(args.base_dir)


    root_experiment_dir = Path(args.checkpoints_dir) / args.name
    root_experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(root_experiment_dir, args)
    loaded_samples_reference_root = (
        Path(args.loaded_samples_reference_dir)
        if args.loaded_samples_reference_dir
        else Path(args.checkpoints_dir) / "loaded_samples_id"
    )

    summary_rows: List[dict] = []

    for run_cfg in run_configs:
        current_train_mods = list(run_cfg["train_mods"])
        ood_mods = list(run_cfg["ood_mods"])
        run_name = str(run_cfg["name"])


        run_dir = root_experiment_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        train_max_real, train_max_fake = pick_max_samples_for_set_ty(args.train_set, ty, args, is_ood=False)
        val_max_real, val_max_fake = pick_max_samples_for_set_ty(args.val_set, ty, args, is_ood=False)
        test_max_real, test_max_fake = pick_max_samples_for_set_ty(args.test_set, ty, args, is_ood=False)

        train_loader = get_subset_embedding_dataloader(
            root=base_dir,
            split=args.train_set,
            ty=ty,
            mods=current_train_mods,
            real_mod=args.real_mod,
            max_samples_real=train_max_real,
            max_samples_fake=train_max_fake,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = get_subset_embedding_dataloader(
            root=base_dir,
            split=args.val_set,
            ty=ty,
            mods=current_train_mods,
            real_mod=args.real_mod,
            max_samples_real=val_max_real,
            max_samples_fake=val_max_fake,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )
        test_loader = get_subset_embedding_dataloader(
            root=base_dir,
            split=args.test_set,
            ty=ty,
            mods=current_train_mods,
            real_mod=args.real_mod,
            max_samples_real=test_max_real,
            max_samples_fake=test_max_fake,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )

        loaded_samples_loaders: List[Tuple[str, DataLoader]] = [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]

        ood_eval_loaders: Dict[str, Dict[str, DataLoader]] = {}
        if ood_mods:
            val_ood_max_real, val_ood_max_fake = pick_max_samples_for_set_ty(args.val_set, ty, args, is_ood=True)
            test_ood_max_real, test_ood_max_fake = pick_max_samples_for_set_ty(args.test_set, ty, args, is_ood=True)

            for ood_mod in ood_mods:
                ood_eval_mods = [args.real_mod, ood_mod]
                val_ood_loader = get_subset_embedding_dataloader(
                    root=base_dir,
                    split=args.val_set,
                    ty=ty,
                    mods=ood_eval_mods,
                    real_mod=args.real_mod,
                    max_samples_real=val_ood_max_real,
                    max_samples_fake=val_ood_max_fake,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                    pin_memory=(device.type == "cuda"),
                )
                test_ood_loader = get_subset_embedding_dataloader(
                    root=base_dir,
                    split=args.test_set,
                    ty=ty,
                    mods=ood_eval_mods,
                    real_mod=args.real_mod,
                    max_samples_real=test_ood_max_real,
                    max_samples_fake=test_ood_max_fake,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                    pin_memory=(device.type == "cuda"),
                )
                ood_eval_loaders[ood_mod] = {"val": val_ood_loader, "test": test_ood_loader}
                val_source_name = "val_ood" if len(ood_mods) == 1 else f"val_ood_{ood_mod}"
                test_source_name = "test_ood" if len(ood_mods) == 1 else f"test_ood_{ood_mod}"
                loaded_samples_loaders.extend([
                    (val_source_name, val_ood_loader),
                    (test_source_name, test_ood_loader),
                ])

        save_loaded_samples_csv(run_dir / "loaded_samples.csv", loaded_samples_loaders)


        if args.training_mode == "leave1out":
            reference_path = (loaded_samples_reference_root/ "leave1out"/ f"loaded_samples_{run_name}.csv")

            compare_loaded_samples_csv(run_dir / "loaded_samples.csv",reference_path,report_path=run_dir / "loaded_samples_compare_report.txt")
        elif args.training_mode == "leave2out":
            reference_path = (loaded_samples_reference_root/ "leave2out"/ f"loaded_samples_{run_name}.csv")

            compare_loaded_samples_csv(run_dir / "loaded_samples.csv",reference_path,report_path=run_dir / "loaded_samples_compare_report.txt")
        else:
            reference_path = (loaded_samples_reference_root/ "allmods"/ f"loaded_samples_{run_name}.csv")

            compare_loaded_samples_csv(run_dir / "loaded_samples.csv",reference_path,report_path=run_dir / "loaded_samples_compare_report.txt")


        model = train_fc_head(
            train_loader,
            val_loader,
            args,
            min_lr=args.fc_min_lr,
            device=device,
            output_dir=run_dir,
        )


        ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()


        val_ap, val_r_acc, val_f_acc, val_acc, val_loss = evaluate_fc(model, val_loader, device)
        (test_ap, test_r_acc, test_f_acc, test_acc, test_loss, test_rows) = evaluate_fc(model, test_loader, device, threshold=0.5, return_rows=True)

        save_predictions_csv(test_rows, run_dir / "test_predictions.csv")


        id_metrics_for_run_cfg = {
            "val_loss": float(val_loss),
            "val_ap": float(val_ap),
            "val_r_acc": float(val_r_acc),
            "val_f_acc": float(val_f_acc),
            "val_acc": float(val_acc),
            "test_loss": float(test_loss),
            "test_ap": float(test_ap),
            "test_r_acc": float(test_r_acc),
            "test_f_acc": float(test_f_acc),
            "test_acc": float(test_acc),
        }
        ood_metrics = None
        if ood_eval_loaders:
            ood_metrics = {}
            for ood_mod, loaders in ood_eval_loaders.items():
                val_ood_loader = loaders["val"]
                test_ood_loader = loaders["test"]

                val_h_ap, val_h_r_acc, val_h_f_acc, val_h_acc, val_h_loss = evaluate_fc(model, val_ood_loader, device)
                test_h_ap, test_h_r_acc, test_h_f_acc, test_h_acc, test_h_loss, test_ood_rows = evaluate_fc(
                    model, test_ood_loader, device, threshold=0.5, return_rows=True
                )
                pred_name = "test_ood_predictions.csv" if len(ood_eval_loaders) == 1 else f"test_ood_{ood_mod}_predictions.csv"
                save_predictions_csv(test_ood_rows, run_dir / pred_name)


                ood_metrics[ood_mod] = {
                    "mods": [args.real_mod, ood_mod],
                    "val_loss": float(val_h_loss),
                    "val_ap": float(val_h_ap),
                    "val_r_acc": float(val_h_r_acc),
                    "val_f_acc": float(val_h_f_acc),
                    "val_acc": float(val_h_acc),
                    "test_loss": float(test_h_loss),
                    "test_ap": float(test_h_ap),
                    "test_r_acc": float(test_h_r_acc),
                    "test_f_acc": float(test_h_f_acc),
                    "test_acc": float(test_h_acc),
                }
        summary_rows.append(

            build_detailed_summary_row(
                run_name=run_name,
                training_mode=args.training_mode,
                train_mods=current_train_mods,
                ood_mods=ood_mods,
                id_metrics=id_metrics_for_run_cfg,
                ood_metrics=ood_metrics,
            )
        )


    save_summary_excel(root_experiment_dir, summary_rows)
    save_summary_csv(root_experiment_dir, summary_rows)


if __name__ == "__main__":
    main()
