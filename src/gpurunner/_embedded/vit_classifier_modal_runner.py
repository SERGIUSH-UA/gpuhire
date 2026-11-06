"""ViT binary image classifier — Modal-runtime variant.

Injected into the Modal container by ``ViTClassifierJob.render_runner_module``.

Differences from the Kaggle runner (``vit_classifier_runner.py``):
  * Reads ``output_root`` from ``params`` (passed by the Modal wrapper) instead
    of writing to ``/kaggle/working``.
  * Pulls the dataset from Kaggle via the ``kaggle`` Python API, using
    credentials supplied via a Modal Secret (``KAGGLE_KEY``,
    optionally with ``KAGGLE_USERNAME``).
  * ``main(params) -> dict`` returns a summary that the Modal wrapper sends
    back as the function's return value.

Auth modes supported (in order):
  1. ``KAGGLE_KEY`` starting with ``KGAT_``  → written to ``~/.kaggle/access_token``
  2. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``     → written to ``~/.kaggle/kaggle.json``
"""

from __future__ import annotations

import csv
import glob
import json
import os
import random
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _setup_kaggle_auth() -> None:
    """Materialise credentials from env vars into ``~/.kaggle/{access_token,kaggle.json}``."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    key = os.environ.get("KAGGLE_KEY", "").strip()
    user = os.environ.get("KAGGLE_USERNAME", "").strip()

    if key.startswith("KGAT_"):
        path = kaggle_dir / "access_token"
        path.write_text(key, encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        print("[auth] wrote ~/.kaggle/access_token (KGAT)", flush=True)
        return

    if user and key:
        path = kaggle_dir / "kaggle.json"
        path.write_text(
            json.dumps({"username": user, "key": key}), encoding="utf-8"
        )
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        print(f"[auth] wrote ~/.kaggle/kaggle.json for user {user!r}", flush=True)
        return

    raise RuntimeError(
        "no Kaggle credentials in env: need KAGGLE_KEY (KGAT) "
        "or KAGGLE_USERNAME + KAGGLE_KEY"
    )


def _download_dataset(slug: str, dest: Path) -> Path:
    """Download a Kaggle Dataset and unpack any nested zips. Returns root path."""
    dest.mkdir(parents=True, exist_ok=True)
    from kaggle import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[dl] kaggle.dataset_download_files({slug!r}, unzip=True)", flush=True)
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)

    # If Kaggle stored sub-zips (because we used --dir-mode zip on upload),
    # unzip them in-place and remove the archives.
    for z in list(dest.glob("*.zip")):
        print(f"[dl] unzipping nested {z.name}", flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()
    return dest


def _load_train_split(
    root: Path,
    train_csv: str,
    val_split: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    import pandas as pd

    csv_path = root / train_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"train_csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError(f"train_csv must have 'filename' + 'label' cols, got: {list(df.columns)}")

    samples: list[tuple[Path, int]] = []
    missing = 0
    for _, row in df.iterrows():
        fp = root / str(row["filename"])
        if not fp.exists():
            missing += 1
            continue
        samples.append((fp, int(row["label"])))
    if missing:
        print(f"[warn] {missing} train rows missing images, skipped", flush=True)
    if not samples:
        raise RuntimeError("train set empty after filtering missing files")

    rng = random.Random(seed)
    rng.shuffle(samples)

    pos = [s for s in samples if s[1] == 1]
    neg = [s for s in samples if s[1] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_val_pos = max(1, int(len(pos) * val_split))
    n_val_neg = max(1, int(len(neg) * val_split))
    val = pos[:n_val_pos] + neg[:n_val_neg]
    train = pos[n_val_pos:] + neg[n_val_neg:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val


def _list_inference(root: Path, pattern: str) -> list[Path]:
    matches = sorted(Path(p) for p in glob.glob(str(root / pattern)))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise RuntimeError(f"inference_glob matched 0 files: {root / pattern}")
    return matches


def _build_transforms(image_size: int):
    from torchvision import transforms

    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # Stronger augmentation for cross-archetype robustness (1796 Роспись vs
    # 1848 Ведомость): rotation, perspective, blur, broader color jitter.
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomRotation(degrees=5, fill=255),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=255),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


class _BinaryImageDataset:
    def __init__(self, items, tf):
        self.items = items; self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.items[idx]
        return self.tf(Image.open(path).convert("RGB")), label


class _UnlabeledImageDataset:
    def __init__(self, paths, tf):
        self.paths = paths; self.tf = tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        from PIL import Image
        path = self.paths[idx]
        return self.tf(Image.open(path).convert("RGB")), str(path)


def _weighted_sampler(items, seed):
    import torch
    from torch.utils.data import WeightedRandomSampler
    labels = [lbl for _, lbl in items]
    cnt = [labels.count(0), labels.count(1)]
    weights = [1.0 / max(cnt[lbl], 1) for lbl in labels]
    g = torch.Generator(); g.manual_seed(seed)
    return WeightedRandomSampler(weights, len(items), replacement=True, generator=g)


def _evaluate(model, loader, device):
    import torch
    model.eval()
    losses, scores, labels = [], [], []
    correct = total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_t = torch.as_tensor(y).to(device, non_blocking=True)
            logits = model(pixel_values=x).logits
            losses.append(criterion(logits, y_t).item())
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            correct += (preds == y_t).sum().item()
            total += y_t.numel()
            scores.extend(probs.cpu().tolist())
            labels.extend(y_t.cpu().tolist())
    return (
        sum(losses) / max(len(losses), 1),
        correct / max(total, 1),
        scores,
        labels,
    )


def _threshold_sweep(scores, labels):
    from sklearn.metrics import precision_recall_fscore_support
    out = []
    for thr in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        preds = [1 if s >= thr else 0 for s in scores]
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        out.append({
            "threshold": thr, "precision": float(p), "recall": float(r),
            "f1": float(f1), "above_threshold": sum(preds),
        })
    return out


def main(params: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import ViTForImageClassification

    started = time.time()
    print(f"[{_utc_iso()}] vit_classifier (modal) start: {params!r}", flush=True)

    output_root = Path(params.get("output_root", "/tmp/gpurunner_output"))
    output_root.mkdir(parents=True, exist_ok=True)

    seed = int(params["seed"])
    _seed_everything(seed)

    # Fetch dataset from Kaggle
    _setup_kaggle_auth()
    dataset_root = _download_dataset(params["dataset"], Path("/tmp/dataset"))
    print(f"[dl] dataset root: {dataset_root}", flush=True)
    for top in sorted(dataset_root.iterdir())[:20]:
        print(f"[dl]   {top.name}{'/' if top.is_dir() else ''}", flush=True)

    train_items, val_items = _load_train_split(
        dataset_root, params["train_csv"], float(params["val_split"]), seed
    )
    inference_paths = _list_inference(dataset_root, params["inference_glob"])
    print(
        f"samples: train={len(train_items)} (pos={sum(1 for _, l in train_items if l)}) "
        f"val={len(val_items)} (pos={sum(1 for _, l in val_items if l)}) "
        f"inference={len(inference_paths)}",
        flush=True,
    )

    train_tf, eval_tf = _build_transforms(int(params["image_size"]))
    train_ds = _BinaryImageDataset(train_items, train_tf)
    val_ds = _BinaryImageDataset(val_items, eval_tf)
    infer_ds = _UnlabeledImageDataset(inference_paths, eval_tf)

    sampler = _weighted_sampler(train_items, seed)
    bs = int(params["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=2, pin_memory=True)
    infer_loader = DataLoader(infer_ds, batch_size=bs * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}", flush=True)

    model = ViTForImageClassification.from_pretrained(
        params["model"], num_labels=2, ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]))
    criterion = torch.nn.CrossEntropyLoss()

    epochs = int(params["epochs"])
    log = []
    best_val_acc = -1.0
    best_scores: list[float] = []
    best_labels: list[int] = []
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0; ep_n = 0; ep_correct = 0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y_t = torch.as_tensor(y).to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(pixel_values=x).logits
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * y_t.size(0)
            ep_n += y_t.size(0)
            ep_correct += (logits.argmax(dim=-1) == y_t).sum().item()
        train_loss = ep_loss / max(ep_n, 1)
        train_acc = ep_correct / max(ep_n, 1)
        val_loss, val_acc, val_scores, val_labels = _evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        log.append({
            "epoch": ep, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
            "elapsed_s": round(elapsed, 1),
        })
        print(
            f"epoch {ep}/{epochs}  train_loss={train_loss:.4f} acc={train_acc:.3f} "
            f"| val_loss={val_loss:.4f} acc={val_acc:.3f}  ({elapsed:.0f}s)",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_scores = val_scores
            best_labels = val_labels
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    print("running inference...", flush=True)
    preds_rows: list[tuple[str, float, int]] = []
    decision_threshold = float(params["decision_threshold"])
    model.eval()
    with torch.no_grad():
        for x, paths in infer_loader:
            x = x.to(device, non_blocking=True)
            logits = model(pixel_values=x).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            for p, s in zip(paths, probs.cpu().tolist()):
                # Keep last 2 path components when inference_glob spans subdirs
                # (e.g. inference/*/*.png) to avoid basename collisions across slugs.
                parts = Path(p).parts
                rel = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                preds_rows.append((rel, float(s), 1 if s >= decision_threshold else 0))
    preds_rows.sort(key=lambda r: -r[1])

    with (output_root / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "score", "label"])
        w.writerows([[r[0], f"{r[1]:.6f}", r[2]] for r in preds_rows])

    sweep = _threshold_sweep(best_scores, best_labels)
    val_report = {
        "best_val_acc": best_val_acc,
        "n_val": len(best_labels),
        "n_val_pos": sum(best_labels),
        "threshold_sweep": sweep,
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(val_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "training_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed_s = round(time.time() - started, 1)
    summary = {
        "started_at": _utc_iso(),
        "elapsed_s": elapsed_s,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "model": params["model"],
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_inference": len(inference_paths),
        "n_predictions": len(preds_rows),
        "best_val_acc": best_val_acc,
        "predictions_above_threshold": sum(1 for _, _, lbl in preds_rows if lbl == 1),
    }
    (output_root / "meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{_utc_iso()}] done in {elapsed_s}s. {len(preds_rows)} predictions, "
        f"best val_acc: {best_val_acc:.3f}",
        flush=True,
    )
    return summary
