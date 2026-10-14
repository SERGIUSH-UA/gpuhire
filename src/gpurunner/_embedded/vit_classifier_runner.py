"""ViT binary image classifier — train + inference loop for Kaggle runtime.

Injected into the remote notebook by ``ViTClassifierJob.render_remote_code``.
Self-contained: stdlib + packages from ``ViTClassifierJob.requirements()``
plus what Kaggle ships (torch, torchvision, numpy).

Input layout (inside ``/kaggle/input/<dataset-slug>/``):
  <train_csv>          CSV: filename,label (label ∈ {0, 1})
  <inference_glob>     PNG/JPG/JPEG files (filename used in predictions.csv)

Output layout (inside ``/kaggle/working/``):
  predictions.csv         filename,score,label (score = P(positive))
  training_log.json       [{epoch, train_loss, train_acc, val_loss, val_acc, val_pr_auc}, ...]
  validation_report.json  {threshold_sweep: [...], confusion_at_default: {...}}
  model_checkpoint.pt     state_dict (only if save_checkpoint=true)
  meta.json               run metadata (timing, device, sample counts)
"""

from __future__ import annotations

import csv
import glob
import json
import os
import random
import time
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dump_input_tree(max_depth: int = 4) -> None:
    """Print /kaggle/input tree for debugging where the dataset got mounted."""
    if not KAGGLE_INPUT.exists():
        print(f"[diag] {KAGGLE_INPUT} does not exist", flush=True)
        return
    print(f"[diag] dumping {KAGGLE_INPUT} tree (max depth {max_depth}):", flush=True)
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        rel = Path(root).relative_to(KAGGLE_INPUT)
        depth = len(rel.parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{root}/  ({len(files)} files, {len(dirs)} dirs)", flush=True)
        if depth < max_depth:
            for fn in files[:5]:
                print(f"{indent}  {fn}", flush=True)
            if len(files) > 5:
                print(f"{indent}  ... +{len(files) - 5} more", flush=True)


def _dataset_root(slug: str) -> Path:
    """Locate the dataset root by searching for train_labels.csv anywhere in /kaggle/input.

    Kaggle's mount path depends on how the dataset was attached: it can be
    /kaggle/input/<slug-only>/ for dataset_sources entries, but sometimes
    nested under /kaggle/input/datasets/<slug>/ or similar.
    """
    if not KAGGLE_INPUT.exists():
        raise RuntimeError(f"{KAGGLE_INPUT} does not exist")

    # Strategy 1: exact match on common path patterns
    slug_short = slug.split("/")[-1]
    candidates = [
        KAGGLE_INPUT / slug_short,
        KAGGLE_INPUT / slug.replace("/", "-"),
        KAGGLE_INPUT / slug,
        KAGGLE_INPUT / "datasets" / slug_short,
        KAGGLE_INPUT / "datasets" / slug.split("/")[0] / slug_short,
    ]
    for c in candidates:
        if c.exists() and (c / "train_labels.csv").exists():
            print(f"[diag] dataset root: {c} (by candidate path)", flush=True)
            return c

    # Strategy 2: walk and find any dir containing train_labels.csv
    for root, _dirs, files in os.walk(KAGGLE_INPUT):
        if "train_labels.csv" in files:
            print(f"[diag] dataset root: {root} (by walk for train_labels.csv)", flush=True)
            return Path(root)

    raise RuntimeError(
        f"Could not locate dataset for slug={slug!r}. "
        f"No train_labels.csv found under {KAGGLE_INPUT}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


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
        print(f"[warn] {missing} train rows have missing images and were skipped", flush=True)
    if not samples:
        raise RuntimeError("train set is empty after filtering missing files")

    rng = random.Random(seed)
    rng.shuffle(samples)

    # Stratified split
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

    # ImageNet normalization — ViT pretrained models expect these.
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    def __init__(self, items: list[tuple[Path, int]], tf):
        self.items = items
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image

        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


class _UnlabeledImageDataset:
    def __init__(self, paths: list[Path], tf):
        self.paths = paths
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), str(path)


def _make_model(model_name: str, num_labels: int = 2):
    from transformers import ViTForImageClassification

    return ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )


def _weighted_sampler(items: list[tuple[Path, int]], seed: int):
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = [lbl for _, lbl in items]
    cnt = [labels.count(0), labels.count(1)]
    weights = [1.0 / max(cnt[lbl], 1) for lbl in labels]
    g = torch.Generator(); g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(items), replacement=True, generator=g
    )


def _evaluate(model, loader, device) -> tuple[float, float, list[float], list[int]]:
    import torch

    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    labels: list[int] = []
    correct = 0; total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_t = torch.as_tensor(y).to(device, non_blocking=True)
            out = model(pixel_values=x)
            logits = out.logits
            loss = criterion(logits, y_t)
            losses.append(loss.item())
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            correct += (preds == y_t).sum().item()
            total += y_t.numel()
            scores.extend(probs.cpu().tolist())
            labels.extend(y_t.cpu().tolist())
    avg_loss = sum(losses) / max(len(losses), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, scores, labels


def _threshold_sweep(scores: list[float], labels: list[int]) -> list[dict[str, float]]:
    from sklearn.metrics import precision_recall_fscore_support

    out = []
    for thr in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        preds = [1 if s >= thr else 0 for s in scores]
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0,
        )
        out.append({
            "threshold": thr,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "above_threshold": sum(preds),
        })
    return out


def main(params: dict[str, Any]) -> None:
    import torch
    from torch.utils.data import DataLoader

    started = time.time()
    print(f"[{_utc_iso()}] vit_classifier start: {params!r}", flush=True)

    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)

    seed = int(params["seed"])
    _seed_everything(seed)

    _dump_input_tree()
    root = _dataset_root(params["dataset"])
    print(f"dataset root: {root}", flush=True)

    train_items, val_items = _load_train_split(
        root, params["train_csv"], float(params["val_split"]), seed
    )
    inference_paths = _list_inference(root, params["inference_glob"])
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
    train_loader = DataLoader(
        train_ds, batch_size=int(params["batch_size"]),
        sampler=sampler, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(params["batch_size"]), shuffle=False,
        num_workers=2, pin_memory=True,
    )
    infer_loader = DataLoader(
        infer_ds, batch_size=int(params["batch_size"]) * 2, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}", flush=True)

    model = _make_model(params["model"], num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]))
    criterion = torch.nn.CrossEntropyLoss()

    epochs = int(params["epochs"])
    log: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_scores: list[float] = []
    best_labels: list[int] = []
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0; ep_n = 0
        ep_correct = 0
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
            preds = logits.argmax(dim=-1)
            ep_correct += (preds == y_t).sum().item()
        train_loss = ep_loss / max(ep_n, 1)
        train_acc = ep_correct / max(ep_n, 1)
        val_loss, val_acc, val_scores, val_labels = _evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        log.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
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

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Inference
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
                rel = Path(p).name
                preds_rows.append((rel, float(s), 1 if s >= decision_threshold else 0))

    # Sort by score desc
    preds_rows.sort(key=lambda r: -r[1])
    with (KAGGLE_WORKING / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "score", "label"])
        w.writerows([[r[0], f"{r[1]:.6f}", r[2]] for r in preds_rows])

    # Threshold sweep on val
    sweep = _threshold_sweep(best_scores, best_labels)
    val_report = {
        "best_val_acc": best_val_acc,
        "n_val": len(best_labels),
        "n_val_pos": sum(best_labels),
        "threshold_sweep": sweep,
    }
    (KAGGLE_WORKING / "validation_report.json").write_text(
        json.dumps(val_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (KAGGLE_WORKING / "training_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if bool(params.get("save_checkpoint", False)) and best_state is not None:
        torch.save(best_state, KAGGLE_WORKING / "model_checkpoint.pt")

    meta = {
        "started_at": _utc_iso(),
        "elapsed_s": round(time.time() - started, 1),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "model": params["model"],
        "params": params,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_inference": len(inference_paths),
        "best_val_acc": best_val_acc,
    }
    (KAGGLE_WORKING / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{_utc_iso()}] done. predictions: {len(preds_rows)} rows. "
          f"best val_acc: {best_val_acc:.3f}", flush=True)


if __name__ == "__main__":
    raise RuntimeError("This module is meant to be injected into a Kaggle notebook, not run locally.")
