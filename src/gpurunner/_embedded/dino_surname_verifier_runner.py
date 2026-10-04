#!/usr/bin/env python3
"""Train/evaluate DINO surname verifier v1.

Designed for both local runs and Kaggle/gpurunner notebooks. Input dataset:

  train_labels.csv      filename,label
  holdout_labels.csv    filename,label,detail,source,page,score_bucket
  images/*.png          flat Kaggle layout
  golden_verifier_v2/   optional coverage gate with labels.tsv + images/

Output:

  model_checkpoint/     Hugging Face model + processor
  model_checkpoints_top3/    # role checkpoints, kept under the old name for compatibility
  model_checkpoints_by_epoch/ # every epoch checkpoint for manual selection
  training_log.json
  validation_report.json
  holdout_report.json
  predictions.csv       score for every image under inference_glob
  meta.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="", help="dataset root; auto-detects Kaggle root if omitted")
    ap.add_argument("--train-csv", default="train_labels.csv")
    ap.add_argument("--holdout-csv", default="holdout_labels.csv")
    ap.add_argument("--inference-glob", default="images/*.png")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument(
        "--augment-profile",
        choices=("default", "paper_color"),
        default="default",
        help="Training augmentation profile. paper_color breaks paper/background color shortcuts.",
    )
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--checkpoint-metric", choices=("val_acc", "recall_safe"), default="val_acc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument(
        "--golden-dir",
        default="golden_verifier_v2",
        help="Optional golden verifier set. Relative paths are resolved against dataset root, cwd, and repo root.",
    )
    ap.add_argument("--save-checkpoint", action="store_true", default=True)
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_dataset_root(explicit: str, train_csv: str) -> Path:
    if explicit:
        root = Path(explicit)
        if not (root / train_csv).exists():
            raise FileNotFoundError(root / train_csv)
        return root
    candidates = [Path.cwd(), Path("/kaggle/input")]
    for base in candidates:
        if not base.exists():
            continue
        if (base / train_csv).exists():
            return base
        for root, _dirs, files in os.walk(base):
            if train_csv in files:
                return Path(root)
    raise RuntimeError(f"Could not auto-detect dataset root containing {train_csv}")


def read_labeled(root: Path, csv_name: str, require_detail: bool = False) -> list[dict]:
    path = root / csv_name
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fp = root / row["filename"]
            if not fp.exists():
                continue
            item = {
                "path": fp,
                "filename": row["filename"],
                "label": int(row["label"]),
                "detail": row.get("detail", ""),
                "source": row.get("source", ""),
                "page": row.get("page", ""),
                "score_bucket": row.get("score_bucket", ""),
            }
            if require_detail and not item["detail"]:
                item["detail"] = "unknown"
            rows.append(item)
    return rows


def resolve_optional_dir(root: Path, value: str) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    candidates = [
        raw,
        root / raw,
        Path.cwd() / raw,
    ]
    file_name = globals().get("__file__")
    if file_name:
        candidates.append(Path(file_name).resolve().parent.parent / raw)
    for path in candidates:
        if path.exists():
            return path
    return None


def read_golden(golden_dir: Path | None) -> list[dict]:
    if golden_dir is None:
        return []
    labels_path = golden_dir / "labels.tsv"
    images_dir = golden_dir / "images"
    if not labels_path.exists() or not images_dir.exists():
        return []
    rows: list[dict] = []
    with labels_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            fp = images_dir / row["filename"]
            if not fp.exists():
                continue
            rows.append({
                "path": fp,
                "filename": row["filename"],
                "label": int(row["label"]),
                "detail": row.get("detail", ""),
                "source": row.get("source", ""),
                "source_group": row.get("source_group", ""),
                "source_set": row.get("source_set", ""),
                "era_bucket": row.get("era_bucket", ""),
                "page": row.get("page", ""),
                "score_bucket": row.get("score_bucket", ""),
            })
    return rows


def stratified_split(items: list[dict], val_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    pos = [r for r in items if r["label"] == 1]
    neg = [r for r in items if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val_pos = max(1, round(len(pos) * val_split))
    n_val_neg = max(1, round(len(neg) * val_split))
    val = pos[:n_val_pos] + neg[:n_val_neg]
    train = pos[n_val_pos:] + neg[n_val_neg:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_transforms(image_size: int, augment_profile: str = "default"):
    from torchvision import transforms

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment_profile == "paper_color":
        train_tf = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomRotation(degrees=4, fill=255),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.25,
                contrast=0.22,
                saturation=0.12,
                hue=0.02,
            )], p=0.85),
            transforms.RandomAutocontrast(p=0.15),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.20),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomRotation(degrees=3, fill=255),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.10),
            transforms.ToTensor(),
            norm,
        ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


class ImageDataset:
    def __init__(self, rows: list[dict], tf, labeled: bool = True):
        self.rows = rows
        self.tf = tf
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        from PIL import Image

        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        x = self.tf(img)
        if self.labeled:
            return x, row["label"], idx
        return x, idx


def weighted_sampler(rows: list[dict], seed: int):
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = [r["label"] for r in rows]
    counts = {0: labels.count(0), 1: labels.count(1)}
    weights = [1.0 / max(counts[label], 1) for label in labels]
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(weights, len(rows), replacement=True, generator=g)


def make_model(model_name: str):
    from transformers import AutoModelForImageClassification

    return AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "fp", 1: "tp"},
        label2id={"fp": 0, "tp": 1},
        ignore_mismatched_sizes=True,
    )


def clone_model_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def evaluate(model, loader, device) -> tuple[float, float, list[float], list[int], list[int]]:
    import torch

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    labels: list[int] = []
    indices: list[int] = []
    correct = total = 0
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(pixel_values=x)
            loss = criterion(out.logits, y)
            probs = torch.softmax(out.logits, dim=-1)[:, 1]
            pred = (probs >= 0.5).long()
            correct += (pred == y).sum().item()
            total += y.numel()
            losses.append(loss.item())
            scores.extend(probs.cpu().tolist())
            labels.extend(y.cpu().tolist())
            indices.extend(idx.cpu().tolist())
    return sum(losses) / max(len(losses), 1), correct / max(total, 1), scores, labels, indices


def threshold_sweep(scores: list[float], labels: list[int]) -> list[dict]:
    rows = []
    for thr in (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        preds = [1 if s >= thr else 0 for s in scores]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({
            "threshold": thr,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "above": sum(preds),
        })
    return rows


def best_recall_row(scores: list[float], labels: list[int]) -> dict:
    sweep = threshold_sweep(scores, labels)
    safe = [row for row in sweep if row["recall"] == 1.0]
    if safe:
        row = min(safe, key=lambda item: (item["fp"], -item["threshold"]))
        return {
            "safe": True,
            "threshold": row["threshold"],
            "recall": row["recall"],
            "fp": row["fp"],
            "fn": row["fn"],
            "tp": row["tp"],
            "precision": row["precision"],
        }
    row = max(sweep, key=lambda item: (item["recall"], -item["fp"], item["threshold"]))
    return {
        "safe": False,
        "threshold": row["threshold"],
        "recall": row["recall"],
        "fp": row["fp"],
        "fn": row["fn"],
        "tp": row["tp"],
        "precision": row["precision"],
    }


def positive_score_stats(scores: list[float], labels: list[int]) -> dict:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    if not positives:
        return {"min_pos_score": None, "median_pos_score": None}
    sorted_pos = sorted(positives)
    return {
        "min_pos_score": min(sorted_pos),
        "median_pos_score": sorted_pos[len(sorted_pos) // 2],
    }


def recall_safe_checkpoint_key(scores: list[float], labels: list[int], val_acc: float) -> tuple:
    """Rank checkpoints for verifier usage.

    Primary goal: keep all validation positives above some threshold. Among
    recall-safe thresholds, prefer fewer FP. This aligns checkpoint selection
    with our downstream queue-reduction objective better than raw val accuracy.
    """
    sweep = threshold_sweep(scores, labels)
    safe = [row for row in sweep if row["recall"] == 1.0]
    if safe:
        best_safe = min(safe, key=lambda row: (row["fp"], -row["threshold"]))
        return (1, -best_safe["fp"], best_safe["threshold"], val_acc)
    best = max(sweep, key=lambda row: (row["recall"], -row["fp"], row["threshold"]))
    return (0, best["recall"], -best["fp"], val_acc)


def holdout_positive_stats(scores: list[float], labels: list[int]) -> dict:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    if not positives:
        return {
            "min_pos_score": None,
            "median_pos_score": None,
            "safe_at_0_003": False,
            "safe_at_0_01": False,
            "recall_at_0_003": 0.0,
            "recall_at_0_01": 0.0,
        }
    min_pos = min(positives)
    sorted_pos = sorted(positives)
    return {
        "min_pos_score": min_pos,
        "median_pos_score": sorted_pos[len(sorted_pos) // 2],
        "safe_at_0_003": min_pos >= 0.003,
        "safe_at_0_01": min_pos >= 0.01,
        "recall_at_0_003": sum(1 for score in positives if score >= 0.003) / len(positives),
        "recall_at_0_01": sum(1 for score in positives if score >= 0.01) / len(positives),
    }


def best_val_safe(scores: list[float], labels: list[int]) -> dict:
    safe = [row for row in threshold_sweep(scores, labels) if row["recall"] == 1.0]
    if safe:
        row = min(safe, key=lambda item: (item["fp"], -item["threshold"]))
        return {
            "val_safe": 1,
            "val_fp": row["fp"],
            "val_threshold": row["threshold"],
            "val_recall": 1.0,
        }
    row = max(threshold_sweep(scores, labels), key=lambda item: (item["recall"], -item["fp"], item["threshold"]))
    return {
        "val_safe": 0,
        "val_fp": row["fp"],
        "val_threshold": row["threshold"],
        "val_recall": row["recall"],
    }


def role_checkpoint_keys(candidate: dict) -> dict[str, tuple]:
    """Rank checkpoints by explicit downstream roles.

    Queue construction benefits from more than one judge:
    - recall_guard keeps hard positives alive;
    - fp_suppressor is allowed to be stricter, but must stay validation-safe;
    - balanced is the default production checkpoint.
    """
    holdout_stats = holdout_positive_stats(candidate["holdout_scores"], candidate["holdout_labels"])
    holdout_min = holdout_stats["min_pos_score"] or 0.0
    holdout_median = holdout_stats["median_pos_score"] or 0.0
    h003 = holdout_stats["recall_at_0_003"]
    h001 = holdout_stats["recall_at_0_01"]
    safe003 = 1 if holdout_stats["safe_at_0_003"] else 0
    safe001 = 1 if holdout_stats["safe_at_0_01"] else 0
    val = best_val_safe(candidate["scores"], candidate["labels"])
    golden = candidate.get("golden") or {}
    golden_best = golden.get("best") or {"safe": False, "recall": 0.0, "fp": 10**9, "threshold": 0.0}
    golden_stats = golden.get("positive_stats") or {"min_pos_score": 0.0, "median_pos_score": 0.0}
    golden_safe = 1 if golden_best.get("safe") else 0
    golden_recall = float(golden_best.get("recall") or 0.0)
    golden_fp = int(golden_best.get("fp") or 0)
    golden_min = float(golden_stats.get("min_pos_score") or 0.0)
    golden_median = float(golden_stats.get("median_pos_score") or 0.0)
    return {
        "best_recall_guard": (
            golden_recall,
            golden_safe,
            golden_min,
            h001,
            h003,
            safe001,
            safe003,
            holdout_min,
            holdout_median,
            val["val_safe"],
            -val["val_fp"],
            candidate["val_acc"],
        ),
        "best_fp_suppressor": (
            golden_safe,
            golden_recall,
            -golden_fp,
            val["val_safe"],
            -val["val_fp"],
            val["val_threshold"],
            candidate["val_acc"],
            h003,
            h001,
            holdout_min,
        ),
        "best_balanced": (
            golden_recall,
            golden_safe,
            -golden_fp,
            golden_median,
            h001,
            val["val_safe"],
            -val["val_fp"],
            h003,
            candidate["val_acc"],
            holdout_min,
            -candidate["val_loss"],
        ),
        "best_val_loss": (
            golden_recall,
            h001,
            val["val_safe"],
            -candidate["val_loss"],
            candidate["val_acc"],
            holdout_min,
        ),
    }


def diverse_checkpoint_keys(candidate: dict) -> dict[str, tuple]:
    keys = role_checkpoint_keys(candidate)
    # Backward-compatible alias for older tooling; the default checkpoint should
    # be the balanced role, not raw validation accuracy.
    keys["best_recall_safe"] = keys["best_balanced"]
    return keys


def checkpoint_key_for_metric(metric: str, candidate: dict) -> tuple:
    keys = role_checkpoint_keys(candidate)
    if metric == "recall_safe":
        return keys["best_balanced"]
    if metric == "val_acc":
        holdout_stats = holdout_positive_stats(candidate["holdout_scores"], candidate["holdout_labels"])
        golden = candidate.get("golden") or {}
        golden_best = golden.get("best") or {"safe": False, "recall": 0.0, "fp": 10**9}
        golden_stats = golden.get("positive_stats") or {"min_pos_score": 0.0}
        return (
            float(golden_best.get("recall") or 0.0),
            1 if golden_best.get("safe") else 0,
            -int(golden_best.get("fp") or 0),
            float(golden_stats.get("min_pos_score") or 0.0),
            1 if holdout_stats["safe_at_0_01"] else 0,
            candidate["val_acc"],
            holdout_stats["min_pos_score"] or 0.0,
            -candidate["val_loss"],
        )
    raise ValueError(f"unknown checkpoint metric: {metric}")


def holdout_report(rows: list[dict], scores: list[float], labels: list[int], indices: list[int]) -> dict:
    scored = []
    for score, label, idx in zip(scores, labels, indices):
        row = dict(rows[idx])
        row["score"] = score
        row["label"] = label
        scored.append(row)
    by_detail = {}
    for detail in sorted({r["detail"] for r in scored}):
        part = [r for r in scored if r["detail"] == detail]
        positives = [r for r in part if r["label"] == 1]
        by_detail[detail] = {
            "count": len(part),
            "pos": len(positives),
            "min_pos_score": min((r["score"] for r in positives), default=None),
        }
    return {
        "count": len(scored),
        "positives": sum(1 for r in scored if r["label"] == 1),
        "min_positive_score": min((r["score"] for r in scored if r["label"] == 1), default=None),
        "by_detail": by_detail,
        "threshold_sweep": threshold_sweep(scores, labels),
        "rows": [
            {
                "filename": r["filename"],
                "label": r["label"],
                "detail": r["detail"],
                "source": r["source"],
                "page": r["page"],
                "score_bucket": r["score_bucket"],
                "score": round(float(r["score"]), 8),
            }
            for r in sorted(scored, key=lambda x: x["score"])
        ],
    }


def holdout_epoch_summary(scores: list[float], labels: list[int]) -> dict:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    sweep = threshold_sweep(scores, labels)
    recall_by_threshold = {
        str(row["threshold"]): row["recall"]
        for row in sweep
        if row["threshold"] in {0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5}
    }
    return {
        "holdout_pos": len(positives),
        "holdout_min_pos_score": round(min(positives), 8) if positives else None,
        "holdout_median_pos_score": round(sorted(positives)[len(positives) // 2], 8) if positives else None,
        "holdout_recall": recall_by_threshold,
    }


def golden_epoch_summary(scores: list[float], labels: list[int]) -> dict:
    if not scores:
        return {}
    best = best_recall_row(scores, labels)
    stats = positive_score_stats(scores, labels)
    return {
        "golden_pos": sum(1 for label in labels if label == 1),
        "golden_neg": sum(1 for label in labels if label == 0),
        "golden_best_recall": best["recall"],
        "golden_best_threshold": best["threshold"],
        "golden_best_fp": best["fp"],
        "golden_best_fn": best["fn"],
        "golden_safe": best["safe"],
        "golden_min_pos_score": round(stats["min_pos_score"], 8) if stats["min_pos_score"] is not None else None,
        "golden_median_pos_score": round(stats["median_pos_score"], 8) if stats["median_pos_score"] is not None else None,
    }


def scored_report(rows: list[dict], scores: list[float], labels: list[int], indices: list[int]) -> dict:
    scored = []
    for score, label, idx in zip(scores, labels, indices):
        row = dict(rows[idx])
        row["score"] = score
        row["label"] = label
        scored.append(row)
    return {
        "count": len(scored),
        "positives": sum(1 for row in scored if row["label"] == 1),
        "negatives": sum(1 for row in scored if row["label"] == 0),
        "threshold_sweep": threshold_sweep(scores, labels),
        "best_recall": best_recall_row(scores, labels) if scored else None,
        "positive_score_stats": positive_score_stats(scores, labels),
        "rows": [
            {
                "filename": row["filename"],
                "label": row["label"],
                "detail": row.get("detail", ""),
                "source": row.get("source", ""),
                "source_group": row.get("source_group", ""),
                "source_set": row.get("source_set", ""),
                "era_bucket": row.get("era_bucket", ""),
                "page": row.get("page", ""),
                "score": round(float(row["score"]), 8),
            }
            for row in sorted(scored, key=lambda item: item["score"])
        ],
    }


def val_recall_safe_summary(scores: list[float], labels: list[int]) -> dict:
    safe = [row for row in threshold_sweep(scores, labels) if row["recall"] == 1.0]
    if not safe:
        best = max(threshold_sweep(scores, labels), key=lambda row: (row["recall"], -row["fp"], row["threshold"]))
        return {
            "val_recall_safe": False,
            "best_recall": best["recall"],
            "best_recall_threshold": best["threshold"],
            "best_recall_fp": best["fp"],
        }
    best_safe = min(safe, key=lambda row: (row["fp"], -row["threshold"]))
    return {
        "val_recall_safe": True,
        "val_recall_safe_threshold": best_safe["threshold"],
        "val_recall_safe_fp": best_safe["fp"],
        "val_recall_safe_above": best_safe["above"],
    }


def golden_recall_by_detail(scored: list[dict], thresholds: list[float]) -> dict:
    by_detail: dict[str, list[dict]] = {}
    for row in scored:
        if row["label"] == 1:
            by_detail.setdefault(row.get("detail") or "unknown", []).append(row)
    out = {}
    for detail, rows in sorted(by_detail.items()):
        out[detail] = [
            {
                "threshold": threshold,
                "hits": sum(1 for row in rows if row["score"] >= threshold),
                "total": len(rows),
                "recall": sum(1 for row in rows if row["score"] >= threshold) / max(len(rows), 1),
            }
            for threshold in thresholds
        ]
    return out


def write_golden_eval(model, golden_dir: Path, tf, device, out_dir: Path, model_name: str, image_size: int) -> None:
    """Evaluate current best model on frozen golden set if it is available.

    Kaggle training datasets usually do not include the local golden set. In
    that case this is a no-op; local/fetched model evaluation can still run via
    scripts/evaluate_dino_golden.py.
    """
    labels_path = golden_dir / "labels.tsv"
    images_dir = golden_dir / "images"
    if not labels_path.exists() or not images_dir.exists():
        return

    import torch
    from PIL import Image

    rows: list[dict] = []
    with labels_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            img_path = images_dir / row["filename"]
            if not img_path.exists():
                continue
            rows.append({
                "filename": row["filename"],
                "label": int(row["label"]),
                "detail": row.get("detail", ""),
                "source": row.get("source", ""),
                "page": row.get("page", ""),
                "path": img_path,
            })
    if not rows:
        return

    model.eval()
    scored = []
    with torch.no_grad():
        for row in rows:
            x = tf(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device)
            score = torch.softmax(model(pixel_values=x).logits, dim=-1)[0, 1].item()
            item = dict(row)
            item.pop("path", None)
            item["score"] = float(score)
            scored.append(item)

    thresholds = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    sweep = threshold_sweep([row["score"] for row in scored], [row["label"] for row in scored])
    safe = [row for row in sweep if row["recall"] == 1.0]
    recall_safe = min(safe, key=lambda row: (row["fp"], -row["threshold"])) if safe else None
    report = {
        "created": utc_iso(),
        "golden_dir": str(golden_dir),
        "model": model_name,
        "image_size": image_size,
        "n": len(scored),
        "by_label": {
            "pos": sum(1 for row in scored if row["label"] == 1),
            "neg": sum(1 for row in scored if row["label"] == 0),
        },
        "thresholds": sweep,
        "recall_by_detail": golden_recall_by_detail(scored, thresholds),
        "recall_safe_best": recall_safe,
    }
    artifact_stem = golden_dir.name or "golden_verifier"
    (out_dir / f"{artifact_stem}_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / f"{artifact_stem}_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "detail", "source", "page", "score"])
        writer.writeheader()
        for row in sorted(scored, key=lambda item: item["score"], reverse=True):
            out = dict(row)
            out["score"] = f"{row['score']:.8f}"
            writer.writerow(out)


def write_predictions(model, root: Path, pattern: str, tf, device, out_path: Path) -> None:
    import torch
    from torch.utils.data import DataLoader

    paths = sorted(Path(p) for p in glob.glob(str(root / pattern)))
    rows = [{"path": p, "filename": p.relative_to(root).as_posix()} for p in paths if p.is_file()]
    loader = DataLoader(ImageDataset(rows, tf, labeled=False), batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "score", "label"])
        writer.writeheader()
        with torch.no_grad():
            for x, idx in loader:
                x = x.to(device, non_blocking=True)
                probs = torch.softmax(model(pixel_values=x).logits, dim=-1)[:, 1].cpu().tolist()
                for score, row_idx in zip(probs, idx.tolist()):
                    writer.writerow({"filename": rows[row_idx]["filename"], "score": f"{score:.8f}", "label": int(score >= 0.5)})


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    started = time.time()

    root = detect_dataset_root(args.dataset_root, args.train_csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{utc_iso()}] dino_surname_verifier start model={args.model} root={root}", flush=True)

    train_all = read_labeled(root, args.train_csv)
    holdout = read_labeled(root, args.holdout_csv, require_detail=True)
    golden_dir = resolve_optional_dir(root, args.golden_dir)
    golden_rows = read_golden(golden_dir)
    train_rows, val_rows = stratified_split(train_all, args.val_split, args.seed)
    print(
        f"samples: train={len(train_rows)} pos={sum(r['label'] for r in train_rows)} "
        f"val={len(val_rows)} pos={sum(r['label'] for r in val_rows)} "
        f"holdout={len(holdout)} pos={sum(r['label'] for r in holdout)} "
        f"golden={len(golden_rows)} pos={sum(r['label'] for r in golden_rows)}",
        flush=True,
    )

    import torch
    from torch.utils.data import DataLoader

    train_tf, eval_tf = build_transforms(args.image_size, args.augment_profile)
    train_loader = DataLoader(
        ImageDataset(train_rows, train_tf),
        batch_size=args.batch_size,
        sampler=weighted_sampler(train_rows, args.seed),
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ImageDataset(val_rows, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    holdout_loader = DataLoader(
        ImageDataset(holdout, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    golden_loader = (
        DataLoader(
            ImageDataset(golden_rows, eval_tf),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        if golden_rows
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""), flush=True)
    model = make_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best = {"key": None, "val_acc": -1.0, "state": None, "scores": [], "labels": [], "epoch": 0}
    diverse_checkpoints: dict[str, dict] = {}
    log = []
    epoch_checkpoint_dir = out_dir / "model_checkpoints_by_epoch"
    if args.save_checkpoint:
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = correct = total = 0
        for x, y, _idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(pixel_values=x).logits
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.numel()
        val_loss, val_acc, scores, labels, _indices = evaluate(model, val_loader, device)
        holdout_loss_ep, holdout_acc_ep, h_scores_ep, h_labels_ep, _h_indices_ep = evaluate(
            model, holdout_loader, device
        )
        if golden_loader is not None:
            _g_loss_ep, _g_acc_ep, g_scores_ep, g_labels_ep, _g_indices_ep = evaluate(
                model, golden_loader, device
            )
            golden = {
                "best": best_recall_row(g_scores_ep, g_labels_ep),
                "positive_stats": positive_score_stats(g_scores_ep, g_labels_ep),
            }
        else:
            g_scores_ep, g_labels_ep = [], []
            golden = {}
        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(total, 1), 5),
            "train_acc": round(correct / max(total, 1), 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "holdout_loss": round(holdout_loss_ep, 5),
            "holdout_acc_at_0_5": round(holdout_acc_ep, 5),
            "elapsed_s": round(time.time() - t0, 1),
        }
        row.update(val_recall_safe_summary(scores, labels))
        row.update(holdout_epoch_summary(h_scores_ep, h_labels_ep))
        row.update(golden_epoch_summary(g_scores_ep, g_labels_ep))
        state = clone_model_state(model)
        candidate = {
            "val_acc": val_acc,
            "val_loss": val_loss,
            "state": state,
            "scores": scores,
            "labels": labels,
            "holdout_scores": h_scores_ep,
            "holdout_labels": h_labels_ep,
            "golden": golden,
            "epoch": epoch,
        }
        checkpoint_key = checkpoint_key_for_metric(args.checkpoint_metric, candidate)
        row["checkpoint_key"] = list(checkpoint_key)
        diverse_keys = diverse_checkpoint_keys(candidate)
        row["diverse_checkpoint_keys"] = {name: list(key) for name, key in diverse_keys.items()}
        if args.save_checkpoint:
            epoch_dir = epoch_checkpoint_dir / f"epoch_{epoch:02d}"
            model.save_pretrained(epoch_dir)
            row["epoch_checkpoint"] = str(epoch_dir.relative_to(out_dir))
        log.append(row)
        print(f"[epoch {epoch}/{args.epochs}] {row}", flush=True)
        for name, key in diverse_keys.items():
            current = diverse_checkpoints.get(name)
            if current is None or key > current["key"]:
                diverse_checkpoints[name] = {**candidate, "key": key}
        if best["key"] is None or checkpoint_key > best["key"]:
            best = {
                "key": checkpoint_key,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "state": state,
                "scores": scores,
                "labels": labels,
                "holdout_scores": h_scores_ep,
                "holdout_labels": h_labels_ep,
                "golden": golden,
                "epoch": epoch,
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    holdout_loss, holdout_acc, h_scores, h_labels, h_indices = evaluate(model, holdout_loader, device)
    h_report = holdout_report(holdout, h_scores, h_labels, h_indices)
    if golden_loader is not None:
        _golden_loss, _golden_acc, golden_scores, golden_labels, golden_indices = evaluate(model, golden_loader, device)
        golden_report = scored_report(golden_rows, golden_scores, golden_labels, golden_indices)
        golden_report["golden_dir"] = str(golden_dir)
        golden_artifact_stem = golden_dir.name if golden_dir is not None else "golden_verifier"
        (out_dir / f"{golden_artifact_stem}_eval.json").write_text(
            json.dumps(golden_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    v_report = {
        "best_val_acc": round(float(best["val_acc"]), 5),
        "best_val_loss": round(float(best["val_loss"]), 5),
        "best_epoch": best["epoch"],
        "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_key": list(best["key"]) if best["key"] is not None else None,
        "epoch_checkpoints_dir": "model_checkpoints_by_epoch" if args.save_checkpoint else "",
        "diverse_checkpoints": [
            {
                "name": name,
                "epoch": item["epoch"],
                "val_acc": round(float(item["val_acc"]), 5),
                "val_loss": round(float(item["val_loss"]), 5),
                "holdout_min_pos_score": holdout_positive_stats(
                    item["holdout_scores"], item["holdout_labels"]
                )["min_pos_score"],
                "holdout_recall_at_0_003": holdout_positive_stats(
                    item["holdout_scores"], item["holdout_labels"]
                )["recall_at_0_003"],
                "holdout_recall_at_0_01": holdout_positive_stats(
                    item["holdout_scores"], item["holdout_labels"]
                )["recall_at_0_01"],
                "checkpoint_key": list(item["key"]),
                "golden_best": item.get("golden", {}).get("best"),
                "golden_positive_stats": item.get("golden", {}).get("positive_stats"),
            }
            for name, item in sorted(diverse_checkpoints.items())
        ],
        "threshold_sweep": threshold_sweep(best["scores"], best["labels"]),
    }
    (out_dir / "training_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "validation_report.json").write_text(json.dumps(v_report, ensure_ascii=False, indent=2), encoding="utf-8")
    h_report["holdout_loss"] = round(float(holdout_loss), 5)
    h_report["holdout_acc_at_0_5"] = round(float(holdout_acc), 5)
    (out_dir / "holdout_report.json").write_text(json.dumps(h_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if golden_dir is not None and not (out_dir / f"{golden_dir.name}_eval.json").exists():
        write_golden_eval(model, golden_dir, eval_tf, device, out_dir, args.model, args.image_size)

    write_predictions(model, root, args.inference_glob, eval_tf, device, out_dir / "predictions.csv")
    if args.save_checkpoint:
        model.save_pretrained(out_dir / "model_checkpoint")
        top_dir = out_dir / "model_checkpoints_top3"
        for name, item in sorted(diverse_checkpoints.items()):
            model.load_state_dict(item["state"])
            model.save_pretrained(top_dir / f"{name}_epoch_{item['epoch']}")
        if best["state"] is not None:
            model.load_state_dict(best["state"])
    meta = {
        "model": args.model,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "augment_profile": args.augment_profile,
        "checkpoint_metric": args.checkpoint_metric,
        "epoch_checkpoints_dir": "model_checkpoints_by_epoch" if args.save_checkpoint else "",
        "golden_dir": str(golden_dir) if golden_dir else "",
        "dataset_root": str(root),
        "elapsed_s": round(time.time() - started, 1),
        "created": utc_iso(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
