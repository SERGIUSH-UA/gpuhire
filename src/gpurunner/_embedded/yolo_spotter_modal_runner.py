"""YOLOv8 handwriting word-spotter — Modal-runtime variant.

A *self-contained Modal app* that trains and runs inference for a single-class
YOLOv8 object detector. Use case: locate one handwritten word (a surname) on
historical archive scans where OCR cannot read the script. Real labeled set is small (~40 images / 64 bboxes); the training mix
is augmented heavily with a synthetic generator (~2000 images).

This file is dual-use:

  1. ``modal run yolo_spotter_modal_runner.py::train --dataset-archive <local.tgz>``
     ``modal run yolo_spotter_modal_runner.py::infer --weights-path runs/<id>/best.pt --images-archive <local.tgz>``
     Invokes the Modal ``local_entrypoint`` wrappers directly; the heavy lift
     happens in ``@app.function`` GPU containers.

  2. ``main(params: dict) -> dict`` keeps the ``_embedded`` shipping contract
     used by ``gpurunner.backends.modal``. A future ``YoloSpotterJob`` can
     ``render_runner_module()`` this file and the framework's ``_REMOTE_WRAPPER``
     will exec the module + call ``main``. When invoked that way, the
     ``@app.function`` decorators are inert (we are already inside Modal); the
     function bodies are reused as plain helpers.

Volume layout (single Modal volume, see ``_VOLUME_NAME``)::

    /vol/
      datasets/<archive_stem>/             # extracted dataset (images/ labels/ data.yaml)
      runs/<run_id>/
        best.pt
        last.pt
        results.csv
        confusion_matrix.png
        ...                                # whatever Ultralytics emits
      infer/<run_id>/
        predictions.json                   # {image_basename: [{x1,y1,x2,y2,conf}]}

Augmentation, mosaic, mixup, hsv, fliplr — Ultralytics defaults. Validation
split is expected to live inside the archive as ``data.yaml`` (with ``train`` /
``val`` keys); if the archive provides only flat ``images/`` + ``labels/`` we
write a 90/10 split data.yaml ourselves.

The container image starts from ``debian_slim`` 3.12, installs CUDA-enabled
``torch`` + ``ultralytics`` + ``opencv-python-headless``. ``ultralytics`` pulls
``yolov8s.pt`` into ``/root/.config/Ultralytics/`` on first use; that cache
survives container restarts via Modal layer caching.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import time
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------
# We import modal at module import time when available (i.e. when ``modal run``
# loads this file). When this module is shipped as runner source via
# gpurunner's ``render_runner_module()``, ``modal`` is also available inside
# the container — but the decorators below have no effect because we are
# *already* executing inside a Modal function. ``main(params)`` then just
# calls the underlying helpers directly.

try:
    import modal  # type: ignore[import-not-found]
    _MODAL_AVAILABLE = True
except Exception:  # pragma: no cover — covers offline test imports
    modal = None  # type: ignore[assignment]
    _MODAL_AVAILABLE = False


# Standalone `modal run` mode only; under the job framework the backend mounts
# whatever `YoloSpotterJob.modal_input_volumes` names.
_VOLUME_NAME = (os.environ.get("GPURUNNER_MODAL_VOLUME_YOLO_SPOTTER")
                or os.environ.get("GPURUNNER_MODAL_VOLUME")
                or "gpurunner-spotter")
_VOLUME_MOUNT = "/vol"

# Per-archive uploads land here first (small enough for FunctionCall payload
# but archives >256 MB should be staged via ``modal volume put`` instead and
# referenced by name — see the ``--dataset-name`` flag on the entrypoints).
_UPLOAD_DIR = "/vol/_uploads"

# Ultralytics installs torch with CUDA wheels picked up automatically once
# Modal attaches a GPU. We pin loose ranges to stay current; pinning hard
# breaks too often as ultralytics fast-moves.
_PIP_PACKAGES = [
    "ultralytics>=8.2,<9.0",
    "torch>=2.2",
    "torchvision>=0.17",
    "opencv-python-headless>=4.9",
    "pillow>=10.0",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "tqdm>=4.66",
]

if _MODAL_AVAILABLE:
    app = modal.App("yolo-spotter")

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("libgl1", "libglib2.0-0")  # opencv runtime
        .pip_install(*_PIP_PACKAGES)
    )

    volume = modal.Volume.from_name(_VOLUME_NAME, create_if_missing=True)
else:  # gpurunner-injection path (decorators are no-ops at module top)
    app = None  # type: ignore[assignment]
    image = None  # type: ignore[assignment]
    volume = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Utilities (work both inside and outside Modal)
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _extract_archive(archive_path: Path, dest: Path) -> Path:
    """Extract .tar.gz / .tgz / .tar / .zip into ``dest``. Returns dest."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    else:
        raise ValueError(f"unsupported archive format: {archive_path.name}")
    return dest


def _make_archive(src_dir: Path, archive_path: Path) -> Path:
    """Pack ``src_dir`` contents into a tar.gz at ``archive_path``."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(src_dir)))
    return archive_path


def _find_dataset_root(extracted: Path) -> Path:
    """Find the directory that contains ``images/`` and ``labels/``.

    Archives often wrap the dataset in a single top-level folder; descend at
    most 3 levels to find the real root.
    """
    candidates = [extracted, *extracted.glob("*/"), *extracted.glob("*/*/")]
    for c in candidates:
        if not c.is_dir():
            continue
        if (c / "images").is_dir() and (c / "labels").is_dir():
            return c
    raise FileNotFoundError(
        f"dataset archive must contain images/ and labels/ — searched {extracted}"
    )


def _ensure_data_yaml(dataset_root: Path, val_split: float, seed: int) -> Path:
    """Return path to data.yaml; synthesise a 90/10 split if missing."""
    existing = dataset_root / "data.yaml"
    if existing.exists():
        # Trust caller's yaml — but Ultralytics needs absolute paths.
        return _rewrite_yaml_abs(existing, dataset_root)

    import random as _random

    images_dir = dataset_root / "images"
    all_imgs = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not all_imgs:
        raise FileNotFoundError(f"no images under {images_dir}")

    rng = _random.Random(seed)
    rng.shuffle(all_imgs)
    n_val = max(1, int(len(all_imgs) * val_split))
    val = all_imgs[:n_val]
    train = all_imgs[n_val:]

    splits_dir = dataset_root / "_splits"
    splits_dir.mkdir(exist_ok=True)
    train_txt = splits_dir / "train.txt"
    val_txt = splits_dir / "val.txt"
    train_txt.write_text(
        "\n".join(str(p.resolve()) for p in train), encoding="utf-8"
    )
    val_txt.write_text(
        "\n".join(str(p.resolve()) for p in val), encoding="utf-8"
    )

    yaml_text = (
        f"path: {dataset_root.resolve()}\n"
        f"train: {train_txt.resolve()}\n"
        f"val: {val_txt.resolve()}\n"
        "nc: 1\n"
        "names: ['word']\n"
    )
    out = dataset_root / "data.yaml"
    out.write_text(yaml_text, encoding="utf-8")
    return out


def _rewrite_yaml_abs(yaml_path: Path, dataset_root: Path) -> Path:
    """Rewrite a user-provided data.yaml so paths are absolute.

    Ultralytics resolves relative paths against ``ULTRALYTICS_CONFIG_DIR``,
    which inside the container is not the dataset directory. Easier to
    pre-rewrite than to chdir.
    """
    import yaml  # provided by ultralytics' deps

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    root = dataset_root.resolve()

    def _abs(p: Any) -> Any:
        if isinstance(p, str) and not p.startswith("/"):
            return str((root / p).resolve())
        return p

    def _rewrite_split_txt(p: Any) -> Any:
        p = _abs(p)
        if not isinstance(p, str):
            return p
        txt = Path(p)
        if txt.suffix.lower() != ".txt" or not txt.exists():
            return p
        lines = []
        changed = False
        for line in txt.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item:
                continue
            if item.startswith("/"):
                lines.append(item)
            else:
                lines.append(str((root / item).resolve()))
                changed = True
        if not changed:
            return p
        out_txt = txt.with_name(f"{txt.stem}.absolute{txt.suffix}")
        out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(out_txt.resolve())

    if "path" not in data:
        data["path"] = str(root)
    for k in ("train", "val", "test"):
        if k in data:
            data[k] = _rewrite_split_txt(data[k])
    data.setdefault("nc", 1)
    data.setdefault("names", ["word"])

    out = dataset_root / "data.absolute.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1.0, aa + ba - inter)


def _target_coverage(det: list[float], target: list[float]) -> float:
    dx1, dy1, dx2, dy2 = det
    tx1, ty1, tx2, ty2 = target
    ix1, iy1 = max(dx1, tx1), max(dy1, ty1)
    ix2, iy2 = min(dx2, tx2), min(dy2, ty2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    target_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
    return inter / target_area


def _center_inside(det: list[float], target: list[float]) -> bool:
    dx1, dy1, dx2, dy2 = det
    tx1, ty1, tx2, ty2 = target
    cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
    return tx1 <= cx <= tx2 and ty1 <= cy <= ty2


def _image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _materialize_audit_pack(audit_pack: str | None, vol_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    if not audit_pack:
        return None, None
    src = Path(audit_pack)
    if not src.is_absolute():
        src = vol_root / "audits" / audit_pack
    local_root = Path("/tmp/yolo_recall_audit") / src.stem.replace(".tar", "")
    if src.is_file():
        if local_root.exists():
            shutil.rmtree(local_root)
        local_root.mkdir(parents=True, exist_ok=True)
        _extract_archive(src, local_root)
    elif src.is_dir():
        local_root = src
    else:
        raise FileNotFoundError(
            f"audit_pack={audit_pack!r} not found. Expected /vol/audits/<name>.tgz or directory."
        )
    manifest = local_root / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"audit pack has no manifest.json: {local_root}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        f"[train][audit] loaded pack {audit_pack}: "
        f"critical={len(data.get('critical') or [])} "
        f"golden_pos={len((data.get('sets') or {}).get('golden_pos') or [])}",
        flush=True,
    )
    return local_root, data


def _audit_predict_boxes(model: Any, image: Path, imgsz: int, infer_conf: float) -> list[dict[str, Any]]:
    result = model.predict(str(image), imgsz=imgsz, conf=infer_conf, verbose=False)[0]
    out = []
    for box in result.boxes:
        out.append({
            "conf": float(box.conf[0]),
            "cls": int(box.cls[0]),
            "box": [float(v) for v in box.xyxy[0].tolist()],
        })
    return out


def _audit_max_conf(model: Any, image: Path, imgsz: int, infer_conf: float) -> float:
    boxes = _audit_predict_boxes(model, image, imgsz, infer_conf)
    return max((float(b["conf"]) for b in boxes), default=0.0)


def _run_realtime_audit(
    *,
    model: Any,
    audit_root: Path,
    manifest: dict[str, Any],
    imgsz: int,
    conf: float,
    infer_conf: float,
    hard_neg_limit: int,
) -> dict[str, Any]:
    min_coverage = float(manifest.get("min_coverage", 0.35))
    min_iou = float(manifest.get("min_iou", 0.10))
    critical = manifest.get("critical") or []
    sets = manifest.get("sets") or {}

    critical_rows = []
    critical_hits = 0
    for item in critical:
        image = audit_root / str(item["image"])
        target = [float(v) for v in item["bbox"]]
        boxes = _audit_predict_boxes(model, image, imgsz, infer_conf)
        best = {
            "conf": 0.0,
            "iou": 0.0,
            "coverage": 0.0,
            "center_inside": False,
            "box": None,
        }
        for det in boxes:
            box = list(det["box"])
            rec = {
                "conf": float(det["conf"]),
                "iou": _box_iou(box, target),
                "coverage": _target_coverage(box, target),
                "center_inside": _center_inside(box, target),
                "box": [round(v, 1) for v in box],
            }
            if (rec["coverage"], rec["iou"], rec["conf"]) > (
                best["coverage"],
                best["iou"],
                best["conf"],
            ):
                best = rec
        hit = (
            best["conf"] >= conf
            and (
                best["coverage"] >= min_coverage
                or best["iou"] >= min_iou
                or bool(best["center_inside"])
            )
        )
        critical_hits += int(hit)
        critical_rows.append({"id": item.get("id"), "hit": hit, **best})

    def _count(paths: list[str]) -> tuple[int, int]:
        hit = 0
        total = 0
        for rel in paths:
            c = _audit_max_conf(model, audit_root / rel, imgsz, infer_conf)
            hit += int(c >= conf)
            total += 1
        return hit, total

    golden_pos = list(sets.get("golden_pos") or [])
    golden_neg = list(sets.get("golden_neg") or [])
    rod_pos = list(sets.get("rod_pos") or [])
    rod_neg = list(sets.get("rod_neg") or [])
    hard_neg = list(sets.get("hard_neg") or [])[:hard_neg_limit]

    golden_hit, golden_total = _count(golden_pos)
    golden_fp, golden_neg_total = _count(golden_neg)
    rod_hit, rod_total = _count(rod_pos)
    rod_fp, rod_neg_total = _count(rod_neg)
    hard_fp, hard_total = _count(hard_neg)

    return {
        "critical_hits": critical_hits,
        "critical_total": len(critical),
        "golden_pos_hits": golden_hit,
        "golden_pos_total": golden_total,
        "golden_neg_fp": golden_fp,
        "golden_neg_total": golden_neg_total,
        "rod_pos_hits": rod_hit,
        "rod_pos_total": rod_total,
        "rod_neg_fp": rod_fp,
        "rod_neg_total": rod_neg_total,
        "hard_neg_fp": hard_fp,
        "hard_neg_total": hard_total,
        "critical": critical_rows,
        "conf": conf,
        "infer_conf": infer_conf,
    }


# ---------------------------------------------------------------------------
# Train / infer cores (the actual GPU work — invoked by both the @app.function
# decorated wrappers below AND by ``main(params)`` when running inside the
# gpurunner-framework wrapper).
# ---------------------------------------------------------------------------

def _train_core(
    *,
    dataset_archive_bytes: bytes | None = None,
    dataset_name: str | None = None,
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    base_model: str = "yolov8s.pt",
    val_split: float = 0.10,
    seed: int = 42,
    run_id: str | None = None,
    save_period: int = -1,
    cache: str = "disk",
    workers: int = 12,
    audit_pack: str | None = None,
    audit_every: int = 0,
    audit_conf: float = 0.05,
    audit_infer_conf: float = 0.01,
    audit_hard_neg_limit: int = 20,
) -> dict[str, Any]:
    """Train a YOLOv8 single-class spotter. Runs inside the Modal container.

    Either ``dataset_archive_bytes`` (raw archive blob) OR ``dataset_name``
    (path inside ``/vol/datasets/`` already populated via ``modal volume put``)
    must be supplied.
    """
    from ultralytics import YOLO

    started = time.time()
    run_id = run_id or f"yolospot-{datetime.now(tz=UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    print(f"[{_utc_iso()}] train start run_id={run_id} base={base_model} "
          f"epochs={epochs} bs={batch_size} imgsz={imgsz}", flush=True)

    vol_root = Path(_VOLUME_MOUNT)
    datasets_dir = vol_root / "datasets"
    runs_dir = vol_root / "runs"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ---- materialise dataset ------------------------------------------------
    if dataset_archive_bytes is not None:
        upload_dir = Path(_UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        archive_path = upload_dir / f"{run_id}.archive"
        archive_path.write_bytes(dataset_archive_bytes)
        # Sniff format from magic bytes
        head = dataset_archive_bytes[:4]
        if head[:2] == b"PK":
            archive_path = archive_path.with_suffix(".zip")
            (upload_dir / f"{run_id}.archive").rename(archive_path)
        else:
            archive_path = archive_path.with_suffix(".tar.gz")
            (upload_dir / f"{run_id}.archive").rename(archive_path)
        extract_to = datasets_dir / run_id
        _extract_archive(archive_path, extract_to)
        dataset_root = _find_dataset_root(extract_to)
        print(f"[train] extracted archive → {dataset_root}", flush=True)
    elif dataset_name:
        archive = next((datasets_dir / f"{dataset_name}{ext}"
                        for ext in (".tgz", ".tar.gz", ".zip")
                        if (datasets_dir / f"{dataset_name}{ext}").is_file()), None)
        vol_dir = datasets_dir / dataset_name
        if archive is not None:
            # ⚡ ВИТЯГ НА ЛОКАЛЬНИЙ ДИСК (/tmp, швидкий SSD), НЕ на мережевий volume.
            # Розпакування тисяч дрібних файлів НА VOLUME ~1 файл/с → години (інцидент
            # v14 2026-05-25). Локальний диск розпаковує ті самі файли за секунди.
            # tgz читається з volume (1 великий файл — швидко), ваги пишуться на volume.
            local_dir = Path("/tmp/ds") / dataset_name
            if not local_dir.is_dir():
                print(f"[train] extracting {archive.name} → {local_dir} (local SSD)", flush=True)
                local_dir.mkdir(parents=True, exist_ok=True)
                _extract_archive(archive, local_dir)
            dataset_root = _find_dataset_root(local_dir)
        elif vol_dir.is_dir():
            dataset_root = _find_dataset_root(vol_dir)
        else:
            raise FileNotFoundError(
                f"dataset_name={dataset_name!r} not found under {datasets_dir} "
                f"(neither dir nor .tgz/.tar.gz/.zip). "
                f"Use `modal volume put {_VOLUME_NAME} <local> datasets/{dataset_name}` first."
            )
        print(f"[train] using dataset {dataset_root}", flush=True)
    else:
        raise ValueError("either dataset_archive_bytes or dataset_name is required")

    data_yaml = _ensure_data_yaml(dataset_root, val_split=val_split, seed=seed)
    print(f"[train] data.yaml: {data_yaml}", flush=True)
    print(f"[train]   {data_yaml.read_text(encoding='utf-8').strip()}", flush=True)

    # ---- train --------------------------------------------------------------
    project_dir = runs_dir / run_id
    project_dir.mkdir(parents=True, exist_ok=True)
    audit_root, audit_manifest = _materialize_audit_pack(audit_pack, vol_root)

    model_ref = base_model
    model_path = Path(base_model)
    if not model_path.is_absolute():
        vol_model_path = vol_root / base_model
        if vol_model_path.exists():
            model_ref = str(vol_model_path)
    print(f"[train] model_ref: {model_ref}", flush=True)
    model = YOLO(model_ref)

    # --- ПЕРІОДИЧНИЙ БЕКАП ЧЕКПОЙНТА (фікс 2026-05-24) ---------------------
    # Ultralytics пише last.pt/best.pt у /vol/runs/.../weights ЩОЕПОХИ, АЛЕ
    # записи в Modal Volume довговічні ЛИШЕ після volume.commit(). Раніше commit
    # був тільки в кінці train → kill по timeout відкидав УСІ незакомічені ваги
    # (v13 втратила ~22 епохи). Тепер комітимо кожні CKPT_EVERY епох: kill втратить
    # максимум CKPT_EVERY-1 епох, і run можна продовжити з last.pt (resume=True).
    CKPT_EVERY = 3
    if _MODAL_AVAILABLE and volume is not None:
        def _commit_ckpt(trainer):  # on_fit_epoch_end → last.pt вже на диску
            ep = int(getattr(trainer, "epoch", 0)) + 1
            if ep % CKPT_EVERY == 0:
                try:
                    volume.commit()
                    print(f"[train][ckpt] volume.commit() after epoch {ep}", flush=True)
                except Exception as e:
                    print(f"[train][ckpt][warn] commit failed @ep{ep}: {e}", flush=True)
        model.add_callback("on_fit_epoch_end", _commit_ckpt)

    if audit_root is not None and audit_manifest is not None and audit_every > 0:
        audit_jsonl = project_dir / "realtime_audit.jsonl"

        def _audit_epoch(trainer):
            ep = int(getattr(trainer, "epoch", 0)) + 1
            if ep % audit_every != 0:
                return
            weights = project_dir / "weights" / "last.pt"
            if not weights.exists():
                print(f"[train][audit][warn] epoch {ep}: no last.pt yet", flush=True)
                return
            try:
                audit_model = YOLO(str(weights))
                row = _run_realtime_audit(
                    model=audit_model,
                    audit_root=audit_root,
                    manifest=audit_manifest,
                    imgsz=imgsz,
                    conf=audit_conf,
                    infer_conf=audit_infer_conf,
                    hard_neg_limit=audit_hard_neg_limit,
                )
                row["epoch"] = ep
                row["weights"] = str(weights.relative_to(vol_root))
                with audit_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"[train][audit] ep={ep} "
                    f"critical={row['critical_hits']}/{row['critical_total']} "
                    f"golden={row['golden_pos_hits']}/{row['golden_pos_total']} "
                    f"golden_fp={row['golden_neg_fp']}/{row['golden_neg_total']} "
                    f"rod={row['rod_pos_hits']}/{row['rod_pos_total']} "
                    f"hard_fp={row['hard_neg_fp']}/{row['hard_neg_total']} "
                    f"conf={audit_conf}",
                    flush=True,
                )
                if _MODAL_AVAILABLE and volume is not None:
                    volume.commit()
            except Exception as e:
                print(f"[train][audit][warn] epoch {ep}: audit failed: {e}", flush=True)
            finally:
                try:
                    del audit_model
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        model.add_callback("on_fit_epoch_end", _audit_epoch)

    # Ultralytics writes into <project>/<name>/{weights, results.csv, ...}
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=str(runs_dir),
        name=run_id,
        exist_ok=True,
        seed=seed,
        # ── АУГМЕНТАЦІЯ (v14, 2026-05-25): augmentation-only замість отруєної
        # copy-paste синтетики. Трансформуємо РЕАЛЬНІ позитиви на льоту — прізвище
        # лишається у своєму контексті тим самим чорнилом (отрута неможлива).
        # USER-список: нахил / стиснення-розтяг / колір чорнила всього тексту / руйнація.
        # (90° розворот — НЕ тут, ultralytics не вміє дискретні; робиться офлайн-копіями.)
        degrees=10.0,          # нахил ±10° (реальні скани похилі)
        scale=0.6,             # стиснення/розтяг масштабу (ширше за дефолт 0.5)
        shear=4.0,             # зсув/розтяг
        perspective=0.0005,    # легка перспектива (фото під кутом)
        translate=0.1,         # зсув
        hsv_h=0.10,            # ВІДТІНОК: міняє колір чорнила ВСЬОГО тексту разом (однорідно!)
        hsv_s=0.7, hsv_v=0.5,  # насиченість/яскравість (вицвіле/насичене чорнило, експозиція)
        erasing=0.4,           # руйнація: випадкове стирання ділянок (плями/дефекти)
        # fliplr/flipud=0 — текст НЕ дзеркалити (дзеркальні літери ≠ те саме слово).
        fliplr=0.0,
        flipud=0.0,
        # Single GPU, default device.
        device=0,
        # save_period=-1 → лише best/last (економить місце). save_period=1 →
        # чекпойнт КОЖНОЇ епохи (epochN.pt) для per-epoch eval (recall-вибір епохи,
        # бо ultralytics best.pt = best mAP50, а наш критерій = recall@low-conf).
        save_period=save_period,
        # cache='disk' → декодувати PNG-тайли у .npy ОДИН раз (на локальний /tmp SSD),
        # далі читати raw — прибирає повторне декодування щоепохи (data-bound боттлнек
        # на imgsz1280). workers — паралельні data-loader'и (потребує cpu↑ у контейнері).
        # cache='ram' НЕ юзаємо: 20k×3MB тайлів ≈ 65GB > RAM контейнера.
        cache=cache,
        workers=workers,
        # Print to stdout (Modal log capture)
        verbose=True,
    )

    # Locate output weights (Ultralytics layout)
    out_dir = runs_dir / run_id
    weights_dir = out_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    results_csv = out_dir / "results.csv"

    # Best-effort metrics extraction
    metrics: dict[str, Any] = {}
    try:
        # ``results`` is a ultralytics.utils.metrics.DetMetrics-like object
        if hasattr(results, "results_dict"):
            metrics = {
                k: float(v) if hasattr(v, "__float__") else v
                for k, v in results.results_dict.items()
            }
    except Exception as e:
        print(f"[train][warn] could not extract metrics: {e}", flush=True)

    # Commit volume so client sees the writes
    try:
        if _MODAL_AVAILABLE and volume is not None:
            volume.commit()
    except Exception as e:
        print(f"[train][warn] volume.commit failed: {e}", flush=True)

    elapsed = round(time.time() - started, 1)
    summary = {
        "run_id": run_id,
        "weights_path": str(best_pt.relative_to(vol_root)) if best_pt.exists() else None,
        "last_weights_path": str(last_pt.relative_to(vol_root)) if last_pt.exists() else None,
        "results_csv": str(results_csv.relative_to(vol_root)) if results_csv.exists() else None,
        "project_dir": str(out_dir.relative_to(vol_root)),
        "metrics": metrics,
        "elapsed_s": elapsed,
        "started_at": _utc_iso(),
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "base_model": base_model,
        "audit_pack": audit_pack,
    }
    print(f"[{_utc_iso()}] train done in {elapsed}s — best={summary['weights_path']}", flush=True)
    return summary


def _infer_core(
    *,
    weights_path: str,
    images_archive_bytes: bytes | None = None,
    images_dirname: str | None = None,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run YOLOv8 predict on an image set. Returns per-image bbox JSON.

    Either ``images_archive_bytes`` or ``images_dirname`` (path inside
    ``/vol/datasets/`` already populated) must be supplied. ``weights_path``
    is relative to ``/vol`` (e.g. ``runs/<run_id>/weights/best.pt``).
    """
    from ultralytics import YOLO

    started = time.time()
    run_id = run_id or f"infer-{datetime.now(tz=UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    print(f"[{_utc_iso()}] infer start run_id={run_id} weights={weights_path} "
          f"conf={conf_threshold} imgsz={imgsz}", flush=True)

    vol_root = Path(_VOLUME_MOUNT)
    weights_full = vol_root / weights_path
    if not weights_full.exists():
        raise FileNotFoundError(f"weights not found: {weights_full}")

    # ---- materialise images -------------------------------------------------
    if images_archive_bytes is not None:
        upload_dir = Path(_UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        head = images_archive_bytes[:4]
        if head[:2] == b"PK":
            archive_path = upload_dir / f"{run_id}.zip"
        else:
            archive_path = upload_dir / f"{run_id}.tar.gz"
        archive_path.write_bytes(images_archive_bytes)
        extract_to = vol_root / "infer_inputs" / run_id
        _extract_archive(archive_path, extract_to)
        images_root = extract_to
    elif images_dirname:
        images_root = vol_root / "datasets" / images_dirname
        if not images_root.is_dir():
            raise FileNotFoundError(
                f"images_dirname={images_dirname!r} not found under {vol_root / 'datasets'}"
            )
    else:
        raise ValueError("either images_archive_bytes or images_dirname is required")

    # Find images (recurse — archives often have nested subdirs)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_paths = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in exts)
    if not img_paths:
        raise FileNotFoundError(f"no images found under {images_root}")
    print(f"[infer] {len(img_paths)} images", flush=True)

    # ---- predict ------------------------------------------------------------
    model = YOLO(str(weights_full))

    # We deliberately do not use Ultralytics' `save=True` (writes annotated
    # images, large). We loop manually and collect JSON.
    predictions: dict[str, list[dict[str, float]]] = {}
    batch_size = 16
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i : i + batch_size]
        # `predict` accepts list[str | Path]; ``verbose=False`` quiets per-batch logs.
        results_list = model.predict(
            source=[str(p) for p in batch],
            conf=conf_threshold,
            imgsz=imgsz,
            device=0,
            verbose=False,
            save=False,
        )
        for src_path, res in zip(batch, results_list, strict=True):
            rel = str(src_path.relative_to(images_root))
            boxes_out: list[dict[str, float]] = []
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = (float(v) for v in xyxy[j])
                    boxes_out.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(conf[j]),
                    })
            predictions[rel] = boxes_out

    # ---- persist ------------------------------------------------------------
    infer_dir = vol_root / "infer" / run_id
    infer_dir.mkdir(parents=True, exist_ok=True)
    pred_path = infer_dir / "predictions.json"
    pred_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        if _MODAL_AVAILABLE and volume is not None:
            volume.commit()
    except Exception as e:
        print(f"[infer][warn] volume.commit failed: {e}", flush=True)

    elapsed = round(time.time() - started, 1)
    n_with_hits = sum(1 for v in predictions.values() if v)
    n_total_boxes = sum(len(v) for v in predictions.values())
    summary = {
        "run_id": run_id,
        "weights_path": weights_path,
        "predictions_path": str(pred_path.relative_to(vol_root)),
        "n_images": len(img_paths),
        "n_images_with_hits": n_with_hits,
        "n_total_boxes": n_total_boxes,
        "conf_threshold": conf_threshold,
        "elapsed_s": elapsed,
        "started_at": _utc_iso(),
        # Inline predictions in return value too (capped — Modal payload limit
        # 256 MB; bbox JSON for ~2000 images is well under that).
        "predictions": predictions,
    }
    print(f"[{_utc_iso()}] infer done in {elapsed}s — "
          f"{n_with_hits}/{len(img_paths)} images had hits ({n_total_boxes} boxes)",
          flush=True)
    return summary


# ---------------------------------------------------------------------------
# Modal @app.function wrappers
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE:

    @app.function(  # type: ignore[union-attr]
        image=image,
        gpu="T4",
        # cpu НЕ задаємо: cache='disk' прискорює лише швидкі GPU; на дефолтному T4 cpu↑
        # марний і дорожчає run (тягне RAM). Для A100+ підіймати через gpurunner -p cpu=8.
        volumes={_VOLUME_MOUNT: volume},
        timeout=240 * 60,  # 240 min — imgsz 1280 / large oversampled datasets
        # NB: 120 min було замало для v13 (5498 train @1280, ~156 min/30ep) →
        # Modal убив run на 7200s до volume.commit() (save_period=-1 → ваги пропали).
    )
    def train_remote(
        dataset_archive_bytes: bytes | None = None,
        dataset_name: str | None = None,
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        base_model: str = "yolov8s.pt",
        val_split: float = 0.10,
        seed: int = 42,
        run_id: str | None = None,
        save_period: int = -1,
        cache: str = "disk",
        workers: int = 12,
        audit_pack: str | None = None,
        audit_every: int = 0,
        audit_conf: float = 0.05,
        audit_infer_conf: float = 0.01,
        audit_hard_neg_limit: int = 20,
    ) -> dict[str, Any]:
        return _train_core(
            dataset_archive_bytes=dataset_archive_bytes,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            base_model=base_model,
            val_split=val_split,
            seed=seed,
            run_id=run_id,
            save_period=save_period,
            cache=cache,
            workers=workers,
            audit_pack=audit_pack,
            audit_every=audit_every,
            audit_conf=audit_conf,
            audit_infer_conf=audit_infer_conf,
            audit_hard_neg_limit=audit_hard_neg_limit,
        )

    @app.function(  # type: ignore[union-attr]
        image=image,
        gpu="T4",
        volumes={_VOLUME_MOUNT: volume},
        timeout=15 * 60,  # 15 min for inference
    )
    def infer_remote(
        weights_path: str,
        images_archive_bytes: bytes | None = None,
        images_dirname: str | None = None,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return _infer_core(
            weights_path=weights_path,
            images_archive_bytes=images_archive_bytes,
            images_dirname=images_dirname,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            run_id=run_id,
        )

    # ---- local entrypoints (callable via `modal run`) ---------------------

    @app.local_entrypoint()  # type: ignore[union-attr]
    def train(
        dataset_archive: str = "",
        dataset_name: str = "",
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        base_model: str = "yolov8s.pt",
        val_split: float = 0.10,
        seed: int = 42,
        run_id: str = "",
        save_period: int = -1,
        cache: str = "disk",
        workers: int = 12,
        audit_pack: str = "",
        audit_every: int = 0,
        audit_conf: float = 0.05,
        audit_infer_conf: float = 0.01,
        audit_hard_neg_limit: int = 20,
    ) -> None:
        """Train YOLOv8 spotter.

        Provide ONE of:
          --dataset-archive <local-path>  : tar.gz / zip of images/ + labels/
          --dataset-name <subdir>         : already at /vol/datasets/<subdir>

        Examples (PowerShell):
          uv run modal run yolo_spotter_modal_runner.py::train \\
              --dataset-archive ./datasets/v1.tgz \\
              --epochs 50 --batch-size 16

          uv run modal run yolo_spotter_modal_runner.py::train \\
              --dataset-name synthetic-v1 --epochs 80
        """
        if not dataset_archive and not dataset_name:
            raise SystemExit("--dataset-archive or --dataset-name is required")
        if dataset_archive and dataset_name:
            raise SystemExit("pass exactly one of --dataset-archive / --dataset-name")

        kwargs: dict[str, Any] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "base_model": base_model,
            "val_split": val_split,
            "seed": seed,
            "run_id": run_id or None,
            "save_period": save_period,
            "cache": cache,
            "workers": workers,
            "audit_pack": audit_pack or None,
            "audit_every": audit_every,
            "audit_conf": audit_conf,
            "audit_infer_conf": audit_infer_conf,
            "audit_hard_neg_limit": audit_hard_neg_limit,
        }
        if dataset_archive:
            archive_bytes = Path(dataset_archive).read_bytes()
            size_mb = len(archive_bytes) / 1024 / 1024
            print(f"[local] uploading dataset_archive ({size_mb:.1f} MB)", flush=True)
            if size_mb > 250:
                print(
                    "[local][warn] archive >250 MB — consider `modal volume put "
                    f"{_VOLUME_NAME} {dataset_archive} datasets/<name>` and pass "
                    "--dataset-name instead.",
                    flush=True,
                )
            kwargs["dataset_archive_bytes"] = archive_bytes
        else:
            kwargs["dataset_name"] = dataset_name

        # .spawn() submits the call as an independent FunctionCall that Modal
        # tracks regardless of this client — unlike .remote(), it is NOT canceled
        # when the local caller disconnects (works correctly under `modal run --detach`).
        call = train_remote.spawn(**kwargs)
        print(f"[local] spawned train_remote — FunctionCall id: {call.object_id}", flush=True)
        print(
            "[local] survives client disconnect. Reconnect later with:\n"
            f"        uv run python -c \"import modal; "
            f"print(modal.FunctionCall.from_id('{call.object_id}').get())\"",
            flush=True,
        )
        result = call.get()  # blocks for the result, but the call itself is disconnect-safe
        print(json.dumps(
            {k: v for k, v in result.items() if k != "metrics"} | {"metrics": result.get("metrics")},
            ensure_ascii=False, indent=2,
        ))

    @app.local_entrypoint()  # type: ignore[union-attr]
    def infer(
        weights_path: str,
        images_archive: str = "",
        images_dirname: str = "",
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        run_id: str = "",
        out: str = "",
    ) -> None:
        """Run YOLOv8 spotter inference.

        ``weights_path`` is relative to the volume root, e.g.
        ``runs/<run_id>/weights/best.pt``.

        Provide ONE of:
          --images-archive <local-path>  : tar.gz / zip of images
          --images-dirname <subdir>      : already at /vol/datasets/<subdir>

        ``--out <local.json>`` (optional): dump predictions.json locally too.
        """
        if not images_archive and not images_dirname:
            raise SystemExit("--images-archive or --images-dirname is required")
        if images_archive and images_dirname:
            raise SystemExit("pass exactly one of --images-archive / --images-dirname")

        kwargs: dict[str, Any] = {
            "weights_path": weights_path,
            "conf_threshold": conf_threshold,
            "imgsz": imgsz,
            "run_id": run_id or None,
        }
        if images_archive:
            archive_bytes = Path(images_archive).read_bytes()
            size_mb = len(archive_bytes) / 1024 / 1024
            print(f"[local] uploading images_archive ({size_mb:.1f} MB)", flush=True)
            if size_mb > 250:
                print(
                    "[local][warn] archive >250 MB — consider `modal volume put` "
                    "and pass --images-dirname instead.",
                    flush=True,
                )
            kwargs["images_archive_bytes"] = archive_bytes
        else:
            kwargs["images_dirname"] = images_dirname

        # .spawn() — disconnect-safe under `modal run --detach` (see train()).
        call = infer_remote.spawn(**kwargs)
        print(f"[local] spawned infer_remote — FunctionCall id: {call.object_id}", flush=True)
        result = call.get()

        # Print everything except the (potentially large) inline predictions.
        slim = {k: v for k, v in result.items() if k != "predictions"}
        print(json.dumps(slim, ensure_ascii=False, indent=2))

        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(
                json.dumps(result["predictions"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[local] wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# gpurunner-framework entry point — keeps the ``_embedded`` shipping contract.
# ---------------------------------------------------------------------------

def main(params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to ``_train_core`` / ``_infer_core`` based on ``params['mode']``.

    Called by ``gpurunner.backends.modal``'s ``_REMOTE_WRAPPER`` when a future
    ``YoloSpotterJob`` ships this file via ``render_runner_module()``. The
    wrapper passes ``params["output_root"] = "/mnt/outputs"`` (not used here —
    we own our own volume layout under ``/vol``) and expects ``main`` to return
    a JSON-serialisable summary dict.

    Required: ``params["mode"] ∈ {"train", "infer"}``. The rest of the params
    forward to the matching ``_*_core`` keyword args. To pass dataset bytes
    via this path, base64-encode them as ``params["dataset_archive_b64"]``;
    larger inputs should be staged on the volume beforehand.
    """
    import base64

    mode = str(params.get("mode", "")).lower()
    if mode not in {"train", "infer"}:
        raise ValueError(f"params['mode'] must be 'train' or 'infer', got {mode!r}")

    if mode == "train":
        b64 = params.get("dataset_archive_b64")
        archive_bytes = base64.b64decode(b64) if b64 else None
        return _train_core(
            dataset_archive_bytes=archive_bytes,
            # gpurunner's YoloSpotterJob.validate_params normalises the volume
            # entry under the key "dataset" (Kaggle uses "<owner>/<slug>" there);
            # accept either so `gpurunner run yolo_spotter -b modal -p dataset=…`
            # reaches _train_core instead of raising "dataset_name is required".
            dataset_name=params.get("dataset_name") or params.get("dataset"),
            epochs=int(params.get("epochs", 50)),
            batch_size=int(params.get("batch_size", 16)),
            imgsz=int(params.get("imgsz", 640)),
            base_model=str(params.get("base_model", "yolov8s.pt")),
            val_split=float(params.get("val_split", 0.10)),
            seed=int(params.get("seed", 42)),
            run_id=params.get("run_id"),
            save_period=int(params.get("save_period", -1)),
            cache=str(params.get("cache", "disk")),
            workers=int(params.get("workers", 12)),
            audit_pack=params.get("audit_pack"),
            audit_every=int(params.get("audit_every", 0)),
            audit_conf=float(params.get("audit_conf", 0.05)),
            audit_infer_conf=float(params.get("audit_infer_conf", 0.01)),
            audit_hard_neg_limit=int(params.get("audit_hard_neg_limit", 20)),
        )

    # infer
    b64 = params.get("images_archive_b64")
    images_bytes = base64.b64decode(b64) if b64 else None
    return _infer_core(
        weights_path=str(params["weights_path"]),
        images_archive_bytes=images_bytes,
        images_dirname=params.get("images_dirname"),
        conf_threshold=float(params.get("conf_threshold", 0.25)),
        imgsz=int(params.get("imgsz", 640)),
        run_id=params.get("run_id"),
    )


# Suppress unused-warning for shutil/os/sys when the module is imported but
# entrypoints are not invoked (e.g. during the smoke test below).
_ = (shutil, os, sys)


if __name__ == "__main__":  # pragma: no cover
    # Plain ``python yolo_spotter_modal_runner.py`` is not the intended path —
    # use ``modal run`` instead. Print a hint and exit.
    print(
        "this file is meant to be invoked via:\n"
        "  uv run modal run src/gpurunner/_embedded/yolo_spotter_modal_runner.py::train ...\n"
        "  uv run modal run src/gpurunner/_embedded/yolo_spotter_modal_runner.py::infer ...\n"
        "or imported as a gpurunner runner via render_runner_module()."
    )
