"""Kraken line-мега-пас для дистиляції (remote Kaggle, 2×T4).

Для кожної сторінки датасету: blla-сегментація → вирізання КОЖНОГО рядка
(по-рядково, з дірками замість зсувів індексів) → Kraken-транскрипція рядків
(McCATMuS) тим САМИМ процесом — тож crop_i ↔ kraken_line_i збігаються за
побудовою. Це якір для вирівнювання з CHURRO-текстом (htr_distill_align.py).

Вхід  /kaggle/input/** — сторінки *.jpg + модель *.mlmodel (окремий датасет).
Вихід /kaggle/working/kraken_lines_bundle.tgz:
  lines/<stem>/line_NNN.png + lines/_distill_meta.json (схема htr_distill_extract)
  kraken/<stem>.txt (рядок на seg-рядок, порожній для зфейлених)

⚠ БЕЗ `from __future__ import annotations` — код інжектиться після PARAMS.
"""
import dataclasses
import json
import tarfile
import time
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
OUT = Path("/tmp/klines")


def _find(patterns) -> list:
    seen = {}
    for pat in patterns:
        for p in KAGGLE_INPUT.rglob(pat):
            seen.setdefault(p.name.lower(), p)
    return [seen[k] for k in sorted(seen)]


def main(params: dict[str, Any]) -> None:
    import torch
    from kraken import blla, rpred
    from kraken.kraken import SEGMENTATION_DEFAULT_MODEL
    from kraken.lib import models as kmodels
    from kraken.lib import vgsl
    from kraken.lib.segmentation import extract_polygons
    from PIL import Image

    images = _find(("*.jpg", "*.jpeg", "*.png", "*.JPG"))
    models = _find(("*.mlmodel",))
    if not images:
        raise RuntimeError(f"no images under {KAGGLE_INPUT}")
    if not models:
        raise RuntimeError(f"no *.mlmodel under {KAGGLE_INPUT} (додай model-датасет)")
    rec_path = str(models[0])
    print(f"[klines] {len(images)} pages, model={models[0].name}", flush=True)

    n_gpu = max(1, torch.cuda.device_count())
    (OUT / "lines").mkdir(parents=True, exist_ok=True)
    (OUT / "kraken").mkdir(parents=True, exist_ok=True)
    meta = {"version": 1, "pages": {}}

    t0 = time.time()
    replicas = []
    for gi in range(n_gpu):
        dev = f"cuda:{gi}"
        seg_model = vgsl.TorchVGSLModel.load_model(SEGMENTATION_DEFAULT_MODEL)
        rec_model = kmodels.load_any(rec_path, device=dev)
        replicas.append((dev, seg_model, rec_model))
    print(f"[klines] {n_gpu} replica(s) loaded in {time.time()-t0:.0f}s", flush=True)

    done = [0]

    def run_chunk(replica, chunk):
        dev, seg_model, rec_model = replica
        out = {}
        for src in chunk:
            t1 = time.time()
            stem = src.stem
            try:
                im = Image.open(src).convert("RGB")
                seg = blla.segment(im, model=seg_model, device=dev)
                page_dir = OUT / "lines" / stem
                page_dir.mkdir(exist_ok=True)
                widths, n = [], 0
                for i, ln in enumerate(seg.lines):
                    try:
                        one = dataclasses.replace(seg, lines=[ln])
                        line_im, _ = next(extract_polygons(im, one))
                        line_im.save(page_dir / f"line_{i:03d}.png")
                        widths.append(line_im.width)
                        n += 1
                    except Exception:
                        widths.append(0)
                texts = [""] * len(seg.lines)
                for i, rec in enumerate(rpred.rpred(rec_model, im, seg)):
                    if i < len(texts):
                        texts[i] = rec.prediction or ""
                (OUT / "kraken" / f"{stem}.txt").write_text(
                    "\n".join(texts), encoding="utf-8")
                out[stem] = {"n_lines": n, "widths": widths,
                             "sec": round(time.time() - t1, 1)}
            except Exception as exc:
                out[stem] = {"n_lines": -1,
                             "error": f"{type(exc).__name__}: {exc}"}
            done[0] += 1
            info = out[stem]
            print(f"[klines] {done[0]}/{len(images)} {stem} [{dev}]: "
                  f"{info.get('n_lines')} lines, {info.get('sec', '?')}s"
                  f"{' ERR ' + info['error'] if info.get('error') else ''}",
                  flush=True)
        return out

    per = (len(images) + n_gpu - 1) // n_gpu
    chunks = [images[i * per:(i + 1) * per] for i in range(n_gpu)]
    if n_gpu == 1:
        results = [run_chunk(replicas[0], chunks[0])]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_gpu) as ex:
            futs = [ex.submit(run_chunk, replicas[gi], chunks[gi])
                    for gi in range(n_gpu) if chunks[gi]]
            results = [f.result() for f in futs]
    for r in results:
        meta["pages"].update(r)

    (OUT / "lines" / "_distill_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    bundle = KAGGLE_WORKING / "kraken_lines_bundle.tgz"
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(OUT / "lines", arcname="lines")
        tf.add(OUT / "kraken", arcname="kraken")
    summary = {"n_pages": len(images), "n_gpu": n_gpu,
               "wall_sec": round(time.time() - t0, 1), "finished": _utc_iso()}
    (KAGGLE_WORKING / "klines_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[klines] done: {summary}", flush=True)


if __name__ == "__main__":
    main({})
