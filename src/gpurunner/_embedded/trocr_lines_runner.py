"""TrOCR line-пас: blla-сегментація → кропи рядків → TrOCR-транскрипція (GPU).

Кейс: пілот якості Kansallisarkisto/cyrillic-htr-model на кириличних
метриках (ф.904 оп.24) → якщо читає, стає вчителем для кириличного kraken.

Для кожної сторінки: blla-сегментація → вирізання КОЖНОГО рядка (діри замість
зсувів індексів) → батчева TrOCR-транскрипція кропів. crop_i ↔ text_i збігаються
за побудовою — та сама схема, що kraken_lines (htr_distill_extract-сумісна).

Вхід  /kaggle/input/** — сторінки *.jpg/*.png.
Вихід /kaggle/working/trocr_lines_bundle.tgz:
  lines/<stem>/line_NNN.png + lines/_distill_meta.json
  trocr/<stem>.txt (рядок на seg-рядок, порожній для зфейлених/пропущених)
+ trocr_summary.json

⚠ БЕЗ `from __future__ import annotations` — код інжектиться після PARAMS.
"""
import dataclasses
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
OUT = Path("/tmp/trocr_lines")

MIN_W, MIN_H = 32, 10  # менше — сміттєвий кроп, OCR не ганяємо


def _ensure_deps() -> None:
    """Lightning не ставить requirements() — бутстрапимось самі (Kaggle: no-op)."""
    need = []
    try:
        import kraken  # noqa: F401
    except Exception:
        need.append("kraken>=5.0")
    try:
        import transformers  # noqa: F401
    except Exception:
        need.append("transformers>=4.45")
    if need:
        print(f"[trocr] pip install {need}…", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need],
                       check=True)


def _find_images() -> list:
    seen = {}
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        for p in KAGGLE_INPUT.rglob(pat):
            seen.setdefault(p.name.lower(), p)
    return [seen[k] for k in sorted(seen)]


def main(params: dict[str, Any]) -> None:
    _ensure_deps()
    import torch
    from kraken import blla
    from kraken.kraken import SEGMENTATION_DEFAULT_MODEL
    from kraken.lib import vgsl
    from kraken.lib.segmentation import extract_polygons
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    images = _find_images()
    pages = {s.strip() for s in str(params.get("pages", "")).split(",") if s.strip()}
    if pages:
        images = [p for p in images if p.stem in pages]
    if not images:
        raise RuntimeError(f"no images under {KAGGLE_INPUT} (pages={sorted(pages)})")

    model_id = params["model"]
    batch = int(params["batch"])
    max_new = int(params["max_new_tokens"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"[trocr] {len(images)} pages, model={model_id}, dev={device}, "
          f"batch={batch}", flush=True)

    t0 = time.time()
    seg_model = vgsl.TorchVGSLModel.load_model(SEGMENTATION_DEFAULT_MODEL)
    # Kansallisarkisto тримає processor у підтеці processor/, не в корені репо
    proc = None
    for kwargs in ({}, {"subfolder": "processor"}):
        try:
            proc = TrOCRProcessor.from_pretrained(model_id, **kwargs)
            break
        except Exception as exc:
            print(f"[trocr] processor {kwargs or 'root'}: "
                  f"{type(exc).__name__}", flush=True)
    if proc is None:
        fallback = "microsoft/trocr-large-handwritten"
        print(f"[trocr] processor fallback → {fallback}", flush=True)
        proc = TrOCRProcessor.from_pretrained(fallback)
    ocr = VisionEncoderDecoderModel.from_pretrained(model_id).to(device).eval()
    cold_start = round(time.time() - t0, 1)
    print(f"[trocr] models loaded in {cold_start}s", flush=True)

    def ocr_batch(crops: list) -> list:
        texts = []
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            pix = proc(images=chunk, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                if use_fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        ids = ocr.generate(pix, max_new_tokens=max_new)
                else:
                    ids = ocr.generate(pix, max_new_tokens=max_new)
            texts.extend(proc.batch_decode(ids, skip_special_tokens=True))
        return texts

    (OUT / "lines").mkdir(parents=True, exist_ok=True)
    (OUT / "trocr").mkdir(parents=True, exist_ok=True)
    meta = {"version": 1, "model": model_id, "pages": {}}

    t_run = time.time()
    for pi, src in enumerate(images, 1):
        t1 = time.time()
        stem = src.stem
        try:
            im = Image.open(src).convert("RGB")
            seg = blla.segment(im, model=seg_model, device=device)
            page_dir = OUT / "lines" / stem
            page_dir.mkdir(exist_ok=True)
            widths, crops, idxs = [], [], []
            for i, ln in enumerate(seg.lines):
                try:
                    one = dataclasses.replace(seg, lines=[ln])
                    line_im, _ = next(extract_polygons(im, one))
                    line_im.save(page_dir / f"line_{i:03d}.png")
                    widths.append(line_im.width)
                    if line_im.width >= MIN_W and line_im.height >= MIN_H:
                        crops.append(line_im)
                        idxs.append(i)
                except Exception:
                    widths.append(0)
            t_seg = time.time() - t1
            texts = [""] * len(seg.lines)
            for i, txt in zip(idxs, ocr_batch(crops)):
                texts[i] = txt
            (OUT / "trocr" / f"{stem}.txt").write_text(
                "\n".join(texts), encoding="utf-8")
            info = {"n_lines": len(idxs), "widths": widths,
                    "sec_seg": round(t_seg, 1),
                    "sec_ocr": round(time.time() - t1 - t_seg, 1)}
        except Exception as exc:
            info = {"n_lines": -1, "error": f"{type(exc).__name__}: {exc}"}
        meta["pages"][stem] = info
        print(f"[trocr] {pi}/{len(images)} {stem}: {info.get('n_lines')} lines, "
              f"seg {info.get('sec_seg', '?')}s + ocr {info.get('sec_ocr', '?')}s"
              f"{' ERR ' + info['error'] if info.get('error') else ''}",
              flush=True)

    (OUT / "lines" / "_distill_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
    bundle = KAGGLE_WORKING / "trocr_lines_bundle.tgz"
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(OUT / "lines", arcname="lines")
        tf.add(OUT / "trocr", arcname="trocr")
    summary = {"n_pages": len(images), "model": model_id, "device": device,
               "cold_start_sec": cold_start,
               "wall_sec": round(time.time() - t_run, 1),
               "finished": _utc_iso()}
    (KAGGLE_WORKING / "trocr_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[trocr] done: {summary}", flush=True)


if __name__ == "__main__":
    main({"model": "Kansallisarkisto/cyrillic-htr-model", "batch": 24,
          "max_new_tokens": 96, "pages": ""})
