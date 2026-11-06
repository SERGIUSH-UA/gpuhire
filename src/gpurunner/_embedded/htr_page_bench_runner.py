"""HTR page-level бенчмарк — TrOCR + Surya (remote Kaggle execution).

Самодостатній: stdlib + transformers/surya/pillow/rapidfuzz. Контракт:
вхід  /kaggle/input/<slug>/ — images/*.png + ground_truth.json + targets.json
вихід /kaggle/working/      — results_htr.tsv + results_htr.json

Page-level: движок транскрибує ПОВНУ сторінку → текст; fuzzy-max по цільових
формах прізвища → score (0..100); pred = score >= threshold. t_infer міряється
per-page (warm), cold_start (load моделі) — окремо per-движок.

TrOCR — word/line-level: на повній сторінці без сегментації дає 1 рядок (baseline,
показує межу методу). Surya — з вбудованою детекцією рядків (справжній page-HTR).
Падіння одного движка не валить job.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def _score(text: str, targets: list[str]) -> float:
    """Max partial_ratio по всіх цільових формах (текст сторінки → є прізвище?)."""
    from rapidfuzz import fuzz
    t = _norm(text)
    if not t:
        return 0.0
    best = 0.0
    for tgt in targets:
        tg = _norm(tgt)
        sc = float(fuzz.partial_ratio(tg, t))  # шукаємо коротку ціль у довгому тексті
        best = max(best, sc)
    return round(best, 1)


# ── движки: load() -> stateful obj; transcribe(obj, path) -> str ────────────────

def _load_trocr(model_id: str):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device).eval()
    return ("trocr", proc, model, device)


def _trans_trocr(obj, path: Path) -> str:
    import torch
    from PIL import Image
    _, proc, model, device = obj
    pix = proc(images=Image.open(path).convert("RGB"),
               return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pix, max_new_tokens=64)
    return proc.batch_decode(ids, skip_special_tokens=True)[0]


def _load_surya(_model_id: str):
    """Surya full-page OCR. Запінено surya-ocr==0.20.0 (API: RecognitionPredictor
    + full_page=True → PageOCRResult.blocks[].html). Fallback на старий
    foundation+detection API, якщо версія інша."""
    from surya.recognition import RecognitionPredictor
    try:  # 0.20.x — простий full-page виклик, без surya.detection (він кидає на import)
        return ("surya_fp", RecognitionPredictor())
    except Exception:  # старий foundation+detection API (0.14–0.16)
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        return ("surya_legacy", RecognitionPredictor(FoundationPredictor()), DetectionPredictor())


def _strip_html(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", " ", s or "")


def _trans_surya(obj, path: Path) -> str:
    from PIL import Image
    kind, rec = obj[0], obj[1]
    img = Image.open(path).convert("RGB")
    if kind == "surya_fp":  # 0.20: full_page → blocks[].html
        page = rec([img], full_page=True)[0]
        return " ".join(_strip_html(getattr(b, "html", "")) for b in getattr(page, "blocks", []))
    det = obj[2]  # legacy: text_lines
    lines = getattr(rec([img], det_predictor=det)[0], "text_lines", None) or []
    return " ".join(getattr(ln, "text", "") for ln in lines)


def main(params: dict[str, Any]) -> dict[str, Any]:
    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
    gt_file = str(params["ground_truth_file"])
    root = _dataset_root(str(params["dataset"]), gt_file)
    gt = json.loads((root / gt_file).read_text(encoding="utf-8"))
    paths = sorted(p for p in (root / params["image_glob"].split("/")[0]).glob("*.png")
                   if p.name in gt)
    targets = params.get("targets") or []
    tfile = root / params.get("targets_file", "targets.json")
    if tfile.exists():
        targets = json.loads(tfile.read_text(encoding="utf-8")).get("targets", targets)
    threshold = float(params["threshold"])
    print(f"[info] {len(paths)} pages, targets={targets}", flush=True)

    rows = [["engine", "filename", "label", "pred", "score", "t_infer_ms", "raw_text"]]
    meta: dict[str, Any] = {}
    errors: dict[str, str] = {}
    per_page: list[dict] = []

    for model_id in params["models"]:
        is_surya = model_id.lower() == "surya"
        print(f"\n=== {model_id} ===", flush=True)
        t_cold = time.monotonic()
        try:
            obj = _load_surya(model_id) if is_surya else _load_trocr(model_id)
            # warm-up на першій сторінці (час відкидається з per-page)
            if paths:
                (_trans_surya if is_surya else _trans_trocr)(obj, paths[0])
        except Exception as exc:
            print(f"  ! LOAD FAILED: {type(exc).__name__}: {exc}", flush=True)
            errors[model_id] = f"load: {type(exc).__name__}: {exc}"
            continue
        cold_start_s = round(time.monotonic() - t_cold, 2)
        trans = _trans_surya if is_surya else _trans_trocr

        for p in paths:
            t_p = time.monotonic()
            try:
                text = trans(obj, p)
            except Exception as exc:
                text = ""
                print(f"  ! {p.name}: {type(exc).__name__}: {exc}", flush=True)
            t_infer_ms = round((time.monotonic() - t_p) * 1000, 1)
            sc = _score(text, targets)
            pred = 1 if sc >= threshold else 0
            label = gt[p.name]["label"]
            rows.append([model_id, p.name, label, str(pred), f"{sc}",
                         str(t_infer_ms), text.replace("\t", " ").replace("\n", " ")[:300]])
            per_page.append({"engine": model_id, "filename": p.name, "label": label,
                             "pred": pred, "score": sc, "t_infer_ms": t_infer_ms})
        meta[model_id] = {"cold_start_s": cold_start_s, "n_pages": len(paths)}
        print(f"  cold_start={cold_start_s}s, done {len(paths)} pages", flush=True)
        # звільнити VRAM між движками
        try:
            import torch
            del obj
            torch.cuda.empty_cache()
        except Exception:
            pass

    with (KAGGLE_WORKING / "results_htr.tsv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows(rows)
    (KAGGLE_WORKING / "results_htr.json").write_text(json.dumps({
        "engines": meta, "errors": errors, "threshold": threshold,
        "targets": targets, "started_at": _utc_iso(), "per_page": per_page,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[info] done", flush=True)
    return {"engines": list(meta), "errors": errors}


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        main(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    else:
        main(json.loads(os.environ.get("GPURUNNER_PARAMS", "{}")))
