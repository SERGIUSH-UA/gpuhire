"""PaddleOCR PP-OCRv5 page-level бенчмарк (remote Kaggle execution).

Окремий від htr_page_bench через несумісний стек (paddlepaddle-gpu з CN index).
Контракт той самий: вхід /kaggle/input/<slug>/ (images + ground_truth.json + targets.json),
вихід /kaggle/working/ — results_paddle.tsv + results_paddle.json.

Page-level: PP-OCRv5 розпізнає рядки повної сторінки → join тексту → fuzzy-max по
цільових формах прізвища → pred/score. t_infer per-page (warm), cold_start окремо.
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
    from rapidfuzz import fuzz
    t = _norm(text)
    if not t:
        return 0.0
    return round(max((float(fuzz.partial_ratio(_norm(tg), t)) for tg in targets), default=0.0), 1)


def _ocr_text(ocr, path: Path) -> str:
    result = ocr.predict(str(path))
    parts: list[str] = []
    for res in result:
        parts.extend(res.get("rec_texts", []) or [])
    return " ".join(parts)


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
    lang = str(params.get("lang", "ru"))
    print(f"[info] {len(paths)} pages, lang={lang}, targets={targets}", flush=True)

    from paddleocr import PaddleOCR
    t_cold = time.monotonic()
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, lang=lang)
    if paths:  # warm-up
        _ocr_text(ocr, paths[0])
    cold_start_s = round(time.monotonic() - t_cold, 2)
    print(f"[info] cold_start={cold_start_s}s", flush=True)

    rows = [["engine", "filename", "label", "pred", "score", "t_infer_ms", "raw_text"]]
    per_page = []
    for p in paths:
        t_p = time.monotonic()
        try:
            text = _ocr_text(ocr, p)
        except Exception as exc:
            text = ""
            print(f"  ! {p.name}: {type(exc).__name__}: {exc}", flush=True)
        t_infer_ms = round((time.monotonic() - t_p) * 1000, 1)
        sc = _score(text, targets)
        pred = 1 if sc >= threshold else 0
        label = gt[p.name]["label"]
        rows.append(["paddleocr", p.name, label, str(pred), f"{sc}",
                     str(t_infer_ms), text.replace("\t", " ").replace("\n", " ")[:300]])
        per_page.append({"engine": "paddleocr", "filename": p.name, "label": label,
                         "pred": pred, "score": sc, "t_infer_ms": t_infer_ms})
        print(f"  {p.name}: score={sc} pred={pred} {t_infer_ms}ms", flush=True)

    with (KAGGLE_WORKING / "results_paddle.tsv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows(rows)
    (KAGGLE_WORKING / "results_paddle.json").write_text(json.dumps({
        "engine": "paddleocr", "cold_start_s": cold_start_s, "threshold": threshold,
        "targets": targets, "lang": lang, "n_pages": len(paths),
        "started_at": _utc_iso(), "per_page": per_page,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[info] done", flush=True)
    return {"engine": "paddleocr", "cold_start_s": cold_start_s, "n_pages": len(paths)}


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        main(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    else:
        main(json.loads(os.environ.get("GPURUNNER_PARAMS", "{}")))
