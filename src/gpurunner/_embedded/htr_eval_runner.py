"""HTR zero-shot benchmark loop (remote Kaggle execution).

Самодостатній: stdlib + пакети з HTREvalJob.requirements(). Контракти:
вхід /kaggle/input/<slug>/ (images + ground_truth.json), вихід /kaggle/working/.
Одна модель, що впала, лишає запис в errors і не валить job.
"""

from __future__ import annotations

import csv
import glob
import json
import time
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def _match(text: str, target: str) -> float:
    """partial_ratio з гардом на короткі рядки (інакше 'до' → 100)."""
    from rapidfuzz import fuzz
    t = _norm(text)
    if not t:
        return 0.0
    score = float(fuzz.partial_ratio(t, target))
    if len(t) < len(target) * 0.6:
        score *= len(t) / (len(target) * 0.6)
    return round(score, 1)


# ── model handlers ────────────────────────────────────────────────────────────

def _run_trocr(model_id: str, paths: list[Path]) -> list[str]:
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device).eval()
    out = []
    with torch.no_grad():
        for p in paths:
            pix = proc(images=Image.open(p).convert("RGB"), return_tensors="pt").pixel_values.to(device)
            ids = model.generate(pix, max_new_tokens=32)
            out.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
    del model
    torch.cuda.empty_cache()
    return out


def _run_qwen_vl(model_id: str, paths: list[Path]) -> list[str]:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto").eval()
    prompt = ("Це вирізка одного рукописного слова з церковної книги XIX століття "
              "(кирилиця, дореформена орфографія). Напиши ЛИШЕ це слово, без пояснень.")
    out = []
    with torch.no_grad():
        for p in paths:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(p)},
                {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(messages)
            inputs = proc(text=[text], images=imgs, videos=vids,
                          padding=True, return_tensors="pt").to(model.device)
            ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            ids = ids[:, inputs.input_ids.shape[1]:]
            out.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
    del model
    torch.cuda.empty_cache()
    return out


def _run_surya(paths: list[Path]) -> list[str]:
    """Surya recognition; API мігрує між версіями — пробуємо відомі варіанти."""
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in paths]
    try:  # >=0.14: foundation predictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        rec = RecognitionPredictor(FoundationPredictor())
        preds = rec(images)
    except Exception:
        try:  # 0.6-0.13: detection+recognition predictors
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            rec = RecognitionPredictor()
            preds = rec(images, det_predictor=DetectionPredictor())
        except Exception:  # legacy run_ocr
            from surya.model.detection.model import load_model as load_det
            from surya.model.detection.model import load_processor as load_det_proc
            from surya.model.recognition.model import load_model as load_rec
            from surya.model.recognition.processor import load_processor as load_rec_proc
            from surya.ocr import run_ocr
            preds = run_ocr(images, [["ru", "uk"]] * len(images),
                            load_det(), load_det_proc(), load_rec(), load_rec_proc())
    out = []
    for pr in preds:
        lines = getattr(pr, "text_lines", None) or []
        out.append(" ".join(getattr(ln, "text", "") for ln in lines))
    return out


def main(params: dict[str, Any]) -> dict[str, Any]:
    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
    gt_file = str(params["ground_truth_file"])
    root = _dataset_root(str(params["dataset"]), gt_file)
    gt = json.loads((root / gt_file).read_text(encoding="utf-8"))
    paths = sorted(Path(p) for p in glob.glob(str(root / params["image_glob"])))
    paths = [p for p in paths if p.name in gt]
    target = str(params["target"])
    print(f"[info] {len(paths)} images, target={target!r}", flush=True)

    rows = [["model", "filename", "label", "era", "gt_text", "prediction", "match"]]
    summary: dict[str, Any] = {}
    errors: dict[str, str] = {}
    started = _utc_iso()
    t0 = time.monotonic()

    for model_id in params["models"]:
        print(f"\n=== {model_id} ===", flush=True)
        t_m = time.monotonic()
        try:
            if model_id.lower() == "surya":
                preds = _run_surya(paths)
            elif "qwen" in model_id.lower():
                preds = _run_qwen_vl(model_id, paths)
            else:
                preds = _run_trocr(model_id, paths)
        except Exception as exc:
            print(f"  ! FAILED: {type(exc).__name__}: {exc}", flush=True)
            errors[model_id] = f"{type(exc).__name__}: {exc}"
            continue

        scored = []
        for p, pred in zip(paths, preds):
            meta = gt[p.name]
            m = _match(pred, target)
            scored.append({"label": meta["label"], "era": meta["era"], "match": m})
            rows.append([model_id, p.name, meta["label"], meta["era"],
                         meta.get("text", ""), pred, str(m)])

        def auc(items: list[dict]) -> float | None:
            tp = [r["match"] for r in items if r["label"] == "tp"]
            fp = [r["match"] for r in items if r["label"] == "fp"]
            if not tp or not fp:
                return None
            wins = sum(1 for a in tp for b in fp if a > b) + 0.5 * sum(
                1 for a in tp for b in fp if a == b)
            return round(wins / (len(tp) * len(fp)), 3)

        def med(items: list[dict], label: str) -> float:
            xs = sorted(r["match"] for r in items if r["label"] == label)
            return xs[len(xs) // 2] if xs else 0.0

        eras = sorted({r["era"] for r in scored})
        per_era = {e: {"auc": auc([r for r in scored if r["era"] == e]),
                       "tp_med": med([r for r in scored if r["era"] == e], "tp"),
                       "fp_med": med([r for r in scored if r["era"] == e], "fp")}
                   for e in eras}
        summary[model_id] = {"auc_overall": auc(scored), "tp_median": med(scored, "tp"),
                             "fp_median": med(scored, "fp"), "per_era": per_era,
                             "elapsed_s": round(time.monotonic() - t_m, 1)}
        print(f"  AUC={summary[model_id]['auc_overall']} "
              f"tp_med={summary[model_id]['tp_median']} fp_med={summary[model_id]['fp_median']}",
              flush=True)

    (KAGGLE_WORKING / "results.json").write_text(
        json.dumps({"summary": summary, "errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with (KAGGLE_WORKING / "results.tsv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows(rows)
    (KAGGLE_WORKING / "inference_log.json").write_text(json.dumps({
        "started_at": started, "finished_at": _utc_iso(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "models": params["models"], "n_images": len(paths), "errors": errors,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[info] done", flush=True)
    return summary


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        main(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    else:
        main(json.loads(os.environ.get("GPURUNNER_PARAMS", "{}")))
