"""Spotter page-level бенчмарк з ANCHOR-RANK (remote Kaggle execution).

Повний спотер-конвеєр роду (не голий conf): YOLO (conf 0.01, висока recall) тайлить
сторінку → top-K кандидатів за conf → DINO-backbone embedding (prep gray, 1536D=CLS+patch
mean, L2) → cosine до банку якорів → page-score = max cosine по кандидатах. Саме anchor-rank
виправляє рангування «ФП>ТП», яке є у голого YOLO conf.

Вхід  /kaggle/input/<slug>/ — images/*.png + ground_truth.json + weights/<file>.pt
                              + anchor/dino_v4/(model.safetensors,config.json)
                              + anchor/v4-backbone__gray.npz (anchor embeddings [46,1536])
Вихід /kaggle/working/      — results_spotter.tsv + results_spotter.json
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
EMB_DIM = 1536


def _tiles(w: int, h: int, tile: int, overlap: float):
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    for y in ys:
        for x in xs:
            yield max(0, x), max(0, y), min(w, x + tile), min(h, y + tile)


def _embed(backbone, crops, device, image_size, prep):
    """list[PIL] -> [N,1536] L2-norm float32 (prep gray = grayscale→RGB)."""
    import numpy as np
    import torch
    from torchvision import transforms
    if not crops:
        return np.zeros((0, EMB_DIM), dtype="float32")
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch = []
    for c in crops:
        c = c.convert("L").convert("RGB") if prep == "gray" else c.convert("RGB")
        batch.append(tf(c))
    x = torch.stack(batch).to(device)
    with torch.no_grad():
        hs = backbone(pixel_values=x).last_hidden_state          # [N,257,768]
        emb = torch.cat([hs[:, 0], hs[:, 1:].mean(dim=1)], dim=-1)  # [N,1536]
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.detach().cpu().float().numpy()


def main(params: dict[str, Any]) -> dict[str, Any]:
    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageClassification
    from ultralytics import YOLO

    gt_file = str(params["ground_truth_file"])
    root = _dataset_root(str(params["dataset"]), gt_file)
    gt = json.loads((root / gt_file).read_text(encoding="utf-8"))
    paths = sorted(p for p in (root / params["image_glob"].split("/")[0]).glob("*.png")
                   if p.name in gt)
    tile = int(params["tile"]); overlap = float(params["overlap"])
    imgsz = int(params["imgsz"]); conf = float(params["conf"])
    top_k = int(params["top_k"]); threshold = float(params["threshold"])
    prep = str(params["prep"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] {len(paths)} pages, conf={conf}, top_k={top_k}, prep={prep}", flush=True)

    # ── cold start: YOLO + DINO backbone + банк якорів + warm-up ─────────────────
    t_cold = time.monotonic()
    model = YOLO(str(root / params["weights"]))
    clf = AutoModelForImageClassification.from_pretrained(str(root / params["backbone_dir"]))
    backbone = clf.dinov2.to(device).eval()
    image_size = int(getattr(clf.config, "image_size", 518))
    npz = np.load(str(root / params["bank_emb"]), allow_pickle=False)
    anchor_embs = npz["embs"].astype("float32")               # [46,1536]
    print(f"[info] anchors={anchor_embs.shape}, image_size={image_size}", flush=True)
    _dummy = Image.new("RGB", (tile, tile), (255, 255, 255))
    model.predict(_dummy, imgsz=imgsz, conf=conf, device=0 if device.type == "cuda" else "cpu", verbose=False)
    _embed(backbone, [_dummy], device, image_size, prep)
    cold_start_s = round(time.monotonic() - t_cold, 2)
    print(f"[info] cold_start={cold_start_s}s", flush=True)

    rows = [["engine", "filename", "label", "pred", "score", "t_infer_ms", "n_cand"]]
    per_page = []
    ydev = 0 if device.type == "cuda" else "cpu"
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        t_p = time.monotonic()
        cands = []  # (conf, PIL crop)
        for (x0, y0, x1, y1) in _tiles(w, h, tile, overlap):
            tcrop = img.crop((x0, y0, x1, y1))
            res = model.predict(tcrop, imgsz=imgsz, conf=conf, device=ydev, verbose=False)
            b = res[0].boxes
            if b is None or not len(b):
                continue
            for bb, cf in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                bx0, by0, bx1, by1 = [int(v) for v in bb]
                if bx1 > bx0 and by1 > by0:
                    cands.append((float(cf), tcrop.crop((bx0, by0, bx1, by1))))
        cands.sort(key=lambda t: -t[0])
        cands = cands[:top_k]
        if cands:
            embs = _embed(backbone, [c for _, c in cands], device, image_size, prep)
            sims = embs @ anchor_embs.T            # [K,46]
            page_score = float(sims.max()) if sims.size else 0.0
        else:
            page_score = 0.0
        t_infer_ms = round((time.monotonic() - t_p) * 1000, 1)
        pred = 1 if page_score >= threshold else 0
        label = gt[p.name]["label"]
        rows.append(["spotter_anchor", p.name, label, str(pred), f"{page_score:.4f}",
                     str(t_infer_ms), str(len(cands))])
        per_page.append({"filename": p.name, "label": label, "pred": pred,
                         "score": round(page_score, 4), "t_infer_ms": t_infer_ms,
                         "n_cand": len(cands)})
        print(f"  {p.name}: anchor={page_score:.3f} pred={pred} cand={len(cands)} {t_infer_ms}ms", flush=True)

    with (KAGGLE_WORKING / "results_spotter.tsv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows(rows)
    (KAGGLE_WORKING / "results_spotter.json").write_text(json.dumps({
        "engine": "spotter_anchor",
        "weights": Path(params["weights"]).name,
        "config": {"tile": tile, "overlap": overlap, "imgsz": imgsz, "conf": conf,
                   "top_k": top_k, "threshold": threshold, "prep": prep,
                   "backbone": "v4", "n_anchors": int(anchor_embs.shape[0])},
        "cold_start_s": cold_start_s, "n_pages": len(paths),
        "started_at": _utc_iso(), "per_page": per_page,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[info] done", flush=True)
    return {"n_pages": len(paths), "cold_start_s": cold_start_s}


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        main(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    else:
        main(json.loads(os.environ.get("GPURUNNER_PARAMS", "{}")))
