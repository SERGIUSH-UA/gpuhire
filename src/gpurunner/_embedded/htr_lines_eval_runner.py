"""HTR line-eval: N моделей (TrOCR / PARSeq) по ГОТОВИХ кропах рядків (GPU).

Кейс: порівняння кандидатів-вчителів для кириличного kraken на ОДНИХ
і тих САМИХ кропах (сегментація вже зроблена job'ом trocr_lines) — щоб
відсортувати, кого тюнити.

Вхід  /kaggle/input/** — lines.tgz АБО розпаковані lines/<stem>/line_NNN.png.
Вихід /kaggle/working/htr_eval_bundle.tgz:
  <model_slug>/<stem>.txt (рядок на seg-індекс, діри = порожні рядки)
+ htr_eval_summary.json (per-model cold_start / wall / n_fail)

Формат params["models"]: кома-список "engine:repo_id[@hNNN]",
  engine ∈ {trocr, parseq}; @hNNN = пре-ресайз кропа до висоти NNN px
  (aspect preserved) перед процесором — мімікрія тренування (укр. TrOCR).

⚠ БЕЗ `from __future__ import annotations` — код інжектиться після PARAMS.
"""
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
OUT = Path("/tmp/htr_eval")

MIN_W, MIN_H = 32, 10  # синхронно з trocr_lines: менше — не ганяємо OCR


def _pip(*pkgs) -> None:
    print(f"[heval] pip install {pkgs}…", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                   check=True)


def _ensure_deps() -> None:
    try:
        import transformers  # noqa: F401
    except Exception:
        _pip("transformers>=4.45")


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", model_id).strip("_")


def _find_lines_root() -> Path:
    """lines.tgz → розпакувати; інакше шукаємо теку lines/ або <stem>/line_*.png."""
    tgzs = sorted(KAGGLE_INPUT.rglob("*.tgz"))
    if tgzs:
        dest = Path("/tmp/heval_input")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgzs[0]) as tf:
            tf.extractall(dest)
        print(f"[heval] extracted {tgzs[0].name} -> {dest}", flush=True)
        cand = dest / "lines"
        return cand if cand.is_dir() else dest
    for d in KAGGLE_INPUT.rglob("lines"):
        if d.is_dir():
            return d
    return KAGGLE_INPUT


def _load_pages(root: Path) -> dict:
    """{stem: [(line_idx, Path), ...]} — сортовано за індексом."""
    pages = {}
    for page_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        items = []
        for f in sorted(page_dir.glob("line_*.png")):
            m = re.match(r"line_(\d+)$", f.stem)
            if m:
                items.append((int(m.group(1)), f))
        if items:
            pages[page_dir.name] = items
    return pages


# ── engines ──────────────────────────────────────────────────────────────────

def _load_trocr(repo: str):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = None
    fallbacks = ({"pretrained_model_name_or_path": repo},
                 {"pretrained_model_name_or_path": repo, "subfolder": "processor"},
                 {"pretrained_model_name_or_path": "kazars24/trocr-base-handwritten-ru"},
                 {"pretrained_model_name_or_path": "microsoft/trocr-base-handwritten"})
    for kw in fallbacks:
        try:
            proc = TrOCRProcessor.from_pretrained(**kw)
            print(f"[heval] processor <- {kw}", flush=True)
            break
        except Exception as exc:
            print(f"[heval] processor {kw}: {type(exc).__name__}", flush=True)
    if proc is None:
        raise RuntimeError(f"no processor for {repo}")
    model = VisionEncoderDecoderModel.from_pretrained(repo).to(device).eval()
    return proc, model, device


def _run_trocr(state, crops, batch: int, max_new: int, pre_h: int) -> list:
    import torch
    proc, model, device = state
    if pre_h:
        crops = [c.resize((max(8, round(c.width * pre_h / c.height)), pre_h))
                 for c in crops]
    texts = []
    for i in range(0, len(crops), batch):
        pix = proc(images=crops[i:i + batch],
                   return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    ids = model.generate(pix, max_new_tokens=max_new)
            else:
                ids = model.generate(pix, max_new_tokens=max_new)
        texts.extend(proc.batch_decode(ids, skip_special_tokens=True))
    return texts


def _load_parseq(repo: str):
    """best.pt (model_state+charset+config) + код baudm/parseq (strhub)."""
    import torch
    try:
        import strhub  # noqa: F401
    except Exception:
        _pip("git+https://github.com/baudm/parseq.git", "pytorch-lightning",
             "timm", "nltk")
    from strhub.models.parseq.system import PARSeq

    if repo == "local" or repo.startswith("local:"):
        # власний чекпойнт (job parseq_train) — лежить серед вхідних датасетів
        want = repo.partition(":")[2]
        cands = sorted(KAGGLE_INPUT.rglob("*.pt"))
        if want:
            cands = [p for p in cands if want in p.name] or cands
        if not cands:
            raise RuntimeError(f"parseq:{repo} — *.pt не знайдено під {KAGGLE_INPUT}")
        ckpt = str(cands[0])
        print(f"[heval] parseq <- local {ckpt}", flush=True)
    else:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo, "best.pt")
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    charset = payload["charset"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = dict(
        charset_train=charset, charset_test=charset,
        max_label_length=int(cfg.get("max_label_length", 100)),
        batch_size=1, lr=1e-4, warmup_pct=0.1, weight_decay=0.0,
        img_size=[int(cfg.get("img_height", 48)), int(cfg.get("img_width", 512))],
        patch_size=cfg.get("patch_size", [8, 8]),
        embed_dim=int(cfg.get("embed_dim", 384)),
        enc_num_heads=int(cfg.get("enc_num_heads", 6)),
        enc_mlp_ratio=int(cfg.get("enc_mlp_ratio", 4)),
        enc_depth=int(cfg.get("enc_depth", 12)),
        dec_num_heads=int(cfg.get("dec_num_heads", 12)),
        dec_mlp_ratio=int(cfg.get("dec_mlp_ratio", 4)),
        dec_depth=int(cfg.get("dec_depth", 1)),
        perm_num=6, perm_forward=True, perm_mirrored=True,
        decode_ar=bool(cfg.get("decode_ar", True)),
        refine_iters=int(cfg.get("refine_iters", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    ps = cfg.get("patch_size")
    if isinstance(ps, int):
        kwargs["patch_size"] = [ps, ps]
    model = PARSeq(**kwargs)
    sd = payload["model_state"]
    # ключі чекпойнта можуть бути без/з префіксом "model." — беремо варіант
    # з найбільшим перетином із state_dict модуля (strict=False не кидає)
    model_keys = set(model.state_dict().keys())
    variants = (sd,
                {"model." + k: v for k, v in sd.items()},
                {k[len("model."):]: v for k, v in sd.items()
                 if k.startswith("model.")})
    sd = max(variants, key=lambda v: len(set(v) & model_keys))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[heval] parseq load: matched={len(set(sd) & model_keys)} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if len(missing) > len(model_keys) // 2:
        raise RuntimeError(f"parseq state_dict mismatch: {missing[:5]}")
    model = model.to(device).eval()
    h, w = kwargs["img_size"]
    return model, device, (h, w)


def _run_parseq(state, crops, batch: int, _max_new: int, _pre_h: int) -> list:
    import torch
    model, device, (h, w) = state
    from PIL import Image
    tensors = []
    for c in crops:
        im = c.convert("RGB").resize((w, h), Image.LANCZOS)
        import numpy as np
        arr = (np.asarray(im, dtype="float32") / 255.0 - 0.5) / 0.5
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    texts = []
    for i in range(0, len(tensors), batch):
        x = torch.stack(tensors[i:i + batch]).to(device)
        with torch.no_grad():
            probs = model(x).softmax(-1)
        preds, _ = model.tokenizer.decode(probs)
        texts.extend(preds)
    return texts


ENGINES = {"trocr": (_load_trocr, _run_trocr, 24),
           "parseq": (_load_parseq, _run_parseq, 64)}


def main(params: dict[str, Any]) -> None:
    _ensure_deps()
    from PIL import Image

    root = _find_lines_root()
    pages = _load_pages(root)
    if not pages:
        raise RuntimeError(f"no lines/<stem>/line_NNN.png under {root}")
    n_crops = sum(len(v) for v in pages.values())
    print(f"[heval] {len(pages)} pages, {n_crops} line crops", flush=True)

    specs = [s.strip() for s in str(params["models"]).split(",") if s.strip()]
    max_new = int(params["max_new_tokens"])
    summary = {"n_pages": len(pages), "n_crops": n_crops, "models": {}}

    for spec in specs:
        engine, _, rest = spec.partition(":")
        repo, _, hs = rest.partition("@")
        pre_h = int(hs[1:]) if hs.startswith("h") else 0
        slug = _slug(repo)
        load_fn, run_fn, batch = ENGINES[engine]
        print(f"[heval] === {spec} -> {slug} ===", flush=True)
        t0 = time.time()
        try:
            state = load_fn(repo)
        except Exception as exc:
            print(f"[heval] LOAD FAIL {spec}: {type(exc).__name__}: {exc}",
                  flush=True)
            summary["models"][slug] = {"spec": spec,
                                       "error": f"load: {exc}"}
            continue
        cold = round(time.time() - t0, 1)
        out_dir = OUT / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        n_fail = 0
        for pi, (stem, items) in enumerate(pages.items(), 1):
            idxs, crops = [], []
            max_idx = items[-1][0]
            for idx, f in items:
                im = Image.open(f).convert("RGB")
                if im.width >= MIN_W and im.height >= MIN_H:
                    idxs.append(idx)
                    crops.append(im)
            texts = [""] * (max_idx + 1)
            try:
                for idx, txt in zip(idxs, run_fn(state, crops, batch,
                                                 max_new, pre_h)):
                    texts[idx] = txt.replace("\n", " ")
            except Exception as exc:
                n_fail += 1
                print(f"[heval] {slug} {stem} FAIL: "
                      f"{type(exc).__name__}: {exc}", flush=True)
            (out_dir / f"{stem}.txt").write_text("\n".join(texts),
                                                 encoding="utf-8")
            print(f"[heval] {slug} {pi}/{len(pages)} {stem}: "
                  f"{len(crops)} lines", flush=True)
        summary["models"][slug] = {"spec": spec, "cold_start_sec": cold,
                                   "wall_sec": round(time.time() - t1, 1),
                                   "n_pages_failed": n_fail}
        del state  # звільнити VRAM перед наступною моделлю
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
    bundle = KAGGLE_WORKING / "htr_eval_bundle.tgz"
    with tarfile.open(bundle, "w:gz") as tf:
        for d in sorted(OUT.iterdir()):
            if d.is_dir():
                tf.add(d, arcname=d.name)
    summary["finished"] = _utc_iso()
    (KAGGLE_WORKING / "htr_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[heval] done: {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main({"models": "trocr:cyrillic-trocr/trocr-ukrainian-handwritten@h128",
          "max_new_tokens": 96})
