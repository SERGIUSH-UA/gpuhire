"""A/B двох наборів моделей на ОДНІЙ машині: швидкість по етапах і тексти для якості.

Вхід (`/kaggle/input`, на Vast — залиті теки):
  frames/*.tar             — кадри справ архівом (тека першого рівня = справа);
  models/                   — ваги: pysar_cyr_v17.pt, pysar_cyr_v18.pt,
                              diak_cyr_v4.mlmodel, skryba_f792_v6.mlmodel,
                              <літописець>.safetensors.
Проходи, послідовно на тій самій карті:
  A  — бойове читання як було: Писар v17 + Дяк v4 (другий голос) + Скриба
       (`--with latin`), сегментація рахується вперше;
  B  — Писар v18 + Скриба на ТІЙ САМІЙ сегментації (кеш справи), без Дяка;
  L  — Літописець (kraken ≥ 7.1, окреме середовище) читає ті самі рядки з кешу
       сегментації пачками, як другий голос.
Вихід у `/kaggle/working`: `ab_bundle.tgz` (A/, B/, L/ — тексти й мети прогонів),
`ab_timing.json` (час кожного кроку), `ab.log`.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

IN = Path("/kaggle/input")
OUT = Path("/kaggle/working")
NYSH_VENV = Path("/opt/nysh")
K71 = Path("/opt/k71")
WS = Path("/opt/ws")
TIMING: dict[str, Any] = {}


def log(msg: str) -> None:
    line = f"[ab {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "ab.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sh(cmd: list[str], env: dict | None = None, label: str = "") -> int:
    t = time.time()
    log(f"$ {' '.join(map(str, cmd))[:400]}")
    with open(OUT / "ab.log", "a", encoding="utf-8") as fh:
        rc = subprocess.run([str(c) for c in cmd], env={**os.environ, **(env or {})},
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    dt = round(time.time() - t, 1)
    if label:
        TIMING[label] = dt
    log(f"  rc={rc} за {dt}s")
    return rc


def _find(name: str) -> Path:
    hits = sorted(IN.rglob(name))
    if not hits:
        raise RuntimeError(f"немає {name} під {IN}")
    return hits[0]


def setup(params: dict) -> dict:
    env = {"NYSHPORKA_WORKSPACE": str(WS)}
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"], label="setup_uv")
    sh(["uv", "venv", str(NYSH_VENV), "--python", "3.11"])
    if sh(["uv", "pip", "install", "--python", str(NYSH_VENV / "bin/python"),
           f"nyshporka[app]=={params['nysh_version']}"], label="setup_nysh"):
        raise RuntimeError("nyshporka не встановився")
    nysh = str(NYSH_VENV / "bin/nysh")
    WS.mkdir(parents=True, exist_ok=True)
    sh([nysh, "init", str(WS), "--yes", "--preset", "researcher"], env=env)
    if sh([nysh, "htr", "install"], env=env, label="setup_engines"):
        raise RuntimeError("nysh htr install не пройшов")
    models = WS / "data" / "spotter" / "models"
    models.mkdir(parents=True, exist_ok=True)
    for f in ("pysar_cyr_v17.pt", "pysar_cyr_v18.pt", "diak_cyr_v4.mlmodel",
              "skryba_f792_v6.mlmodel"):
        shutil.copy(_find(f), models / f)
    (models / "PRODUCTION.json").write_text(json.dumps({"production": {
        "cyrillic": {"model": "pysar_cyr_v17.pt"},
        "latin": {"model": "skryba_f792_v6.mlmodel"}}}), encoding="utf-8")
    # Літописець: kraken 7.1 у власному середовищі; torch під cu126 — V100 (sm_70)
    # у колесах cu128 уже немає
    sh(["uv", "venv", str(K71), "--python", "3.11"])
    sh(["uv", "pip", "install", "--python", str(K71 / "bin/python"), "kraken==7.1.1"],
       label="setup_k71")
    sh(["uv", "pip", "install", "--python", str(K71 / "bin/python"), "--reinstall",
        "torch==2.14.0", "torchvision==0.29.1", "--index-url",
        "https://download.pytorch.org/whl/cu126"], label="setup_k71_torch")
    sh(["nvidia-smi"])
    return env


LIT_SCRIPT = r'''
import gzip, json, os, sys, time
from pathlib import Path
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
def main():
    # Літописець читає ТІ САМІ кропи, що й Писар: рамка й полігон кожного рядка
    # з `<сторінка>.lines.json` прогону Писаря, кропи складаються в полотно з
    # проміжками й читаються пачкою (так у конвеєрі читав би другий голос).
    # Вирізання за базовою лінією з цілого аркуша (перша редакція) коштувало
    # ~12 с/стор процесора — це ціна вирізання, а не моделі.
    model_p, run_dir, frames, out, batch = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
    from PIL import Image, ImageDraw
    import torch
    from kraken.configs import RecognitionInferenceConfig
    from kraken.containers import BBoxLine, Segmentation
    from kraken.tasks import RecognitionTaskModel
    t0 = time.time(); m = RecognitionTaskModel.load_model(model_p); load = time.time() - t0
    cfg = RecognitionInferenceConfig(accelerator="gpu", device=[0], batch_size=batch)
    out.mkdir(parents=True, exist_ok=True)
    per = {}; tot_lines = 0; t_crop = 0.0; t_rec = 0.0; t1 = time.time()
    for lj in sorted(run_dir.glob("*.lines.json")):
        stem = lj.name[:-len(".lines.json")]
        cand = [p for p in frames.iterdir() if p.stem == stem]
        if not cand: continue
        tc = time.time()
        d = json.load(open(lj, encoding="utf-8"))
        page = Image.open(cand[0]).convert("L")
        if list(page.size) != list(d.get("size", page.size)):
            page = page.resize(tuple(d["size"]))
        crops = []
        for (x0, y0, x1, y1), poly in zip(d["boxes"], d["polys"]):
            c = page.crop((x0, y0, x1, y1))
            if poly:
                mask = Image.new("L", c.size, 0)
                ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in poly], fill=255)
                bg = Image.new("L", c.size, 255); bg.paste(c, (0, 0), mask); c = bg
            crops.append(c)
        gap = 64
        if not crops:
            (out / f"{stem}.txt").write_text(chr(10), encoding="utf-8"); continue
        w = max(c.width for c in crops)
        cv = Image.new("L", (w + 2 * gap, sum(c.height + gap for c in crops) + gap), 255)
        boxes, y = [], gap
        for k, c in enumerate(crops):
            cv.paste(c, (gap, y)); boxes.append(BBoxLine(id=f"l{k}", bbox=(gap, y, gap + c.width, y + c.height))); y += c.height + gap
        seg = Segmentation(type="bbox", imagename=stem, text_direction="horizontal-lr", script_detection=False, lines=boxes)
        t_crop += time.time() - tc
        torch.cuda.synchronize(); tp = time.time()
        txt = [r.prediction for r in m.predict(im=cv, segmentation=seg, config=cfg)]
        torch.cuda.synchronize(); dt = time.time() - tp; t_rec += dt
        if len(txt) != len(crops): txt = [""] * len(crops)
        per[stem] = {"sec": round(dt, 3), "lines": len(txt)}
        tot_lines += len(txt)
        (out / f"{stem}.txt").write_text(chr(10).join(t.strip() for t in txt) + chr(10), encoding="utf-8")
    json.dump({"load_sec": round(load, 2), "crop_sec": round(t_crop, 2), "rec_sec": round(t_rec, 2),
               "read_sec": round(time.time() - t1, 2), "lines": tot_lines, "pages": per, "batch": batch},
              open(out / "_lit_timing.json", "w"), indent=1)
# Захист головного модуля без буквального рядка, яким рендер job'а ріже раннер:
# kraken пускає воркерів через multiprocessing, і без захисту кожен з них
# заново запускав би читання.
if globals().get("__name__") == "__main__":
    main()
'''


def main(params: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    env = setup(params)
    nysh = str(NYSH_VENV / "bin/nysh")
    models = WS / "data" / "spotter" / "models"
    lit = _find(params["litopysets"])
    (WS / "lit_read.py").write_text(LIT_SCRIPT, encoding="utf-8")
    # Заливка Vast бере лише файли верхнього рівня теки, тож кадри приходять
    # архівом (frames/*.tar) — справа = тека першого рівня в архіві.
    frames_root = Path("/opt/frames")
    frames_root.mkdir(parents=True, exist_ok=True)
    for arc in sorted(IN.rglob("*.tar")) + sorted(IN.rglob("*.tgz")):
        t = time.time()
        with tarfile.open(arc, "r:*") as tf:
            tf.extractall(frames_root)
        log(f"розпаковано {arc.name} за {time.time() - t:.0f}s")
    cases = sorted(p for p in frames_root.iterdir() if p.is_dir())
    log(f"справи: {[c.name for c in cases]}")
    passes = str(params.get("passes", "A,B,L")).split(",")
    want = [c for c in str(params.get("cases", "")).split(",") if c]
    if want:
        cases = [c for c in cases if c.name in want]
    for case in cases:
        name = case.name
        if "A" in passes:
            rc = sh([nysh, "read", str(case), "--with", "latin", "--case-key", f"AB/{name}",
                     "--out", str(OUT / "A" / name), "--force"], env=env, label=f"A:{name}")
            if rc and case == cases[0]:
                # перший же прохід упав — далі те саме на кожній справі, а оренда йде
                raise RuntimeError(f"прохід A на {name} rc={rc} — див. ab.log")
        if "B" in passes:
            # бойовим Писарем стає v18 — прохід рахує сегментацію сам, якщо A не було
            (models / "PRODUCTION.json").write_text(json.dumps({"production": {
                "cyrillic": {"model": "pysar_cyr_v18.pt"},
                "latin": {"model": "skryba_f792_v6.mlmodel"}}}), encoding="utf-8")
            rc = sh([nysh, "read", str(case), "--with", "latin", "--one-voice", "--case-key", f"AB/{name}",
                     "--out", str(OUT / "B" / name), "--force", "--rerun"], env=env, label=f"B:{name}")
            if rc and case == cases[0]:
                raise RuntimeError(f"прохід B на {name} rc={rc} — див. ab.log")
        if "L" in passes:
            run = OUT / ("B" if "B" in passes else "A") / name
            sh([str(K71 / "bin/python"), str(WS / "lit_read.py"), str(lit), str(run), str(case),
                str(OUT / "L" / name), str(params["lit_batch"])], label=f"L:{name}")
    TIMING["total"] = round(time.time() - t_all, 1)
    (OUT / "ab_timing.json").write_text(json.dumps(TIMING, ensure_ascii=False, indent=1), encoding="utf-8")
    # усі теки виводу: `nysh read --model` може покласти прогін поруч під тегом моделі
    with tarfile.open(OUT / "ab_bundle.tgz", "w:gz") as tf:
        for p in sorted(OUT.iterdir()):
            if p.is_dir():
                tf.add(p, arcname=p.name)
    # Забирати треба ОДИН архів: пофайловий забір тисяч файлів обірвався на 316-му
    # (10-04), і тексти двох проходів зникли разом із машиною.
    for p in list(OUT.iterdir()):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    log(f"готово за {TIMING['total']}s")


if __name__ == "__main__":
    main({"nysh_version": "0.23.2", "litopysets": "litopysets_cyr_v1.safetensors", "lit_batch": 8})
