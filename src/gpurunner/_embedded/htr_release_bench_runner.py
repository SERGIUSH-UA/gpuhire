"""Замір релізу: стара й нова збірка конвеєра на ОДНІЙ машині, по черзі.

Вхід (`/kaggle/input`, на Vast — залиті теки):
  frames/frames.tar   — кадри «справ» (тека першого рівня = справа; префікс
                        `lat-` — латинка, решта — кирилиця з латинкою поруч);
  models_a/, models_b/ — ваги й PRODUCTION.json кожного проходу;
  wheel/*.whl          — збірка nyshporka для проходу B (не з PyPI);
  gold/gold.tar        — голд-сет: data/train/sets/g-*/set.json + data/train/crops/g-*.
Проходи: A — nyshporka з PyPI (`nysh_a`), B — wheel. Кожен у своєму venv і своєму
робочому просторі (власний кеш сегментації, тож сегментація рахується в обох),
рушії ставить `nysh htr install` відповідної збірки. A йде ПЕРШИМ і повністю:
`htr install` збірки B оновлює середовище рушіїв на місці.
Після B — голоси Писаря на голд-сеті (`gold_models`), у середовищі B.
Вихід: `release_bundle.tgz` (A/, B/, gold/ — тексти й мети), `release_timing.json`,
`release.log`.
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
TIMING: dict[str, Any] = {}


def log(msg: str) -> None:
    line = f"[rel {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "release.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sh(cmd: list, env: dict | None = None, label: str = "") -> int:
    t = time.time()
    log(f"$ {' '.join(map(str, cmd))[:400]}")
    with open(OUT / "release.log", "a", encoding="utf-8") as fh:
        rc = subprocess.run([str(c) for c in cmd], env={**os.environ, **(env or {})},
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    dt = round(time.time() - t, 1)
    if label:
        TIMING[label] = dt
    log(f"  rc={rc} за {dt}s")
    return rc


def _find_dir(name: str) -> Path:
    hits = sorted(p for p in IN.rglob(name) if p.is_dir())
    if not hits:
        raise RuntimeError(f"немає теки {name} під {IN}")
    return hits[0]


def setup_pass(tag: str, spec: str, models_dir: Path) -> tuple[str, dict]:
    venv = Path(f"/opt/{tag}")
    ws = Path(f"/opt/ws_{tag}")
    env = {"NYSHPORKA_WORKSPACE": str(ws)}
    sh(["uv", "venv", str(venv), "--python", "3.11"])
    if sh(["uv", "pip", "install", "--python", str(venv / "bin/python"), spec],
          label=f"setup_{tag}_nysh"):
        raise RuntimeError(f"nyshporka для {tag} не встановилась: {spec}")
    nysh = str(venv / "bin/nysh")
    ws.mkdir(parents=True, exist_ok=True)
    sh([nysh, "init", str(ws), "--yes", "--preset", "researcher"], env=env)
    if sh([nysh, "htr", "install"], env=env, label=f"setup_{tag}_engines"):
        raise RuntimeError(f"nysh htr install ({tag}) не пройшов")
    dst = ws / "data" / "spotter" / "models"
    dst.mkdir(parents=True, exist_ok=True)
    for f in models_dir.iterdir():
        if f.is_file():
            shutil.copy(f, dst / f.name)
    log(f"{tag}: моделі {sorted(p.name for p in dst.iterdir())}")
    return nysh, env


def read_cases(tag: str, nysh: str, env: dict, cases: list) -> None:
    for case in cases:
        name = case.name
        if name.startswith("lat-"):
            cmd = [nysh, "read", str(case), "--script", "latin"]
        else:
            cmd = [nysh, "read", str(case), "--with", "latin"]
        cmd += ["--case-key", f"REL/{name}", "--out", str(OUT / tag / name), "--force"]
        sh(cmd, env=env, label=f"{tag}:{name}")


def main(params: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"], label="setup_uv")
    sh(["nvidia-smi"])
    frames_root = Path("/opt/frames")
    frames_root.mkdir(parents=True, exist_ok=True)
    for arc in sorted(IN.rglob("frames*.tar")):
        with tarfile.open(arc) as tf:
            tf.extractall(frames_root)
        log(f"розпаковано {arc.name}")
    cases = sorted(p for p in frames_root.iterdir() if p.is_dir())
    want = [c for c in str(params.get("cases", "")).split(",") if c]
    if want:
        cases = [c for c in cases if c.name in want]
    log(f"справи: {[c.name for c in cases]}")
    passes = str(params.get("passes", "A,B")).split(",")

    if "A" in passes:
        nysh_a, env_a = setup_pass("A", f"nyshporka[app]=={params['nysh_a']}", _find_dir("models_a"))
        read_cases("A", nysh_a, env_a, cases)
    nysh_b = env_b = None
    if "B" in passes or params.get("gold_models"):
        whl = sorted(IN.rglob("nyshporka-*.whl"))
        if not whl:
            raise RuntimeError("немає wheel для проходу B")
        nysh_b, env_b = setup_pass("B", f"{whl[0]}[app]", _find_dir("models_b"))
    if "B" in passes:
        read_cases("B", nysh_b, env_b, cases)

    gold = [m for m in str(params.get("gold_models", "")).split(",") if m]
    gold_tar = sorted(IN.rglob("gold.tar"))
    if gold and gold_tar and nysh_b:
        ws_b = Path("/opt/ws_B")
        t = time.time()
        with tarfile.open(gold_tar[0]) as tf:
            tf.extractall(ws_b)
        log(f"голд-сет розпаковано за {time.time() - t:.0f}s")
        models = ",".join(str(ws_b / "data/spotter/models" / m) for m in gold)
        t = time.time()
        for sd in sorted((ws_b / "data/train/sets").glob("g-*")):
            sh([nysh_b, "train", "voices", "--set", sd.name, "--models", models], env=env_b)
        TIMING["gold_voices"] = round(time.time() - t, 1)
        gdst = OUT / "gold"
        for sd in sorted((ws_b / "data/train/sets").glob("g-*")):
            for m in gold:
                src = sd / "drafts" / Path(m).stem
                if src.is_dir():
                    shutil.copytree(src, gdst / sd.name / Path(m).stem, dirs_exist_ok=True)

    TIMING["total"] = round(time.time() - t_all, 1)
    (OUT / "release_timing.json").write_text(json.dumps(TIMING, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    with tarfile.open(OUT / "release_bundle.tgz", "w:gz") as tf:
        for p in sorted(OUT.iterdir()):
            if p.is_dir():
                tf.add(p, arcname=p.name)
    # Забирати ОДИН архів: пофайловий забір тисяч файлів обривався (10-04)
    for p in list(OUT.iterdir()):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    log(f"готово за {TIMING['total']}s")


if __name__ == "__main__":
    main({"nysh_a": "0.23.2", "passes": "A,B", "cases": "",
          "gold_models": "pysar_cyr_v17.pt,pysar_cyr_v19.pt"})
