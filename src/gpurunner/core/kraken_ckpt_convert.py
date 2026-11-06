"""Чекпойнти ketos `checkpoint_NN-0.xxxx.ckpt` → ваги епох `epoch_NN.*` — удома.

Те саме, що робить `_embedded/kraken_train_runner.py` після трену на боксі, і в
тому ж порядку: спершу VGSL → `epoch_NN.mlmodel`; не-VGSL (PP-OCRv6, kraken ≥ 7.1)
VGSL-класом не відкривається — тоді `ketos convert` → `epoch_NN.safetensors`.
Потрібен, коли бокс зник посеред трену і з R2 доїхали лише чекпойнти.

Скрипт САМОСТІЙНИЙ (без імпортів gpurunner): запускається інтерпретатором, де
стоїть kraken тієї ж версії, що на боксі (типово 7.1.1). У тому інтерпретаторі
gpurunner зазвичай немає, тож `-m` не підходить — через CLI:

    gpurunner train-out convert <тека> --python <python з kraken>

або напряму файлом:

    <python з kraken> <шлях до цього файла> <тека> [--format auto|safetensors]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_CKPT = re.compile(r"^checkpoint_(\d+)-([\d.]+)\.ckpt$")


def _ketos_convert(src: Path, dst: Path) -> int:
    """`ketos convert -o dst src` тим самим інтерпретатором (як лаунчер на боксі)."""
    return subprocess.run(
        [sys.executable, "-c", "from kraken.ketos import cli; cli()",
         "convert", "-o", str(dst), str(src)],
        capture_output=True, text=True).returncode


def convert(folder: Path, *, fmt: str = "auto", out: Path | None = None) -> list[str]:
    """Сконвертувати всі чекпойнти теки, яких ще немає у виході. → імена створених файлів."""
    out = out or folder
    out.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    for p in sorted(folder.glob("checkpoint_*.ckpt")):
        m = _CKPT.match(p.name)
        if not m:
            continue
        nn = int(m.group(1))
        mlmodel = out / f"epoch_{nn:02d}.mlmodel"
        st = out / f"epoch_{nn:02d}.safetensors"
        if mlmodel.exists() or st.exists():
            continue
        if fmt == "auto":
            try:
                from kraken.train import VGSLRecognitionModel

                mm = VGSLRecognitionModel.load_from_checkpoint(str(p), weights_only=False)
                mm.net.save_model(str(mlmodel))
                made.append(mlmodel.name)
                print(f"{p.name} → {mlmodel.name}", flush=True)
                continue
            except Exception as exc:
                print(f"{p.name}: VGSL не відкрився ({type(exc).__name__}) — ketos convert",
                      flush=True)
        rc = _ketos_convert(p, st)
        if rc == 0 and st.exists():
            made.append(st.name)
            print(f"{p.name} → {st.name}", flush=True)
        else:
            print(f"⚠ {p.name} не сконвертувався (ketos convert rc={rc})", flush=True)
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("folder", type=Path, help="Тека з checkpoint_*.ckpt")
    ap.add_argument("--format", choices=("auto", "safetensors"), default="auto",
                    help="auto — як на боксі (VGSL → .mlmodel, інакше .safetensors)")
    ap.add_argument("--out", type=Path, default=None, help="Куди класти epoch_NN.* (типово — та сама тека)")
    args = ap.parse_args(argv)
    made = convert(args.folder, fmt=args.format, out=args.out)
    print(f"створено: {len(made)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
