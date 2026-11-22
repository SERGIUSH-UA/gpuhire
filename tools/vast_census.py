#!/usr/bin/env python3
"""Перепис машин Vast: чи піднімається образ, скільки до SSH, що з каналом.

    python tools/vast_census.py RTX3090x5,V100,RTX4090
    python tools/vast_census.py RTX3090x5 --image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime --install

`RTX3090x5` — п'ять найдешевших надійних РІЗНИХ машин цього типу. Оренди йдуть
паралельно; на кожній піднятій машині виконується проба (`--net`, дефолт):
колесо torch cu126 з download.pytorch.org одним з'єднанням і вісьмома
діапазонами та колесо з PyPI; або (`--install`) установка torch 2.14 cu126 з
індексу й решти рушіїв із пробою ядра. Гасіння — у `finally`, стеля
самознищення — 30 хв. Висновки першого перепису — `docs/vast-engine-env.md`.

🔴 Це оренда: кожна спроба коштує копійки, але коштує.
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import uuid
from pathlib import Path

NET = r"""
set +e
W=$(curl -s https://download.pytorch.org/whl/cu126/torch/ | grep -o 'torch-2.14.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl' | head -1)
U="https://download.pytorch.org/whl/cu126/$W"
S=$(curl -s -o /dev/null -r 0-209715199 --max-time 40 -w '%{size_download} %{time_total}' "$U")
echo "single=$S"
t0=$(date +%s.%N)
for i in 0 1 2 3 4 5 6 7; do
  a=$((i*26214400)); b=$((a+26214399))
  curl -s -o /dev/null -r $a-$b --max-time 40 "$U" &
done
wait
echo "par8=209715200 $(echo "$(date +%s.%N) - $t0" | bc)"
P=$(curl -s https://pypi.org/simple/nvidia-cudnn-cu12/ | grep -o 'https://files.pythonhosted.org[^"#]*nvidia_cudnn_cu12-9[^"#]*manylinux_2_27_x86_64.whl' | tail -1)
echo "pypi_single=$(curl -s -o /dev/null -r 0-209715199 --max-time 40 -w '%{size_download} %{time_total}' "$P")"
"""

INSTALL = r"""
set +e
PY=$(command -v python || command -v python3)
t0=$(date +%s)
(command -v uv || $PY -m pip install -q uv) >/dev/null 2>&1
U=$(command -v uv || echo "$(dirname $PY)/uv")
export UV_HTTP_TIMEOUT=300
t1=$(date +%s)
$U pip install --python $PY -q torch==2.14.0 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
echo "rc_torch=$?"
t2=$(date +%s)
$U pip install --python $PY -q kraken==7.1.1 'timm>=1.0' 'nltk>=3.9' 'pytorch-lightning>=2.0' 'rapidfuzz>=3.10' torch==2.14.0 torchvision==0.29.1
echo "rc_rest=$?"
t3=$(date +%s)
$PY -c "import torch, torchvision, kraken; print('import ok', torch.__version__, float((torch.ones(4, device='cuda')*2).sum()))"
echo "uv=$((t1-t0)) torch=$((t2-t1)) rest=$((t3-t2)) total=$((t3-t0))"
"""


def _probe(gpu: str, nth: int, *, image: str, script: str, skip: set[str],
           out: Path, results: list[dict], lock: threading.Lock) -> None:
    import paramiko

    from gpurunner.auth import vast as vast_auth
    from gpurunner.backends.vast import VastBackend, _load_private_key

    row: dict = {"gpu": gpu, "nth": nth, "image": image}
    bk = VastBackend()
    try:
        offers = bk.search_offers(gpu=gpu, max_price=0.8, disk_gb=40, num_gpus=1, limit=60,
                                  min_cuda=12.6, min_cpu=4)
        good = sorted((o for o in offers
                       if float(o.get("reliability2") or o.get("reliability") or 0) >= 0.97
                       and float(o.get("inet_down") or 0) >= 200),
                      key=lambda o: float(o.get("dph_total") or 9))
        seen: set = set()
        uniq = []
        for o in good:
            mid = o.get("machine_id")
            if mid not in seen and str(mid) not in skip:
                seen.add(mid)
                uniq.append(o)
        if nth >= len(uniq):
            row["error"] = f"лише {len(uniq)} різних машин"
            return
        o = uniq[nth]
        row.update(offer=o.get("id"), machine=o.get("machine_id"), price=o.get("dph_total"),
                   geo=o.get("geolocation"), inet_down=o.get("inet_down"))
        t0 = time.time()
        try:
            info = bk.rent_box(o, disk_gb=40, label=f"census-{uuid.uuid4().hex[:6]}",
                               image=image, autodestroy_hours=0.5)
        except Exception as exc:
            row.update(boot="fail", boot_s=round(time.time() - t0), error=str(exc)[:300])
            return
        iid = info["instance_id"]
        row.update(boot="ok", boot_s=round(time.time() - t0))
        try:
            priv, _ = vast_auth.require_ssh_key()
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(info["ssh_host"], port=int(info["ssh_port"]), username="root",
                      pkey=_load_private_key(priv), timeout=30, banner_timeout=60)
            b64 = base64.b64encode(script.encode()).decode()
            _, so, _ = c.exec_command(f"echo {b64} | base64 -d | bash 2>&1", timeout=1500)
            row["probe"] = so.read().decode(errors="replace").strip().splitlines()[-12:]
        except Exception as exc:
            row["probe_error"] = f"{type(exc).__name__}: {exc}"[:200]
        finally:
            bk.destroy_box(iid)
            row["rent_min"] = round((time.time() - t0) / 60, 1)
    finally:
        with lock:
            results.append(row)
            out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gpus", help="типи карт через кому; `RTX3090x5` — п'ять різних машин")
    ap.add_argument("--image", default="vastai/pytorch:cuda-12.6.3-auto")
    ap.add_argument("--install", action="store_true",
                    help="замість проби каналу — установка torch 2.14 cu126 і рушіїв")
    ap.add_argument("--skip", default="", help="machine_id через кому, які не брати")
    ap.add_argument("--out", type=Path, default=Path("vast_census.json"))
    a = ap.parse_args()
    jobs = []
    for g in a.gpus.split(","):
        name, _, k = g.partition("x")
        jobs += [(name, i) for i in range(int(k or 1))]
    results: list[dict] = []
    lock = threading.Lock()
    skip = set(filter(None, a.skip.split(",")))
    threads = [threading.Thread(target=_probe, args=j,
                                kwargs={"image": a.image, "script": INSTALL if a.install else NET,
                                        "skip": skip, "out": a.out, "results": results,
                                        "lock": lock})
               for j in jobs]
    for t in threads:
        t.start()
        time.sleep(3)
    for t in threads:
        t.join()
    print(f"ГОТОВО → {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
