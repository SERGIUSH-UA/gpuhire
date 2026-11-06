"""Фейковий шард із режимом черги: клейми, текст на сторінку, події прогресу.

Імітує лише те, що бачить бокс-раннер: "--queue", "--claim", "_drain".
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PREFIX = "@@PROGRESS@@ "
DRAIN_DIR = "_drain"


def emit(phase, **kw):
    print(PREFIX + json.dumps({"v": 1, "phase": phase, **kw}), flush=True)


def claim(out_dir, stem):
    d = out_dir / "_claims"
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / f"{stem}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="")
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--progress-json", action="store_true")
    args, _ = ap.parse_known_args()
    k = int(args.shard.split("/")[0])
    starts = Path(os.environ["FAKE_STARTS"])
    with open(starts, "a") as fh:
        fh.write(f"{os.getpid()} {k}\n")
    time.sleep(float(os.environ.get("FAKE_LOAD_SEC", "0.3")))   # «моделі»
    qfile = Path(args.queue)
    pos = 0
    while True:
        lines = [ln for ln in qfile.read_text().splitlines() if ln.strip()]
        if pos >= len(lines):
            if (qfile.parent / DRAIN_DIR / str(k)).exists():
                return 0
            emit("queue_wait")
            time.sleep(0.2)
            continue
        entry = json.loads(lines[pos])
        pos += 1
        if entry.get("end"):
            break
        qi = entry["index"]
        case_dir, out_dir = Path(entry["case_dir"]), Path(entry["out_dir"])
        if not case_dir.is_dir():
            emit("case_done", qi=qi, rc=0)
            continue
        pages = sorted(p for p in case_dir.iterdir() if p.suffix == ".jpg")
        # як Нишпорка: клейми мертвих процесів знімаються при вході в справу
        cdir = out_dir / "_claims"
        if cdir.is_dir():
            for c in cdir.glob("*.claim"):
                try:
                    pid = int(c.read_text().split()[0])
                    os.kill(pid, 0)
                except (ValueError, IndexError, ProcessLookupError, OSError):
                    c.unlink(missing_ok=True)
        quarantined = set()
        qf = out_dir / "_htr_quarantine.json"
        if qf.is_file():
            quarantined = {Path(n).stem for n in json.loads(qf.read_text()).get("pages", {})}
        for i, p in enumerate(pages, 1):
            if p.stem in quarantined:
                continue
            if (out_dir / f"{p.stem}.txt").exists() or not claim(out_dir, p.stem):
                continue
            emit("page_start", qi=qi, i=i, n=len(pages), page=p.name)
            if os.environ.get("FAKE_POISON") == f"{out_dir.parent.name}:{p.stem}":
                os._exit(1)   # нативна смерть: ні винятку, ні рядка
            slow = os.environ.get("FAKE_SLOW", "").split(":")
            if len(slow) == 3 and slow[:2] == [out_dir.parent.name, p.stem]:
                time.sleep(float(slow[2]))   # одна повільна сторінка (голова черги)
            time.sleep(float(os.environ.get("FAKE_PAGE_SEC", "0.05")))
            (out_dir / f"{p.stem}.txt").write_text("текст", encoding="utf-8")
            emit("htr", qi=qi, i=i, n=len(pages), page=p.name, sec=0.05)
        emit("case_done", qi=qi, rc=0)
    emit("queue_done", cases=pos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
