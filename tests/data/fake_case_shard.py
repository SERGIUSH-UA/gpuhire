"""Фейковий шард ОДНІЄЇ справи: правило пропуску, мета, клейми, догін.

Імітує те, на чому стоїть нарізка кадрів на частини: сторінка пропускається,
лише коли є її текст І запис у меті (як `htr_case_run.py`), а порожній файл
замість кадру — це збій сторінки, а не тиха згода. Прапорці: "--case-dir",
"--out-dir", "--shard", "--claim", "--pages", "--progress-json".
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

PREFIX = "@@PROGRESS@@ "


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


def known_pages(out_dir):
    seen = set()
    for p in [out_dir / "_htr_meta.json", *out_dir.glob("_htr_meta.part*.json")]:
        try:
            seen.update(json.loads(p.read_text(encoding="utf-8")).get("pages") or {})
        except (OSError, ValueError):
            continue
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--pages", default="")
    ap.add_argument("--progress-json", action="store_true")
    args, _ = ap.parse_known_args()
    k, n_shards = (int(x) for x in args.shard.split("/"))
    case_dir, out_dir = Path(args.case_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Один раз за тест: текст «зник» уже після того, як частину визнали
    # прочитаною (запобіжник перед догоном мусить довезти справжні кадри).
    vanish = os.environ.get("FAKE_VANISH", "")
    marker = out_dir / "_vanished"
    if vanish and not marker.exists():
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            (out_dir / f"{vanish}.txt").unlink(missing_ok=True)
        except FileExistsError:
            pass

    pages = sorted(p for p in case_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.pages:
        wanted = {int(x) for x in args.pages.split(",") if x.strip()}
        pages = [p for i, p in enumerate(pages, 1) if i in wanted]
    meta_path = out_dir / (f"_htr_meta.part{k}.json" if n_shards > 1 else "_htr_meta.json")
    meta = {"version": 1, "frames_total": len(list(case_dir.iterdir())), "pages": {}}
    if meta_path.is_file():
        with contextlib.suppress(ValueError):
            meta["pages"] = json.loads(meta_path.read_text(encoding="utf-8")).get("pages") or {}
    already = known_pages(out_dir) if n_shards > 1 else set(meta["pages"])
    log = os.environ.get("FAKE_READ_LOG")
    done = skipped = failed = 0
    for i, src in enumerate(pages, 1):
        txt = out_dir / f"{src.stem}.txt"
        if txt.exists() and src.name in already:
            skipped += 1
            emit("htr", i=i, n=len(pages), page=src.name, skipped=True)
            continue
        if args.claim and n_shards > 1 and not claim(out_dir, src.stem):
            continue
        emit("page_start", i=i, n=len(pages), page=src.name)
        if src.stat().st_size == 0:
            failed += 1
            emit("htr", i=i, n=len(pages), page=src.name, error="порожній кадр")
            continue
        time.sleep(float(os.environ.get("FAKE_PAGE_SEC", "0.02")))
        txt.write_text("текст", encoding="utf-8")
        meta["pages"][src.name] = {"lines": 3, "sec": 0.02}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        if log:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(src.stem + "\n")
        done += 1
        emit("htr", i=i, n=len(pages), page=src.name, sec=0.02)
    emit("done", pages=done, skipped=skipped, failed=failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
