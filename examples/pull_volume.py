"""Рекурсивно стягнути modal.Volume у локальну теку (надійніше за глючний `modal volume get` на Windows).

uv run python pull_volume.py <volume_name> <local_dest> [--prefix REMOTE_SUBDIR]
"""
import argparse
import pathlib

import modal

ap = argparse.ArgumentParser()
ap.add_argument("volume")
ap.add_argument("dest")
ap.add_argument("--prefix", default="/")
args = ap.parse_args()

vol = modal.Volume.from_name(args.volume)
dest = pathlib.Path(args.dest)
dest.mkdir(parents=True, exist_ok=True)

n_files = n_bytes = 0
for entry in vol.listdir(args.prefix, recursive=True):
    # FileEntry: .path (str), .type (1=FILE, 2=DIRECTORY)
    is_dir = getattr(entry.type, "name", str(entry.type)).upper().endswith("DIRECTORY") or int(getattr(entry, "type", 1)) == 2
    rel = entry.path.lstrip("/")
    target = dest / rel
    if is_dir:
        target.mkdir(parents=True, exist_ok=True)
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        continue  # resume: skip already-pulled
    with open(target, "wb") as fh:
        for chunk in vol.read_file(entry.path):
            fh.write(chunk)
            n_bytes += len(chunk)
    n_files += 1
    if n_files % 100 == 0:
        print(f"  ...{n_files} файлів", flush=True)

print(f"✓ {args.volume}: {n_files} файлів, {n_bytes/1048576:.1f} MB → {dest}", flush=True)
