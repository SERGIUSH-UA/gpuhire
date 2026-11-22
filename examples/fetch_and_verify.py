"""Fetch outputs from a finished Kaggle kernel and sanity-check the OCR result.

Usage:
    uv run python examples/fetch_and_verify.py <kernel-ref-or-handle-prefix>

If the kernel is not yet terminal, exits with code 4.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from gpurunner.backends import KaggleBackend
from gpurunner.core import JobStatus, manifest

CYR = re.compile(r"[Ѐ-ӿ]")


def main(arg: str) -> int:
    backend = KaggleBackend()
    handle = manifest.get(arg)

    # Allow raw kernel refs ("user/slug") that aren't in our manifest yet.
    if handle is None and "/" in arg:
        from gpurunner.core.models import JobHandle

        handle = JobHandle(
            backend="kaggle",
            remote_id=arg,
            job_name="paddleocr",
            params={},
            gpu="T4",
        )

    if handle is None:
        print(f"!! unknown handle/ref: {arg}", file=sys.stderr)
        return 2

    print(f"→ status of {handle.remote_id}")
    rep = backend.status(handle)
    print(f"  {rep.status}" + (f" — {rep.message}" if rep.message else ""))
    if rep.status not in {JobStatus.COMPLETED, JobStatus.FAILED}:
        print("!! not yet terminal", file=sys.stderr)
        return 4

    out_dir = Path(handle.output_dir) if handle.output_dir else Path("./e2e_out") / handle.id[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ fetching to {out_dir}")
    files = backend.fetch_outputs(handle, out_dir)
    print(f"  {len(files)} file(s) downloaded")
    for p in files[:8]:
        size = Path(p).stat().st_size if Path(p).exists() else 0
        print(f"    {Path(p).name}  ({size:,} bytes)")

    if rep.status == JobStatus.FAILED:
        print("\n--- last logs (tail) ---")
        for ln in list(backend.logs(handle))[-30:]:
            print(f"  | {ln}")
        return 3

    txts = sorted(out_dir.rglob("page_*.txt"))
    print(f"\n→ {len(txts)} page txt files")
    if not txts:
        print("!! no page txt files — runner may have failed silently", file=sys.stderr)
        return 5

    # Pick the middle page to read.
    mid = txts[len(txts) // 2]
    snippet = mid.read_text(encoding="utf-8")
    cyr = len(CYR.findall(snippet))
    print(f"\n=== sample: {mid.name} ({len(snippet)} chars, {cyr} cyrillic) ===")
    print(snippet[:1200])
    print("=" * 70)

    summary = out_dir / "ocr" / "_summary.json"
    if not summary.exists():
        summary = next(iter(out_dir.rglob("_summary.json")), None)
    if summary and summary.exists():
        print(f"\n→ summary: {summary}")
        print(summary.read_text(encoding="utf-8"))

    if cyr < 50:
        print("\n!! very few cyrillic chars on sample page — OCR may have failed", file=sys.stderr)
        return 6
    print("\n[green] OCR quality: cyrillic detected, looks plausible. [/green]")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
