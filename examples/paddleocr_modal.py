"""End-to-end smoke test: submit a tiny PaddleOCR job to Modal, wait, fetch.

Run with:
    uv run python examples/paddleocr_modal.py <pdf-url> [lang]

Same load as ``paddleocr_kaggle.py``, on a Modal T4. Expected total wall-clock
~3-5 min on first run (image build dominates), 30-60 s on subsequent runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from gpurunner.backends import ModalBackend
from gpurunner.core import JobStatus, manifest
from gpurunner.jobs import PaddleOCRJob


def main(url: str, lang: str = "ru") -> int:
    backend = ModalBackend()
    print("→ verifying Modal auth …")
    backend.check_auth()
    print("  ok")

    job = PaddleOCRJob()
    params = {
        "urls": [{"url": url, "label": "sample_01"}],
        "lang": lang,
        "scale": 2.0,
    }
    normalized = job.validate_params(params)
    print(f"→ normalized params: items={len(normalized['items'])} lang={normalized['lang']}")

    print("→ submitting to Modal (T4) — image build can take ~3 min on first run …")
    t_submit = time.monotonic()
    handle = backend.submit(job, params, gpu="T4")
    handle.output_dir = str(Path("./e2e_modal_out") / handle.id[:8])
    manifest.add(handle)
    print(f"  spawned in {time.monotonic() - t_submit:.1f}s")
    print(f"  handle = {handle.id[:8]}")
    print(f"  remote = {handle.remote_id}")
    print(f"  output = {handle.output_dir}")

    print("→ watching status (poll every 15s) …")
    terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    last = None
    started = time.monotonic()
    while True:
        report = backend.status(handle)
        line = f"  [{int(time.monotonic() - started):>4}s] {report.status}"
        if report.message:
            line += f" — {report.message}"
        if report.error and report.status != JobStatus.RUNNING:
            line += f" ! {report.error}"
        if line != last:
            print(line, flush=True)
            last = line
        handle.status = report.status
        manifest.update(handle)
        if report.status in terminal:
            break
        time.sleep(15)

    if report.status != JobStatus.COMPLETED:
        print(f"!! finished with status={report.status} error={report.error}")
        return 2

    print("→ fetching outputs …")
    out_dir = Path(handle.output_dir)
    files = backend.fetch_outputs(handle, out_dir)
    print(f"  {len(files)} file(s) → {out_dir}")
    for p in files[:8]:
        print(f"    {p.name}  ({p.stat().st_size:,} bytes)")

    txts = sorted(out_dir.rglob("page_*.txt"))
    print(f"\n→ {len(txts)} page-level .txt files")
    if txts:
        sample = txts[len(txts) // 2]
        snippet = sample.read_text(encoding="utf-8")[:1000]
        print(f"\n=== sample {sample.name} (first 1000 chars) ===")
        print(snippet)
        print("=" * 70)

        import re

        cyr = re.compile(r"[Ѐ-ӿ]")
        cyr_chars = len(cyr.findall(snippet))
        print(f"\nCyrillic chars in sample: {cyr_chars}")
        if cyr_chars < 50:
            print("!! suspiciously few cyrillic — OCR may have struggled")
            return 6

    summary = out_dir / "_summary.json"
    if summary.exists():
        print("\n=== summary ===")
        print(summary.read_text(encoding="utf-8")[:1500])

    print("\n[OK] OCR pipeline validated end-to-end on Modal.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:3]))
