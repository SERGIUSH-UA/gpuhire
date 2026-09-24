"""End-to-end smoke test: submit a tiny PaddleOCR job to Kaggle, wait, fetch.

Run with:

    uv run python examples/paddleocr_kaggle.py <pdf-url> [lang]

Give it a short scanned PDF you are allowed to fetch (a 16-page issue burns
approximately 5-10 min of Kaggle T4 quota). ``lang`` is a PaddleOCR language
code, default ``ru``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from gpurunner.backends import KaggleBackend
from gpurunner.core import JobStatus, manifest
from gpurunner.jobs import PaddleOCRJob


def main(url: str, lang: str = "ru") -> int:
    backend = KaggleBackend()
    print("→ verifying Kaggle auth …")
    backend.check_auth()
    print(f"  ok, user = {backend._get_username()}")

    job = PaddleOCRJob()
    params = {
        "urls": [{"url": url, "label": "sample_01"}],
        "lang": lang,
        "scale": 2.0,
    }
    normalized = job.validate_params(params)
    print(f"→ normalized params: items={len(normalized['items'])} lang={normalized['lang']}")

    print("→ submitting to Kaggle (T4) …")
    handle = backend.submit(job, params, gpu="T4")
    handle.output_dir = str(Path("./e2e_out") / handle.id[:8])
    manifest.add(handle)

    print(f"  handle = {handle.id[:8]}")
    print(f"  remote = {handle.remote_id}")
    print(f"  output = {handle.output_dir}")

    print("→ watching status (poll every 20s) …")
    terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    last = None
    started = time.monotonic()
    while True:
        report = backend.status(handle)
        line = f"  [{int(time.monotonic() - started):>4}s] {report.status}"
        if report.message:
            line += f" — {report.message}"
        if line != last:
            print(line, flush=True)
            last = line
        handle.status = report.status
        manifest.update(handle)
        if report.status in terminal:
            break
        time.sleep(20)

    if report.status != JobStatus.COMPLETED:
        print(f"!! finished with status={report.status} error={report.error}")
        print("→ fetching logs …")
        for ln in backend.logs(handle):
            print(f"  | {ln}")
        return 2

    print("→ fetching outputs …")
    out_dir = Path(handle.output_dir)
    files = backend.fetch_outputs(handle, out_dir)
    print(f"  {len(files)} file(s) → {out_dir}")
    for p in files[:5]:
        print(f"    {p}")

    # Quick text-quality check.
    txts = sorted(out_dir.rglob("page_*.txt"))
    print(f"→ {len(txts)} page-level .txt files")
    if txts:
        sample = txts[len(txts) // 2]
        snippet = sample.read_text(encoding="utf-8")[:500]
        print(f"→ sample {sample.name} (first 500 chars):")
        print("-" * 60)
        print(snippet)
        print("-" * 60)

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:3]))
