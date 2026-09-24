"""PaddleOCR PP-OCRv5 batch loop executed inside the remote runtime.

This module is **not** meant to be imported by gpurunner locally — it is
read as source by ``PaddleOCRJob.render_remote_code`` and injected into the
remote notebook. It must therefore be self-contained: stdlib + the packages
listed in ``PaddleOCRJob.requirements()``.

Output layout (under ``output_root``):

  <label>/page_0000.txt   plain text per page (one OCR block per line)
  <label>/page_0000.json  {page, blocks: [{text, score, poly}]}
  <label>/_meta.json      {url, pages, elapsed_s, hash, started_at, finished_at}

It is resume-friendly: a page is skipped if both ``page_NNNN.txt`` and
``page_NNNN.json`` already exist (so partial runs can pick up cleanly).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _safe_label(label: str) -> str:
    """Strip path separators and anything else that breaks filesystem layout."""
    bad = '<>:"/\\|?*\0'
    return "".join("_" if c in bad else c for c in label).strip(". ") or "pdf"


def _resolve_local(url: str) -> str:
    """For ``file://`` URLs, return a usable **local filesystem path** if the
    file can be located, else the original url.

    Two Kaggle quirks are handled:
      * a dataset ``owner/slug`` mounts under a directory whose name can differ
        from the URL we guessed → glob by basename under the input roots;
      * Kaggle **strips non-ASCII characters** from uploaded filenames (e.g.
        ``…Д-174-1-т.1.pdf`` becomes ``…-174-1-.1.pdf``) → also glob by the
        ASCII-only basename so cyrillic-named PDFs still resolve.
    """
    if not url.startswith("file://"):
        return url
    import glob as _glob
    from urllib.parse import unquote, urlparse

    path = unquote(urlparse(url).path)
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    ascii_base = "".join(c for c in base if ord(c) < 128)
    patterns = {base, ascii_base} - {""}
    for root in ("/kaggle/input", "/kaggle/working", "/mnt/input"):
        for pat in patterns:
            hits = _glob.glob(os.path.join(root, "**", pat), recursive=True)
            if hits:
                if hits[0] != path:
                    print(f"  file:// fallback: {base!r} -> {hits[0]}", flush=True)
                return hits[0]
    return url  # not found; let _download raise


def _download(url: str, dest: Path, *, max_attempts: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_local(url)

    # Local file (file:// resolved to a path): copy directly. Avoids urlretrieve
    # choking on spaces / non-ASCII in the path, which Kaggle filenames contain.
    if not resolved.startswith(("http://", "https://", "file://")) and os.path.exists(resolved):
        import shutil

        shutil.copy(resolved, dest)
        return

    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + f".part{attempt}")
            # Wikimedia (and other hosts) 403 the default Python-urllib UA.
            req = urllib.request.Request(
                resolved,
                # Контакт у UA — адреса проєкту, не людини: рядок осідає в логах чужих
                # серверів із кожного орендованого боксу.
                headers={"User-Agent": "gpurunner (+https://github.com/SERGIUSH-UA/gpuhire)"},
            )
            with urllib.request.urlopen(req) as resp, tmp.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url} after {max_attempts} attempts: {last_err}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_pages(pdf_path: Path, scale: float):
    import pypdfium2 as pdfium  # type: ignore[import-not-found]

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for idx in range(len(pdf)):
            page = pdf[idx]
            pil = page.render(scale=scale).to_pil()
            yield idx, pil
    finally:
        pdf.close()


def _init_ocr(lang: str) -> Any:
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

    # PaddleOCR 3.x init signature (PP-OCRv5). The booleans below disable
    # extra pipeline stages we don't need for scanned upright pages.
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=lang,
    )


def _ocr_one_page(ocr: Any, pil_img: Any, scratch: Path) -> tuple[list[str], list[dict[str, Any]]]:
    # PaddleOCR 3.x ``predict()`` returns a list of dicts with
    # ``rec_texts``/``rec_scores``/``rec_polys`` keys, one per input image.
    scratch.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(scratch, "PNG")
    result = ocr.predict(str(scratch))

    texts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for res in result:
        rec_texts = res.get("rec_texts", []) or []
        rec_scores = res.get("rec_scores", []) or []
        rec_polys = res.get("rec_polys", []) or []
        for t, s, poly in zip(rec_texts, rec_scores, rec_polys):
            texts.append(t)
            blocks.append(
                {
                    "text": t,
                    "score": float(s),
                    "poly": [[float(x), float(y)] for x, y in poly],
                }
            )
    return texts, blocks


def run_one(item: dict[str, Any], output_root: Path, *, lang: str, scale: float, skip_existing: bool, ocr: Any, scratch_pdf: Path, scratch_png: Path) -> dict[str, Any]:
    label = _safe_label(item.get("label") or "pdf")
    url = item["url"]
    dest_dir = output_root / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    started_iso = _utc_iso()

    print(f"[{started_iso}] {label}: download {url}", flush=True)
    _download(url, scratch_pdf)
    pdf_hash = _sha256(scratch_pdf)
    print(f"  sha256={pdf_hash[:16]}…  size={scratch_pdf.stat().st_size // 1024} KB", flush=True)

    pages_processed = 0
    pages_skipped = 0
    pages_total = 0
    for page_idx, pil in _iter_pages(scratch_pdf, scale):
        pages_total = max(pages_total, page_idx + 1)
        txt_path = dest_dir / f"page_{page_idx:04d}.txt"
        json_path = dest_dir / f"page_{page_idx:04d}.json"
        if skip_existing and txt_path.exists() and json_path.exists():
            pages_skipped += 1
            continue
        t1 = time.monotonic()
        try:
            texts, blocks = _ocr_one_page(ocr, pil, scratch_png)
        except Exception as e:
            print(f"  ! page {page_idx} OCR failed: {e}", flush=True)
            traceback.print_exc()
            continue
        txt_path.write_text("\n".join(texts), encoding="utf-8")
        json_path.write_text(
            json.dumps({"page": page_idx, "blocks": blocks}, ensure_ascii=False),
            encoding="utf-8",
        )
        pages_processed += 1
        if pages_processed % 10 == 0 or pages_processed <= 3:
            print(
                f"  page {page_idx}: {len(texts)} blocks, {time.monotonic() - t1:.1f}s",
                flush=True,
            )

    elapsed = time.monotonic() - started
    meta = {
        "url": url,
        "label": label,
        "pages_total": pages_total,
        "pages_processed": pages_processed,
        "pages_skipped": pages_skipped,
        "elapsed_s": round(elapsed, 1),
        "sha256": pdf_hash,
        "lang": lang,
        "scale": scale,
        "started_at": started_iso,
        "finished_at": _utc_iso(),
    }
    (dest_dir / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Free up the downloaded PDF — it can be huge (40 MB+) and we have it OCR'd.
    try:
        scratch_pdf.unlink(missing_ok=True)
    except Exception:
        pass

    print(
        f"[{_utc_iso()}] {label}: done — {pages_processed} new / {pages_skipped} skipped / "
        f"{pages_total} total in {elapsed:.1f}s",
        flush=True,
    )
    return meta


def main(params: dict[str, Any]) -> dict[str, Any]:
    # Kaggle output retrieval is per-file (one signed-URL GET each) AND the SDK
    # paginates at ~500 files — so a shard emitting thousands of tiny txt/json
    # files is painfully slow and easily truncated to fetch. When ``bundle`` is
    # on (default), write pages to a scratch dir OUTSIDE /kaggle/working and tar
    # the whole tree into a single ``ocr_bundle.tgz`` at the end, so the kernel
    # exposes exactly ONE small output file. The orchestrator untars it locally.
    bundle = bool(params.get("bundle", True))
    if bundle:
        output_root = Path(params.get("output_root") or "/tmp/gpurunner_ocr_out/ocr")
    else:
        output_root = Path(params.get("output_root") or "/kaggle/working/ocr")
    output_root.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = params["items"]
    lang: str = params.get("lang", "ru")
    scale: float = float(params.get("scale", 2.0))
    skip_existing: bool = bool(params.get("skip_existing", True))

    scratch_dir = Path(params.get("scratch_dir") or "/tmp/gpurunner")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_pdf = scratch_dir / "current.pdf"
    scratch_png = scratch_dir / "current_page.png"

    # One-time diagnostic: show what's actually mounted under /kaggle/input so
    # file:// path mismatches are obvious in the log.
    if os.path.isdir("/kaggle/input"):
        print("Mounted /kaggle/input:", flush=True)
        for root, _dirs, files in os.walk("/kaggle/input"):
            depth = root.count("/") - 2
            if depth <= 2:
                print(f"  {root}  ({len(files)} files)", flush=True)

    print(f"PaddleOCR job: {len(items)} pdf(s), lang={lang}, scale={scale}", flush=True)
    print(f"Output root: {output_root}", flush=True)

    print("Initializing PaddleOCR PP-OCRv5…", flush=True)
    t_init = time.monotonic()
    ocr = _init_ocr(lang)
    print(f"  ready in {time.monotonic() - t_init:.1f}s", flush=True)

    summaries: list[dict[str, Any]] = []
    job_started = _utc_iso()
    t_job = time.monotonic()
    for i, item in enumerate(items, 1):
        print(f"\n=== {i}/{len(items)} {item.get('label') or item['url']}", flush=True)
        try:
            summaries.append(
                run_one(
                    item,
                    output_root,
                    lang=lang,
                    scale=scale,
                    skip_existing=skip_existing,
                    ocr=ocr,
                    scratch_pdf=scratch_pdf,
                    scratch_png=scratch_png,
                )
            )
        except Exception as e:
            print(f"  ! item failed: {e}", flush=True)
            traceback.print_exc()
            summaries.append({"url": item["url"], "error": str(e)})

    summary = {
        "started_at": job_started,
        "finished_at": _utc_iso(),
        "elapsed_s": round(time.monotonic() - t_job, 1),
        "items": summaries,
    }
    summary_path = output_root / "_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nFinal summary: {summary_path}", flush=True)

    if bundle:
        import tarfile

        bundle_path = Path("/kaggle/working/ocr_bundle.tgz")
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(output_root, arcname="ocr")
        print(
            f"Bundled output → {bundle_path} "
            f"({bundle_path.stat().st_size // 1024} KB, one file to fetch)",
            flush=True,
        )

    return summary


if __name__ == "__main__":
    # Allow standalone runs by reading params from env or argv[1] (JSON path).
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        params = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        raw = os.environ.get("GPURUNNER_PARAMS", "{}")
        params = json.loads(raw)
    main(params)
