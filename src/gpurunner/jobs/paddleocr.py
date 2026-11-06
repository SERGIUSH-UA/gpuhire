"""PaddleOCRJob — wraps the embedded PaddleOCR runner as a Job.

Validates user-supplied params (URLs, language, render scale, output layout),
then renders the remote notebook body by concatenating:

  1. injected ``PARAMS`` dict literal
  2. the source of ``_embedded/paddleocr_runner.py``
  3. an entry-point line that calls ``main(PARAMS)``

This keeps the OCR loop testable as a normal Python module locally, while
still being self-contained when injected into a Kaggle notebook.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from gpurunner.core.job import Job

_URL_RX = re.compile(r"^(https?|file)://", re.IGNORECASE)


class PaddleOCRJob(Job):
    """Run PaddleOCR PP-OCRv5 across a list of PDFs and emit per-page text."""

    name: ClassVar[str] = "paddleocr"
    description: ClassVar[str] = (
        "PaddleOCR PP-OCRv5 over a list of remote PDFs. Outputs per-page txt + json."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "modal",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    # Sustained throughput observed on Kaggle T4 for cyrillic PEV pages.
    PAGES_PER_SECOND: ClassVar[float] = 1.0

    def requirements(self) -> list[str]:
        # paddle 3.x is hosted on the CN paddle index, not on PyPI proper.
        # Kaggle reaches it now that the account is phone-verified (egress
        # was the blocker for the old 2.6.2 pin). Modal always had full egress.
        return [
            "--extra-index-url",
            "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
            "paddlepaddle-gpu==3.2.0",
            # paddleocr 3.0.0 pulled paddlex 3.0.0 which pinned pandas<=1.5.3
            # (no cp312 wheel; source build fails on pkg_resources removal).
            # 3.2.0+ pulls paddlex 3.2.0+ which only requires pandas>=1.3.
            "paddleocr==3.2.0",
            "pypdfium2>=4.30",
        ]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        items = self._normalize_urls(params)
        if not items:
            raise ValueError(
                "paddleocr needs 'urls' (list[str] or list[{url, label}]) "
                "or 'urls_file' (path to text file with one URL per line)."
            )

        lang = str(params.get("lang", "ru")).strip().lower()
        if not lang:
            raise ValueError("lang must be a non-empty string")

        scale = float(params.get("scale", 2.0))
        if scale < 0.5 or scale > 6.0:
            raise ValueError(f"scale must be in [0.5, 6.0], got {scale}")

        skip_existing = bool(params.get("skip_existing", True))

        output_subdir = str(params.get("output_subdir", "ocr")).strip().strip("/\\")
        if not output_subdir:
            output_subdir = "ocr"

        dataset = params.get("dataset")
        if dataset is not None:
            dataset = str(dataset).strip()
            if dataset and "/" not in dataset:
                raise ValueError(
                    f"dataset must be '<owner>/<slug>' (got {dataset!r})"
                )

        modal_input_volume = params.get("modal_input_volume")
        if modal_input_volume is not None:
            modal_input_volume = str(modal_input_volume).strip() or None

        return {
            "items": items,
            "lang": lang,
            "scale": scale,
            "skip_existing": skip_existing,
            "output_subdir": output_subdir,
            "dataset": dataset or None,
            "modal_input_volume": modal_input_volume,
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        # Assume average issue ~30 pages until we have a better calibration.
        normalized = self.validate_params(params)
        pages = sum(int(item.get("expected_pages", 30)) for item in normalized["items"])
        seconds = pages / max(self.PAGES_PER_SECOND, 0.1)
        return timedelta(seconds=seconds + 120)  # +2 min init/install

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        # Default runs bundle output into a single ocr_bundle.tgz; non-bundle
        # runs emit per-page txt/json.
        return ["ocr_bundle.tgz", "**/*.txt", "**/*.json"]

    def is_output_complete(self, out_dir: Path) -> bool:
        if not out_dir.exists():
            return False
        # Bundle mode: the single tarball (or its untarred _summary.json) is the
        # completion marker.
        if (out_dir / "ocr_bundle.tgz").exists():
            return True
        if (out_dir / "_summary.json").exists() or any(out_dir.rglob("_summary.json")):
            return True
        return any(out_dir.rglob("page_*.txt"))

    def render_remote_code(
        self,
        params: dict[str, Any],
        *,
        shard_index: int = 0,
        total_shards: int = 1,
    ) -> str:
        normalized = self.validate_params(params)
        # Shard support: take every Nth item starting from shard_index.
        if total_shards > 1:
            normalized = dict(normalized)
            normalized["items"] = [
                it for i, it in enumerate(normalized["items"]) if i % total_shards == shard_index
            ]

        # paddlepaddle-gpu==3.2.0 cu126 downgrades nvidia-nccl-cu12 below
        # what Kaggle's torch 2.10.0+cu128 needs (`ncclCommShrink`, added in
        # NCCL 2.27). paddlex 3.2 hard-imports modelscope which lazy-imports
        # torch, so without working nccl the kernel dies during paddleocr
        # init with `undefined symbol: ncclCommShrink`. Bumping nccl back up
        # after the install fixes the chain.
        #
        # paddlex 3.2's retriever pipeline does `if is_dep_available("langchain"):
        # from langchain.docstore.document import Document`. Kaggle's base image
        # ships a newer langchain where `langchain.docstore` was removed, so the
        # guard passes but the import explodes during PaddleOCR() init with
        # `ModuleNotFoundError: No module named 'langchain.docstore'`. The RAG
        # retriever is irrelevant to OCR — uninstalling langchain makes the dep
        # guard False so paddlex skips the import entirely.
        kaggle_preamble = (
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'nvidia-nccl-cu12'], check=False)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'langchain', 'langchain-community'], check=False)"
        )
        return self.render_kaggle_code(
            "paddleocr_runner.py", normalized, prelude=kaggle_preamble
        )

    def render_runner_module(self) -> str:
        # Strip the runner's own ``if __name__ == "__main__"`` guard — Modal /
        # Kaggle wrappers call ``main()`` themselves.
        return self._runner_source("paddleocr_runner.py")

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        normalized = self.validate_params(params)
        ds = normalized.get("dataset")
        return [ds] if ds else []

    def modal_input_volumes(self, params: dict[str, Any]) -> dict[str, str]:
        normalized = self.validate_params(params)
        vol = normalized.get("modal_input_volume")
        return {"/mnt/input": vol} if vol else {}

    def modal_image_spec(self) -> dict[str, Any]:
        # Modal serializes the wrapper function across the local/remote boundary,
        # which means local + image Python versions must match. 3.12 matches
        # gpurunner's .python-version (3.13 breaks because paddlex pulls
        # pandas==1.5.3 which has no cp313 wheel).
        #
        # pip_packages and extra_index_url come from requirements(); we only
        # add the apt packages debian-slim doesn't have (Kaggle base image
        # ships them pre-installed).
        spec = super().modal_image_spec()
        spec["apt_packages"] = [
            "libgomp1",       # paddlepaddle OpenMP runtime
            "libgl1",         # opencv (libGL.so.1)
            "libglib2.0-0",   # opencv (libgthread)
            "libsm6",
            "libxext6",
            "libxrender1",
        ]
        # 4h: empirically, heavy PEV years (~50 PDFs of dense-cyr scans) take
        # 120+ min on T4. Keeping head-room avoids the 1870/1912 timeout that
        # happened with 7200s.
        spec["timeout"] = 14400
        return spec

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _normalize_urls(params: dict[str, Any]) -> list[dict[str, Any]]:
        raw_items: list[Any] = []

        if "items" in params and isinstance(params["items"], list):
            raw_items.extend(params["items"])

        if "urls" in params:
            raw_items.extend(_coerce_url_list(params["urls"]))

        if "urls_file" in params:
            path = Path(str(params["urls_file"]))
            if not path.exists():
                raise FileNotFoundError(f"urls_file not found: {path}")
            for ln in path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    raw_items.append(ln)

        normalized: list[dict[str, Any]] = []
        for i, item in enumerate(raw_items):
            if isinstance(item, str):
                url = item.strip()
                label = None
            elif isinstance(item, dict) and "url" in item:
                url = str(item["url"]).strip()
                label = item.get("label")
            else:
                raise ValueError(f"unsupported item at index {i}: {item!r}")
            if not _URL_RX.match(url):
                raise ValueError(f"item {i}: not an http(s) URL: {url}")
            urlparse(url)  # raises if malformed
            normalized.append(
                {
                    "url": url,
                    "label": str(label) if label else f"pdf_{i:04d}",
                }
            )
        return normalized


def _coerce_url_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        # Comma-separated or newline-separated string.
        parts = [p.strip() for p in re.split(r"[,\n]+", value)]
        return [p for p in parts if p]
    raise ValueError(f"urls must be list or string, got {type(value).__name__}")
