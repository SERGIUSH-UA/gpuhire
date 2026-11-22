"""PaddlePageBenchJob — page-level бенчмарк PaddleOCR PP-OCRv5 на повних сторінках.

Окремий job (не htr_page_bench) через несумісний стек: paddlepaddle-gpu ставиться з
CN-індексу (як у PaddleOCRJob). Той самий вихідний контракт, що інші *_page_bench.

Вхід — Kaggle Dataset: ``images/*.png`` + ``ground_truth.json`` + ``targets.json``.
Вихід у ``/kaggle/working/``: ``results_paddle.tsv`` + ``results_paddle.json``.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class PaddlePageBenchJob(Job):
    """Page-level PaddleOCR PP-OCRv5 benchmark: full-page OCR + fuzzy surname match."""

    name: ClassVar[str] = "paddle_page_bench"
    description: ClassVar[str] = (
        "Page-level PaddleOCR PP-OCRv5 benchmark: full-page OCR, fuzzy surname "
        "match -> per-page pred/score/t_infer_ms."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    def requirements(self) -> list[str]:
        # paddle 3.x — з CN paddle index (як у PaddleOCRJob; Kaggle дотягується після
        # phone-verify акаунта).
        return [
            "--extra-index-url",
            "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
            "paddlepaddle-gpu==3.2.0",
            "paddleocr==3.2.0",
            "rapidfuzz>=3.0",
            "pillow>=10.0",
        ]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError("paddle_page_bench needs 'dataset=<owner>/<slug>'")
        dataset = str(params["dataset"]).strip()
        if "/" not in dataset:
            raise ValueError(f"dataset must be '<owner>/<slug>', got {dataset!r}")
        return {
            "dataset": dataset,
            "image_glob": str(params.get("image_glob", "images/*.png")).strip(),
            "ground_truth_file": str(params.get("ground_truth_file", "ground_truth.json")).strip(),
            "targets_file": str(params.get("targets_file", "targets.json")).strip(),
            "threshold": float(params.get("threshold", 60.0)),
            "lang": str(params.get("lang", "ru")).strip(),
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        n_pages = int(params.get("estimated_n_pages", 31))
        return timedelta(seconds=n_pages * 4 + 600)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["results_paddle.json", "results_paddle.tsv"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        return [params["dataset"]]

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("paddle_page_bench doesn't support sharding")
        return self.render_kaggle_code("paddle_page_bench_runner.py", normalized)
