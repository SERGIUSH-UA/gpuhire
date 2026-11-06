"""HTRPageBenchJob — page-level бенчмарк HTR (TrOCR + Surya) на повних сторінках.

Половина бенчмарку HTR-vs-Spotter: чи знаходять HTR-движки прізвище роду,
транскрибуючи ПОВНУ сторінку, і за який час. Транскрипція → fuzzy-max по цільових
формах → page pred/score. PaddleOCR — окремий job (несумісний стек залежностей).

Вхід — Kaggle Dataset: ``images/*.png`` + ``ground_truth.json`` + ``targets.json``.
Вихід у ``/kaggle/working/``: ``results_htr.tsv`` + ``results_htr.json``
(per-движок cold_start, per-page pred/score/t_infer_ms).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job

DEFAULT_MODELS = ("kazars24/trocr-base-handwritten-ru", "surya")


class HTRPageBenchJob(Job):
    """Page-level HTR benchmark: full-page transcription + fuzzy surname match."""

    name: ClassVar[str] = "htr_page_bench"
    description: ClassVar[str] = (
        "Page-level HTR benchmark (TrOCR + Surya): full-page transcription, "
        "fuzzy surname match -> per-page pred/score/t_infer_ms."
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
        return [
            "transformers>=4.45",
            "pillow>=10.0",
            "rapidfuzz>=3.0",
            "accelerate>=0.30",
            # surya 0.20 жорстко тримає httpx<0.28; без явного піна pip backtrack-ив
            # surya до 0.17.1 (інший, зламаний API). Пінимо обидва.
            "httpx>=0.27,<0.28",
            "surya-ocr==0.20.0",
        ]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError("htr_page_bench needs 'dataset=<owner>/<slug>'")
        dataset = str(params["dataset"]).strip()
        if "/" not in dataset:
            raise ValueError(f"dataset must be '<owner>/<slug>', got {dataset!r}")
        models_raw = params.get("models", ",".join(DEFAULT_MODELS))
        models = (list(models_raw) if isinstance(models_raw, list)
                  else [m.strip() for m in str(models_raw).split(",") if m.strip()])
        if not models:
            raise ValueError("models must have at least one entry")
        return {
            "dataset": dataset,
            "models": models,
            "image_glob": str(params.get("image_glob", "images/*.png")).strip(),
            "ground_truth_file": str(params.get("ground_truth_file", "ground_truth.json")).strip(),
            "targets_file": str(params.get("targets_file", "targets.json")).strip(),
            "threshold": float(params.get("threshold", 60.0)),
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        normalized = self.validate_params(params)
        n_pages = int(params.get("estimated_n_pages", 31))
        return timedelta(seconds=n_pages * 5 * len(normalized["models"]) + 600)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["results_htr.json", "results_htr.tsv"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        return [params["dataset"]]

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("htr_page_bench doesn't support sharding")
        return self.render_kaggle_code("htr_page_bench_runner.py", normalized)
