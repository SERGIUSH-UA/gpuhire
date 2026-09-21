"""KrakenLinesJob — сегментація + кропи рядків + Kraken-текст для дистиляції.

Кейс: база файн-тюну CHURRO→Kraken. Кернел робить blla-сегментацію,
вирізає кожен рядок і транскрибує його McCATMuS'ом ОДНИМ процесом — тож
crop_i ↔ kraken_line_i узгоджені за побудовою (якір для align із CHURRO).

Вхід — датасет(и) Kaggle: сторінки *.jpg + окремий датасет із *.mlmodel.
Вихід — ``kraken_lines_bundle.tgz`` (lines/ + kraken/) + ``klines_summary.json``.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class KrakenLinesJob(Job):
    """Line crops + Kraken anchor text over a page-image dataset (2×T4)."""

    name: ClassVar[str] = "kraken_lines"
    description: ClassVar[str] = (
        "Segment pages (kraken blla), extract per-line crops AND transcribe "
        "them with a .mlmodel — consistent indexing for CHURRO distillation."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    SEC_PER_PAGE: ClassVar[float] = 5.0

    def requirements(self) -> list[str]:
        return ["kraken>=5.0", "pillow>=10.0"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError("kraken_lines needs 'dataset=<owner>/<slug>'")
        dataset = str(params["dataset"]).strip()
        if "/" not in dataset:
            raise ValueError(f"dataset must be '<owner>/<slug>', got {dataset!r}")
        model_dataset = str(params.get("model_dataset", "")).strip()
        if model_dataset and "/" not in model_dataset:
            raise ValueError(f"model_dataset must be '<owner>/<slug>', got {model_dataset!r}")
        return {"dataset": dataset, "model_dataset": model_dataset}

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        n_pages = int(params.get("estimated_n_pages", 500))
        return timedelta(seconds=n_pages * self.SEC_PER_PAGE + 600)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["kraken_lines_bundle.tgz", "klines_summary.json"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        normalized = self.validate_params(params)
        out = [normalized["dataset"]]
        if normalized["model_dataset"]:
            out.append(normalized["model_dataset"])
        return out

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("kraken_lines doesn't support sharding")
        return self.render_kaggle_code("kraken_lines_runner.py", normalized)
