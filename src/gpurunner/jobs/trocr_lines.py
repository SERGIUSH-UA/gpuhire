"""TrOCRLinesJob — blla-сегментація + кропи рядків + TrOCR-транскрипція.

Кейс: пілот якості Kansallisarkisto/cyrillic-htr-model (trocr-large,
кирилиця XVII-XX ст.) на метриках М'ястківки → якщо читає, стає вчителем
псевдо-GT для кириличного kraken (аналог дистиляції CHURRO→Kraken на ф.792).

Вхід — сторінки *.jpg/*.png (Kaggle Dataset або lightning input_root).
Вихід — ``trocr_lines_bundle.tgz`` (lines/ + trocr/) + ``trocr_summary.json``
— та сама схема, що kraken_lines (сумісна з htr_distill_extract/align).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job

DEFAULT_MODEL = "Kansallisarkisto/cyrillic-htr-model"


class TrOCRLinesJob(Job):
    """Line crops + TrOCR text over a page-image dataset (T4)."""

    name: ClassVar[str] = "trocr_lines"
    description: ClassVar[str] = (
        "Segment pages (kraken blla), extract per-line crops AND transcribe "
        "them with a HF TrOCR model (default Kansallisarkisto cyrillic-htr)."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    SEC_PER_PAGE: ClassVar[float] = 15.0  # ~5s blla + ~10s TrOCR-large batch на T4

    def requirements(self) -> list[str]:
        return ["kraken>=5.0", "transformers>=4.45", "pillow>=10.0"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        dataset = str(params.get("dataset", "")).strip()
        return {
            "dataset": dataset,
            "model": str(params.get("model", DEFAULT_MODEL)).strip(),
            "batch": int(params.get("batch", 24)),
            "max_new_tokens": int(params.get("max_new_tokens", 96)),
            # кома-список stem'ів для вибіркового прогону ("0016,0043"); "" = всі
            "pages": str(params.get("pages", "")).strip(),
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        n_pages = int(params.get("estimated_n_pages", 60))
        return timedelta(seconds=n_pages * self.SEC_PER_PAGE + 900)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["trocr_lines_bundle.tgz", "trocr_summary.json"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        normalized = self.validate_params(params)
        return [normalized["dataset"]] if "/" in normalized["dataset"] else []

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        normalized = self.validate_params(params)
        if normalized["dataset"]:
            slug = normalized["dataset"].split("/versions/", 1)[0].split("/")[-1]
            return {slug: slug}
        return {}

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("trocr_lines doesn't support sharding")
        return self.render_kaggle_code("trocr_lines_runner.py", normalized)
