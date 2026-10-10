"""HTRLinesEvalJob — N HTR-моделей (TrOCR/PARSeq) по готових кропах рядків.

Кейс: відбір вчителя для кириличного kraken — прогнати кандидатів
на ОДНИХ і тих самих кропах (сегментація з job'а trocr_lines) і порівняти
phrase-recall на eye-verified сторінках локальним eval'ом.

Вхід — lines.tgz (lines/<stem>/line_NNN.png) через input_root+dataset.
Вихід — ``htr_eval_bundle.tgz`` (<model_slug>/<stem>.txt) + summary.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job

DEFAULT_MODELS = (
    "trocr:cyrillic-trocr/trocr-ukrainian-handwritten@h128,"
    "trocr:cyrillic-trocr/trocr-church-slavonic-handwritten,"
    "parseq:Hukyl/parseq-s-cyrillic-handwritten"
)


class HTRLinesEvalJob(Job):
    """Multi-model HTR eval over precomputed line crops (T4)."""

    name: ClassVar[str] = "htr_lines_eval"
    description: ClassVar[str] = (
        "Run several HTR models (trocr:<repo>[@hNNN] / parseq:<repo>) over "
        "precomputed line crops -> per-model per-page txt bundle."
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
        return ["transformers>=4.45", "pillow>=10.0"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        dataset = str(params.get("dataset", "")).strip()
        models = str(params.get("models", DEFAULT_MODELS)).strip()
        for spec in models.split(","):
            engine = spec.strip().partition(":")[0]
            if engine not in ("trocr", "parseq"):
                raise ValueError(f"unknown engine {engine!r} in {spec!r} "
                                 "(want trocr:<repo> or parseq:<repo>)")
        return {
            "dataset": dataset,
            "models": models,
            "max_new_tokens": int(params.get("max_new_tokens", 96)),
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        n_models = len(self.validate_params(params)["models"].split(","))
        return timedelta(seconds=n_models * 600 + 900)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["htr_eval_bundle.tgz", "htr_eval_summary.json"]

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
            raise ValueError("htr_lines_eval doesn't support sharding")
        return self.render_kaggle_code("htr_lines_eval_runner.py", normalized)
