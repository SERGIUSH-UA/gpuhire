"""HTREvalJob — zero-shot бенчмарк HTR/vision-LLM моделей на розміченому сеті слів.

Кейс: чи читають готові моделі рукописне прізвище у вирізках 1796-1920, і чи
розділяє fuzzy-match транскрипції TP від ФП (вирішує, чи вмикати HTR-стадію в
spotter-конвеєр і чи потрібен fine-tune).

Вхід — Kaggle Dataset:
  * ``images/*.png``        — тісні кропи слів;
  * ``ground_truth.json``   — {filename: {label: tp|fp, era: str, text: str}}.

Вихід у ``/kaggle/working/``:
  * ``results.tsv``     — model, filename, label, era, gt_text, prediction, match;
  * ``results.json``    — по моделях: AUC розділення TP/FP (загалом і по епохах),
                          медіани match, elapsed;
  * ``inference_log.json``.

Підтримувані родини моделей (вибір по id): TrOCR (Vision2Seq), Qwen2-VL (chat),
Surya (surya-ocr, кілька версій API). Падіння однієї моделі не валить весь job.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job

DEFAULT_MODELS = (
    "raxtemur/trocr-base-ru",
    "kazars24/trocr-base-handwritten-ru",
    "Qwen/Qwen2-VL-2B-Instruct",
    "surya",
)


class HTREvalJob(Job):
    """Zero-shot HTR benchmark: транскрипція + fuzzy-match розділення TP/FP."""

    name: ClassVar[str] = "htr_eval"
    description: ClassVar[str] = (
        "Benchmark handwriting-OCR models (TrOCR/Qwen-VL/Surya) on a labeled word-crop "
        "dataset; reports TP/FP fuzzy-match separation per era."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    INFERENCE_SEC_PER_IMAGE: ClassVar[float] = 1.0  # консервативно (vision-LLM)

    def requirements(self) -> list[str]:
        return [
            "transformers>=4.45",
            "pillow>=10.0",
            "rapidfuzz>=3.0",
            "qwen-vl-utils",
            "accelerate>=0.30",
            "surya-ocr",
        ]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError("htr_eval needs 'dataset=<owner>/<slug>'")
        dataset = str(params["dataset"]).strip()
        if "/" not in dataset:
            raise ValueError(f"dataset must be '<owner>/<slug>', got {dataset!r}")

        models_raw = params.get("models", ",".join(DEFAULT_MODELS))
        models = (list(models_raw) if isinstance(models_raw, list)
                  else [m.strip() for m in str(models_raw).split(",") if m.strip()])
        if not models:
            raise ValueError("models must have at least one entry")

        # Шукане слово — дані користувача, дефолту для нього не буває.
        target = str(params.get("target") or "").strip().lower()
        if len(target) < 4:
            raise ValueError(f"target (шукане слово, від 4 літер) обов'язковий: -p target=<слово>; "
                             f"дано {target!r}")

        return {
            "dataset": dataset,
            "image_glob": str(params.get("image_glob", "images/*.png")).strip(),
            "ground_truth_file": str(params.get("ground_truth_file", "ground_truth.json")).strip(),
            "models": models,
            "target": target,
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        normalized = self.validate_params(params)
        n_images = int(params.get("estimated_n_images", 200))
        secs = n_images * self.INFERENCE_SEC_PER_IMAGE * len(normalized["models"])
        return timedelta(seconds=secs + 600)  # + завантаження моделей

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["results.json", "results.tsv", "inference_log.json"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        return [params["dataset"]]

    def render_remote_code(
        self,
        params: dict[str, Any],
        *,
        shard_index: int = 0,
        total_shards: int = 1,
    ) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("htr_eval doesn't support sharding")

        return self.render_kaggle_code("htr_eval_runner.py", normalized)
