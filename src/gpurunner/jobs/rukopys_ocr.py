"""RukopysOCRJob — сторінкове читання VLM'ом ebinan92/Rukopys-OCR-4B.

Кейс: чи знаходить модель конкурсу Handwritten to Data (Qwen3.5-4B,
вчена на рукописах 1919–2025) рід на метриках XIX ст. — замір тим самим
``clan_probe.py``, яким міряно Писаря й Дяка.

Вхід — Kaggle Dataset зі сторінками (*.jpg/*.png). Вихід у ``/kaggle/working/``:
``pages/<stem>.txt`` (рядки тексту), ``raw/<stem>.txt`` (сира відповідь),
``rukopys_results.json`` (токени, секунди, помітки провалу).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class RukopysOCRJob(Job):
    """Rukopys-OCR-4B page transcription (T4 fp16 / bf16 where supported)."""

    name: ClassVar[str] = "rukopys_ocr"
    description: ClassVar[str] = (
        "Transcribe handwritten pages with ebinan92/Rukopys-OCR-4B "
        "(Qwen3.5-4B) -> per-page txt + raw JSON + rukopys_results.json."
    )
    supported_backends: ClassVar[tuple[str, ...]] = ("kaggle", "colab", "vast")

    SEC_PER_PAGE: ClassVar[float] = 240.0  # до 8192 токенів на T4, не міряно

    def requirements(self) -> list[str]:
        return ["transformers>=5.8.1", "accelerate>=1.0", "pillow>=10.0"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        dataset = str(params.get("dataset", "")).strip()
        if not dataset:
            raise ValueError("rukopys_ocr needs 'dataset=<owner>/<slug>'")
        dtype = str(params.get("dtype", "auto")).strip().lower()
        if dtype not in ("auto", "float16", "bfloat16", "float32"):
            raise ValueError(f"dtype must be auto/float16/bfloat16/float32, got {dtype!r}")
        engine = str(params.get("engine", "hf")).strip().lower()
        if engine not in ("hf", "vllm"):
            raise ValueError(f"engine must be hf/vllm, got {engine!r}")
        return {
            "dataset": dataset,
            "model": str(params.get("model", "ebinan92/Rukopys-OCR-4B")).strip(),
            # Стеля з картки моделі (vLLM-приклад): 4096 візуальних токенів.
            "max_pixels": int(params.get("max_pixels", 4096 * 32 * 32)),
            "max_new_tokens": int(params.get("max_new_tokens", 8192)),
            "dtype": dtype,
            # flash-linear-attention: без нього лінійна увага — torch-шлях, ~9 ток/с на T4
            "accel": bool(params.get("accel", True)),
            # hf = transformers.generate (Kaggle T4); vllm = образ vllm/vllm-openai
            # на орендованій карті, усі сторінки одним батчем, без власних стопів
            "engine": engine,
            "gpu_mem_util": float(params.get("gpu_mem_util", 0.9)),
            "max_num_seqs": int(params.get("max_num_seqs", 16)),
            "skip_existing": bool(params.get("skip_existing", True)),
            # кома-список stem'ів для вибіркового прогону; "" = всі
            "pages": str(params.get("pages", "")).strip(),
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        n_pages = int(params.get("estimated_n_pages", 90))
        return timedelta(seconds=n_pages * self.SEC_PER_PAGE / 2 + 900)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["rukopys_results.json"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        return [params["dataset"]]

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("rukopys_ocr doesn't support sharding")
        return self.render_kaggle_code("rukopys_ocr_runner.py", normalized)
