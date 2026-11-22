"""ChurroJob — транскрипція історичних сторінок VLM'ом stanford-oval/churro-3B.

Кейс: пілот «чи читає CHURRO польський канцелярський курсив 1802 та
кириличні метрики краще за Kraken/McCATMuS» (спр. ДАВіО ф.792-1-17 та ін.).

Вхід — Kaggle Dataset зі сторінками (*.jpg/*.png, flat чи в підтеках).
Вихід у ``/kaggle/working/``: ``<stem>.txt`` на сторінку + ``churro_results.json``
(cold_start, per-page chars/sec/error).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class ChurroJob(Job):
    """CHURRO-3B historical-page transcription (fp16, T4)."""

    name: ClassVar[str] = "churro"
    description: ClassVar[str] = (
        "Transcribe historical document pages with stanford-oval/churro-3B "
        "(Qwen2.5-VL base) -> per-page txt + churro_results.json."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    SEC_PER_PAGE: ClassVar[float] = 60.0  # 3B VLM, ~2-3k токенів на щільну сторінку

    def requirements(self) -> list[str]:
        return [
            # Qwen2_5_VL* класи з'явились у transformers 4.49
            "transformers>=4.49",
            "qwen-vl-utils",
            "accelerate>=0.30",
            "pillow>=10.0",
        ]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError("churro needs 'dataset=<owner>/<slug>'")
        dataset = str(params["dataset"]).strip()
        # На Kaggle це '<owner>/<slug>'; на Lightning/vast — просто ім'я підтеки
        # в input_root, тому слеш обов'язковим бути не може.
        if not dataset:
            raise ValueError("dataset must be non-empty")
        mode = str(params.get("mode", "both")).strip().lower()
        if mode not in ("page", "line", "both"):
            raise ValueError(f"mode must be page/line/both, got {mode!r}")
        line_batch = int(params.get("line_batch", 16))
        if line_batch < 1:
            raise ValueError("line_batch must be >= 1")
        return {
            "dataset": dataset,
            "model": str(params.get("model", "stanford-oval/churro-3B")).strip(),
            "max_pixels": int(params.get("max_pixels", 1254400)),
            "max_new_tokens": int(params.get("max_new_tokens", 3072)),
            # Режим входу. Кропи розпізнаються за іменем `line_NNN.png` усередині
            # теки сторінки, решта картинок — сторінки; `both` жене і те, й те в
            # одній GPU-сесії (холодний старт платиться раз, стан моделі спільний).
            "mode": mode,
            # Стеля токенів на КРОП. 3072 для рядка — не запобіжник, а дозвіл на
            # луп у 3000 токенів; медіана рядка метрики — 17 символів.
            "line_max_new_tokens": int(params.get("line_max_new_tokens", 64)),
            "line_max_pixels": int(params.get("line_max_pixels", 200704)),
            "line_batch": line_batch,
            "skip_existing": bool(params.get("skip_existing", True)),
            # кома-список stem'ів для вибіркового прогону ("01600,03100"); "" = всі
            "pages": str(params.get("pages", "")).strip(),
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        n_pages = int(params.get("estimated_n_pages", 12))
        return timedelta(seconds=n_pages * self.SEC_PER_PAGE + 900)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["churro_results.json"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        return [params["dataset"]]

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("churro doesn't support sharding")
        return self.render_kaggle_code("churro_runner.py", normalized)
