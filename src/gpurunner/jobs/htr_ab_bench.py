"""HTRABBenchJob — A/B двох наборів моделей на ОДНІЙ машині (швидкість + тексти).

Навіщо окремий job: порівняти «бойове як було» (Писар v17 + Дяк + Скриба) із
«новим» (Писар v18 + Скриба + Літописець) так, щоб різниця в часі належала
МОДЕЛЯМ, а не машинам. Бойовий наглядач обирає машину сам і другий захід
посадив би на іншу; тут обидва проходи йдуть послідовно на тій самій карті, а
сегментація рахується раз і ділиться (кеш справи) — як у бойовому другому
прогоні іншою моделлю.

Вхід — теки, залиті на бокс (Vast: ``-p inputs='{"frames": "...", "models": "..."}'``):
``frames/<справа>/*.jpg`` і ``models/`` з вагами. Вихід — ``ab_bundle.tgz``
(тексти й мети прогонів кожного проходу), ``ab_timing.json``, ``ab.log``.
Якість рахується вдома, на стороні дослідження.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class HTRABBenchJob(Job):
    """A/B model sets on one box: per-stage timing + texts for quality."""

    name: ClassVar[str] = "htr_ab_bench"
    description: ClassVar[str] = (
        "A/B on ONE box: pass A (Pysar v17 + Diak + Skryba) vs pass B (Pysar v18 + "
        "Skryba) + Litopysets (kraken 7.1) on the same segmentation; timing + texts."
    )
    supported_backends: ClassVar[tuple[str, ...]] = ("vast",)

    def requirements(self) -> list[str]:
        # усе ставить раннер сам: бойовий конвеєр (`nysh htr install`) і окреме
        # середовище kraken 7.1 — у спільне середовище контейнера їх не покласти
        return []

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "nysh_version": str(params.get("nysh_version", "0.23.2")).strip(),
            "litopysets": str(params.get("litopysets", "litopysets_cyr_v1.safetensors")).strip(),
            "lit_batch": int(params.get("lit_batch", 8)),
            # які проходи (A,B,L) і які справи (кома-список тек; порожньо — усі)
            "passes": str(params.get("passes", "A,B,L")).strip(),
            "cases": str(params.get("cases", "")).strip(),
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        pages = int(params.get("estimated_n_pages", 600))
        # ~25 хв встановлення + два проходи по ~6 с/стор на V100 + Літописець
        return timedelta(seconds=1500 + pages * 14)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["ab_bundle.tgz", "ab_timing.json", "ab.log"]

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        return {"frames": "frames", "models": "models"}

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("htr_ab_bench doesn't support sharding")
        return self.render_kaggle_code("htr_ab_bench_runner.py", normalized)
