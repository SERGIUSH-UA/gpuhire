"""HTRReleaseBenchJob — замір релізу: стара й нова збірка конвеєра на ОДНІЙ машині.

Навіщо: цифри для релізу мусять бути чистими — обидві збірки читають ті самі
кадри на тій самій карті, по черзі, без сусідніх процесів. Прохід A — nyshporka
з PyPI зі старими моделями, прохід B — wheel нової збірки з новими; після B —
голоси Писаря на голд-сеті. Якість і швидкість рахуються вдома з `release_bundle.tgz`.

Вхід — теки на боксі (Vast: ``-p inputs='{"frames": …, "models_a": …, "models_b": …,
"wheel": …, "gold": …}'``).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class HTRReleaseBenchJob(Job):
    """Release bench: old vs new pipeline build on one box, + gold-set voices."""

    name: ClassVar[str] = "htr_release_bench"
    description: ClassVar[str] = (
        "Release bench on ONE box: pass A (PyPI nyshporka + old models) vs pass B "
        "(wheel + new models) on the same frames, then PARSeq voices on the gold set."
    )
    supported_backends: ClassVar[tuple[str, ...]] = ("vast",)

    def requirements(self) -> list[str]:
        return []

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "nysh_a": str(params.get("nysh_a", "0.23.2")).strip(),
            "passes": str(params.get("passes", "A,B")).strip(),
            "cases": str(params.get("cases", "")).strip(),
            "gold_models": str(params.get("gold_models", "pysar_cyr_v17.pt,pysar_cyr_v19.pt")).strip(),
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        pages = int(params.get("estimated_n_pages", 250))
        # два встановлення ~15 хв + два проходи ~12 с/стор + голд-сет ~30 хв
        return timedelta(seconds=900 + pages * 24 + 1800)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["release_bundle.tgz", "release_timing.json", "release.log"]

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        return {"frames": "frames", "models_a": "models_a", "models_b": "models_b",
                "wheel": "wheel", "gold": "gold"}

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("htr_release_bench doesn't support sharding")
        return self.render_kaggle_code("htr_release_bench_runner.py", normalized)
