"""SpotterPageBenchJob — page-level бенчмарк YOLO-спотера на повних сторінках.

Половина бенчмарку HTR-vs-Spotter: чи знаходить детектор словоформи
прізвища рід на ПОВНІЙ сторінці-скані і за який час. Тайлить сторінку, бере max
conf по тайлах → page-score; pred = score >= threshold.

Вхід — Kaggle Dataset:
  * ``images/*.png``        — повні сторінки-скани;
  * ``ground_truth.json``   — {filename: {label: pos|neg, ...}};
  * ``weights/<file>.pt``   — ultralytics ваги (YOLOv8/v11).

Вихід у ``/kaggle/working/``:
  * ``results_spotter.tsv``  — engine, filename, label, pred, score, t_infer_ms, n_tiles;
  * ``results_spotter.json`` — config + cold_start_s + per_page.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class SpotterPageBenchJob(Job):
    """Page-level surname-spotter benchmark (tiled YOLO, max-conf → page score)."""

    name: ClassVar[str] = "spotter_page_bench"
    description: ClassVar[str] = (
        "Page-level surname-spotter benchmark with ANCHOR-RANK: tiled YOLO (conf 0.01) "
        "-> top-K crops -> DINO embedding -> max cosine to anchor bank as page score. "
        "Outputs per-page pred/score/t_infer_ms."
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
        return ["ultralytics>=8.2", "pillow>=10.0", "transformers>=4.45",
                "torchvision", "safetensors"]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError("spotter_page_bench needs 'dataset=<owner>/<slug>'")
        dataset = str(params["dataset"]).strip()
        if "/" not in dataset:
            raise ValueError(f"dataset must be '<owner>/<slug>', got {dataset!r}")
        return {
            "dataset": dataset,
            "image_glob": str(params.get("image_glob", "images/*.png")).strip(),
            "ground_truth_file": str(params.get("ground_truth_file", "ground_truth.json")).strip(),
            "weights": str(params.get("weights", "weights/epoch13.pt")).strip(),
            "backbone_dir": str(params.get("backbone_dir", "anchor/dino_v4")).strip(),
            "bank_emb": str(params.get("bank_emb", "anchor/v4-backbone__gray.npz")).strip(),
            "tile": int(params.get("tile", 1024)),
            "overlap": float(params.get("overlap", 0.2)),
            "imgsz": int(params.get("imgsz", 1280)),
            "conf": float(params.get("conf", 0.01)),
            "top_k": int(params.get("top_k", 50)),
            "prep": str(params.get("prep", "gray")).strip(),
            "threshold": float(params.get("threshold", 0.55)),
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        n_pages = int(params.get("estimated_n_pages", 31))
        return timedelta(seconds=n_pages * 30 + 300)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["results_spotter.json", "results_spotter.tsv"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        return [params["dataset"]]

    def render_remote_code(self, params: dict[str, Any], *, shard_index: int = 0,
                           total_shards: int = 1) -> str:
        normalized = self.validate_params(params)
        if total_shards > 1:
            raise ValueError("spotter_page_bench doesn't support sharding")
        return self.render_kaggle_code("spotter_page_bench_runner.py", normalized)
