"""ViTClassifierJob — fine-tune a binary image classifier (ViT) and run inference.

Use case: filter title pages of a multi-thousand-page archival PDF when no OCR
can read the handwritten script. Train on a small hand-labeled set (~100
positives) from one fond, infer on related fond pages.

Inputs are read from a Kaggle Dataset mounted at ``/kaggle/input/<slug>/``:
  * ``train_csv``       — CSV with two cols: ``filename, label`` (0 or 1).
                          Paths in ``filename`` are relative to the dataset root.
  * ``inference_glob``  — glob (relative to dataset root) of unlabeled images
                          to score. Predictions written to ``predictions.csv``.

Outputs in ``/kaggle/working/``:
  * ``predictions.csv``     — filename, score, label_at_threshold
  * ``training_log.json``   — per-epoch loss/accuracy
  * ``validation_report.json`` — per-threshold precision/recall/F1
  * ``model_checkpoint.pt`` — saved fine-tuned state_dict (optional)

The job is intentionally domain-agnostic: any binary image classification task
fits (parish-title pages, blank-vs-filled forms, decorated covers, …). Only the
dataset and label semantics change.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class ViTClassifierJob(Job):
    """Fine-tune a HuggingFace ViT for binary image classification + run inference."""

    name: ClassVar[str] = "vit_classifier"
    description: ClassVar[str] = (
        "Fine-tune a ViT binary image classifier on a labeled set from a Kaggle "
        "Dataset, then score an unlabeled glob from the same dataset."
    )
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "modal",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    # Rough throughput on Kaggle T4 for ViT-base-224. Used by estimate_runtime.
    TRAIN_SEC_PER_BATCH: ClassVar[float] = 0.5
    INFERENCE_SEC_PER_IMAGE: ClassVar[float] = 0.04

    def requirements(self) -> list[str]:
        # Used for Kaggle (stock kernel image already has torch+torchvision).
        # For Modal we use modal_image_spec() instead — see below.
        return [
            "transformers>=4.40",
            "scikit-learn>=1.4",
            "pillow>=10.0",
            "pandas>=2.0",
        ]

    def modal_image_spec(self) -> dict[str, Any]:
        # Modal containers start from debian_slim with no torch — install
        # everything we need (CUDA-enabled torch is auto-selected when the
        # function is decorated with gpu=...).
        # python_version must match the local interpreter that pickles the
        # function (gpurunner requires 3.13).
        return {
            "python_version": "3.12",
            "pip_packages": [
                "torch>=2.2",
                "torchvision>=0.17",
                "transformers>=4.40",
                "scikit-learn>=1.4",
                "pillow>=10.0",
                "pandas>=2.0",
                "numpy>=1.26",
                "kaggle>=1.6",
            ],
            "extra_index_url": None,
            "apt_packages": [],
        }

    def modal_secrets(self, params: dict[str, Any]) -> list[str]:
        # Modal needs Kaggle credentials to download the dataset at runtime.
        # User creates this once:
        #   modal secret create kaggle-creds KAGGLE_KEY=KGAT_...
        return [str(params.get("kaggle_secret", "kaggle-creds"))]

    def render_runner_module(self) -> str:
        from importlib import resources

        return (
            resources.files("gpurunner._embedded")
            .joinpath("vit_classifier_modal_runner.py")
            .read_text(encoding="utf-8")
        )

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError(
                "vit_classifier needs 'dataset=<owner>/<slug>' — a Kaggle Dataset to mount"
            )
        dataset = str(params["dataset"]).strip()
        if "/" not in dataset:
            raise ValueError(f"dataset must be '<owner>/<slug>', got {dataset!r}")

        train_csv = str(params.get("train_csv", "train_labels.csv")).strip()
        inference_glob = str(params.get("inference_glob", "inference/*.png")).strip()
        if not inference_glob:
            raise ValueError("inference_glob must be non-empty")

        model = str(params.get("model", "google/vit-base-patch16-224")).strip()
        epochs = int(params.get("epochs", 8))
        if not 1 <= epochs <= 40:
            raise ValueError(f"epochs must be in [1, 40], got {epochs}")

        batch_size = int(params.get("batch_size", 16))
        if not 1 <= batch_size <= 128:
            raise ValueError(f"batch_size must be in [1, 128], got {batch_size}")

        val_split = float(params.get("val_split", 0.15))
        if not 0.0 <= val_split <= 0.5:
            raise ValueError(f"val_split must be in [0, 0.5], got {val_split}")

        learning_rate = float(params.get("learning_rate", 5e-5))
        if not 1e-6 <= learning_rate <= 1e-2:
            raise ValueError(f"learning_rate out of range: {learning_rate}")

        image_size = int(params.get("image_size", 224))
        if image_size not in (224, 256, 384):
            raise ValueError(f"image_size must be 224/256/384, got {image_size}")

        save_checkpoint = bool(params.get("save_checkpoint", False))
        seed = int(params.get("seed", 42))

        decision_threshold = float(params.get("decision_threshold", 0.5))
        if not 0.0 < decision_threshold < 1.0:
            raise ValueError(f"decision_threshold must be in (0, 1), got {decision_threshold}")

        return {
            "dataset": dataset,
            "train_csv": train_csv,
            "inference_glob": inference_glob,
            "model": model,
            "epochs": epochs,
            "batch_size": batch_size,
            "val_split": val_split,
            "learning_rate": learning_rate,
            "image_size": image_size,
            "save_checkpoint": save_checkpoint,
            "seed": seed,
            "decision_threshold": decision_threshold,
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        # Very rough: assume 2000 training samples + 2000 inference samples.
        n = params.get("estimated_n_train", 2000)
        normalized = self.validate_params(params)
        train_batches = n / normalized["batch_size"] * normalized["epochs"]
        train_s = train_batches * self.TRAIN_SEC_PER_BATCH
        infer_n = params.get("estimated_n_inference", 2000)
        infer_s = infer_n * self.INFERENCE_SEC_PER_IMAGE
        # +5 min for env init + model download
        return timedelta(seconds=train_s + infer_s + 300)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return ["predictions.csv", "training_log.json", "validation_report.json"]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        # ``params`` here is already-normalized when called from KaggleBackend.submit.
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
            raise ValueError("vit_classifier doesn't support sharding")

        return self.render_kaggle_code("vit_classifier_runner.py", normalized)
