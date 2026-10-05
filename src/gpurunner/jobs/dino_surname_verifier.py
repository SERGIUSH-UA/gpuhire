"""DinoSurnameVerifierJob — DINOv2 binary verifier for surname crop candidates."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from gpurunner.core.job import Job


class DinoSurnameVerifierJob(Job):
    """Fine-tune facebook/dinov2-base with a binary classifier head."""

    name: ClassVar[str] = "dino_surname_verifier"
    description: ClassVar[str] = (
        "Fine-tune a DINOv2 binary verifier for surname detector crops with a "
        "separate holdout gate."
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
        return [
            "transformers>=4.40",
            "scikit-learn>=1.4",
            "pillow>=10.0",
            "pandas>=2.0",
        ]

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "dataset" not in params:
            raise ValueError(
                "dino_surname_verifier needs 'dataset=<owner>/<slug>' — a Kaggle Dataset to mount"
            )
        dataset = str(params["dataset"]).strip()
        if "/" not in dataset:
            raise ValueError(f"dataset must be '<owner>/<slug>', got {dataset!r}")

        train_csv = str(params.get("train_csv", "train_labels.csv")).strip()
        holdout_csv = str(params.get("holdout_csv", "holdout_labels.csv")).strip()
        inference_glob = str(params.get("inference_glob", "images/*.png")).strip()
        model = str(params.get("model", "facebook/dinov2-base")).strip()

        epochs = int(params.get("epochs", 10))
        if not 1 <= epochs <= 40:
            raise ValueError(f"epochs must be in [1, 40], got {epochs}")

        batch_size = int(params.get("batch_size", 16))
        if not 1 <= batch_size <= 64:
            raise ValueError(f"batch_size must be in [1, 64], got {batch_size}")

        val_split = float(params.get("val_split", 0.15))
        if not 0.0 <= val_split <= 0.5:
            raise ValueError(f"val_split must be in [0, 0.5], got {val_split}")

        learning_rate = float(params.get("learning_rate", 3e-5))
        if not 1e-6 <= learning_rate <= 1e-2:
            raise ValueError(f"learning_rate out of range: {learning_rate}")

        image_size = int(params.get("image_size", 384))
        if image_size not in (224, 256, 384):
            raise ValueError(f"image_size must be 224/256/384, got {image_size}")

        seed = int(params.get("seed", 42))
        save_checkpoint = bool(params.get("save_checkpoint", True))
        checkpoint_metric = str(params.get("checkpoint_metric", "val_acc")).strip()
        if checkpoint_metric not in {"val_acc", "recall_safe"}:
            raise ValueError("checkpoint_metric must be 'val_acc' or 'recall_safe'")
        augment_profile = str(params.get("augment_profile", "default")).strip()
        if augment_profile not in {"default", "paper_color"}:
            raise ValueError("augment_profile must be 'default' or 'paper_color'")

        return {
            "dataset": dataset,
            "train_csv": train_csv,
            "holdout_csv": holdout_csv,
            "inference_glob": inference_glob,
            "model": model,
            "epochs": epochs,
            "batch_size": batch_size,
            "val_split": val_split,
            "learning_rate": learning_rate,
            "image_size": image_size,
            "seed": seed,
            "save_checkpoint": save_checkpoint,
            "checkpoint_metric": checkpoint_metric,
            "augment_profile": augment_profile,
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        normalized = self.validate_params(params)
        n_train = int(params.get("estimated_n_train", 5768))
        n_infer = int(params.get("estimated_n_inference", 5778))
        train_batches = n_train / normalized["batch_size"] * normalized["epochs"]
        return timedelta(seconds=train_batches * 0.9 + n_infer * 0.07 + 420)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return [
            "model_checkpoint",
            "model_checkpoints_top3",
            "model_checkpoints_by_epoch",
            "predictions.csv",
            "training_log.json",
            "validation_report.json",
            "holdout_report.json",
            "meta.json",
        ]

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
            raise ValueError("dino_surname_verifier doesn't support sharding")

        # Цей раннер — argparse-CLI, а не main(params): його __main__ і є точкою
        # входу, тому блок НЕ зрізається і власного entry ми не дописуємо.
        # Замість цього підмінюємо sys.argv (у Kaggle-клітинці там аргументи
        # ipykernel, які argparse відкинув би з SystemExit(2)).
        argv_bridge = (
            "import sys\n"
            "sys.argv = ['dino_surname_verifier_runner.py',\n"
            "    '--train-csv', PARAMS['train_csv'],\n"
            "    '--holdout-csv', PARAMS['holdout_csv'],\n"
            "    '--inference-glob', PARAMS['inference_glob'],\n"
            "    '--model', PARAMS['model'],\n"
            "    '--epochs', str(PARAMS['epochs']),\n"
            "    '--batch-size', str(PARAMS['batch_size']),\n"
            "    '--image-size', str(PARAMS['image_size']),\n"
            "    '--learning-rate', str(PARAMS['learning_rate']),\n"
            "    '--val-split', str(PARAMS['val_split']),\n"
            "    '--checkpoint-metric', PARAMS['checkpoint_metric'],\n"
            "    '--augment-profile', PARAMS['augment_profile'],\n"
            "    '--seed', str(PARAMS['seed']),\n"
            "    '--out', '/kaggle/working']"
        )
        return self.render_kaggle_code(
            "dino_surname_verifier_runner.py",
            normalized,
            prelude=argv_bridge,
            entry="",
            strip_main=False,
        )
