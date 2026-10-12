"""CropVerifierJob — fine-tune a binary verifier for image crop candidates.

This is a semantic wrapper around the generic ViT binary classifier. The input
contract is the same: a Kaggle Dataset with ``train_labels.csv`` containing
``filename,label`` and image files referenced by ``filename``.

Label semantics for this job:
  * 1 — candidate crop contains the target word/surname.
  * 0 — candidate crop is a false positive or a visually similar non-target.
"""

from __future__ import annotations

from typing import Any, ClassVar

from gpurunner.jobs.vit_classifier import ViTClassifierJob


class CropVerifierJob(ViTClassifierJob):
    """Fine-tune a ViT binary classifier for candidate crop verification."""

    name: ClassVar[str] = "crop_verifier"
    description: ClassVar[str] = (
        "Fine-tune a ViT binary verifier for detector/review crops: "
        "target word vs visually similar false positives."
    )

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = {
            "train_csv": "train_labels.csv",
            "inference_glob": "images/*.png",
            "model": "google/vit-base-patch16-384",
            "image_size": 384,
            "epochs": 12,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "save_checkpoint": True,
            **params,
        }
        return super().validate_params(merged)
