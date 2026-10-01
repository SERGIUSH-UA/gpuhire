"""Concrete Job implementations."""

from gpurunner.core.job import Job
from gpurunner.jobs._net_probe import NetProbeJob
from gpurunner.jobs.churro import ChurroJob
from gpurunner.jobs.crop_verifier import CropVerifierJob
from gpurunner.jobs.dino_surname_verifier import DinoSurnameVerifierJob
from gpurunner.jobs.htr_case import HTRCaseJob
from gpurunner.jobs.htr_eval import HTREvalJob
from gpurunner.jobs.htr_lines_eval import HTRLinesEvalJob
from gpurunner.jobs.htr_page_bench import HTRPageBenchJob
from gpurunner.jobs.kraken_lines import KrakenLinesJob
from gpurunner.jobs.kraken_train import KrakenTrainJob
from gpurunner.jobs.paddle_page_bench import PaddlePageBenchJob
from gpurunner.jobs.paddleocr import PaddleOCRJob
from gpurunner.jobs.parseq_train import ParseqTrainJob
from gpurunner.jobs.rukopys_ocr import RukopysOCRJob
from gpurunner.jobs.spotter_page_bench import SpotterPageBenchJob
from gpurunner.jobs.trocr_lines import TrOCRLinesJob
from gpurunner.jobs.vit_classifier import ViTClassifierJob
from gpurunner.jobs.yolo_spotter import YoloSpotterJob

__all__ = [
    "ChurroJob",
    "CropVerifierJob",
    "DinoSurnameVerifierJob",
    "HTRCaseJob",
    "HTREvalJob",
    "HTRLinesEvalJob",
    "HTRPageBenchJob",
    "KrakenLinesJob",
    "KrakenTrainJob",
    "NetProbeJob",
    "PaddleOCRJob",
    "PaddlePageBenchJob",
    "ParseqTrainJob",
    "RukopysOCRJob",
    "SpotterPageBenchJob",
    "TrOCRLinesJob",
    "ViTClassifierJob",
    "YoloSpotterJob",
    "get_job",
    "list_jobs",
]


def get_job(name: str) -> type[Job]:
    """Resolve a job name to its class."""
    table: dict[str, type[Job]] = {
        "paddleocr": PaddleOCRJob,
        "net-probe": NetProbeJob,
        "churro": ChurroJob,
        "kraken_lines": KrakenLinesJob,
        "kraken_train": KrakenTrainJob,
        "vit_classifier": ViTClassifierJob,
        "crop_verifier": CropVerifierJob,
        "dino_surname_verifier": DinoSurnameVerifierJob,
        "yolo_spotter": YoloSpotterJob,
        "htr_case": HTRCaseJob,
        "htr_eval": HTREvalJob,
        "htr_page_bench": HTRPageBenchJob,
        "paddle_page_bench": PaddlePageBenchJob,
        "spotter_page_bench": SpotterPageBenchJob,
        "trocr_lines": TrOCRLinesJob,
        "htr_lines_eval": HTRLinesEvalJob,
        "parseq_train": ParseqTrainJob,
        "rukopys_ocr": RukopysOCRJob,
    }
    if name not in table:
        raise KeyError(f"Unknown job {name!r}. Available: {sorted(table.keys())}")
    return table[name]


def list_jobs() -> list[type[Job]]:
    return [
        PaddleOCRJob,
        NetProbeJob,
        ChurroJob,
        KrakenLinesJob,
        KrakenTrainJob,
        ViTClassifierJob,
        CropVerifierJob,
        DinoSurnameVerifierJob,
        YoloSpotterJob,
        HTRCaseJob,
        HTREvalJob,
        HTRPageBenchJob,
        PaddlePageBenchJob,
        SpotterPageBenchJob,
        HTRLinesEvalJob,
        TrOCRLinesJob,
        ParseqTrainJob,
        RukopysOCRJob,
    ]
