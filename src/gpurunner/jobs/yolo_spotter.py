"""YoloSpotterJob — fine-tune YOLO for handwriting word spotting on archival scans.

Reuses the existing ``_embedded/yolo_spotter_modal_runner.py`` for the Modal
backend (volume-based dataset staging). For Kaggle we render an inline notebook
body that extracts the dataset archive from ``/kaggle/input/<slug>/``, calls
Ultralytics train, and leaves weights under ``/kaggle/working/runs/...``.

Required params (train mode):
  * ``dataset`` — Kaggle Dataset id ``<owner>/<slug>`` (Kaggle backend),
                  or pre-uploaded Modal volume entry ``dataset_name`` for Modal.
  * ``mode``    — ``"train"`` (only mode currently supported via job framework;
                  ``infer`` is a local-only path, outside the job framework).

Optional params:
  * ``epochs`` (default 50), ``batch_size`` (16), ``imgsz`` (640)
  * ``base_model`` (``yolov8s.pt``/``yolo11m.pt``/``yolo26s.pt`` or a path under
    the Modal volume), ``val_split`` (0.10), ``seed`` (42)
  * ``modal_volume`` — Modal volume with datasets/checkpoints (default: env
    ``GPURUNNER_MODAL_VOLUME_YOLO_SPOTTER`` / ``GPURUNNER_MODAL_VOLUME``,
    else ``gpurunner-spotter``)
  * ``run_id`` (auto-generated if omitted)
  * ``audit_pack`` — optional Modal volume audit archive under ``/vol/audits``.
    When set with ``audit_every>0``, the runner prints per-epoch real recall
    metrics to Modal logs and writes ``realtime_audit.jsonl``.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from importlib import resources
from typing import Any, ClassVar

from gpurunner.config import modal_volume_name
from gpurunner.core.job import Job


class YoloSpotterJob(Job):
    """Fine-tune YOLO for handwriting word detection (single class)."""

    name: ClassVar[str] = "yolo_spotter"
    description: ClassVar[str] = (
        "Fine-tune YOLO for single-class handwriting word detection. "
        "Dataset = YOLO-format tar.gz with images/{train,val} + labels/{train,val} + data.yaml."
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

    SEC_PER_EPOCH_ON_T4: ClassVar[float] = 30.0  # rough, scaled by dataset size

    # Виміряні it/s (train-v34 imgsz1280 cache=disk, бенчі 2026-06-30) для наперед-прорахунку
    # часу/вартості. it/s падає з batch (більше img/iter); throughput плато ~77 img/s з batch32.
    # GPU-ключі = gpurunner-назви (_GPU_MAP). Нема ключа → фолбек-масштабування у estimate_runtime.
    MEASURED_IT_S: ClassVar[dict[str, dict[int, float]]] = {
        "T4": {8: 2.2},
        "L4": {8: 3.1},                              # cache не міняє (compute-bound)
        "A100": {8: 8.1, 16: 4.0, 32: 2.5, 64: 1.5},
        "A100-80GB": {8: 8.1, 16: 4.0, 32: 2.5, 64: 1.5},
    }
    N_TRAIN_DEFAULT: ClassVar[int] = 19311           # train-v34 тайлів
    ONE_TIME_OVERHEAD_S: ClassVar[float] = 135.0     # extract+cache+scan+plots (one-time)
    VAL_SEC_PER_EPOCH: ClassVar[float] = 6.0

    # Готові рецепти запуску (USER 2026-06-30, доведено бенчами). Друкуються в `gpurunner recipes`.
    RECIPES: ClassVar[dict[str, dict]] = {
        # швидко+дешево: A100 batch32 +cpu8 = найдешевший І найшвидший (реал $6.90/2.13год, 30еп)
        "fast": {"backend": "modal", "gpu": "A100",
                 "params": {"cpu": 8, "batch_size": 32, "epochs": 30, "imgsz": 1280, "save_period": 5}},
        # безкоштовно: Kaggle T4 (A100 НЕМАЄ); повільно ~6-9год але $0/квота. cpu/cache не керуються.
        "free": {"backend": "kaggle", "gpu": "T4",
                 "params": {"batch_size": 8, "epochs": 30, "imgsz": 1280, "save_period": 5}},
    }

    def requirements(self) -> list[str]:
        return [
            # >=8.4: підтримка YOLO26 (n/s/m/l/x, e2e NMS-free head). Kaggle і так
            # ставить найсвіжішу в межах пінів при кожному запуску кернела.
            "ultralytics>=8.4,<9.0",
            "pyyaml>=6.0",
            "pillow>=10.0",
        ]

    def modal_image_spec(self) -> dict[str, Any]:
        return {
            "python_version": "3.12",
            "pip_packages": [
                "torch>=2.2",
                "torchvision",
                "ultralytics>=8.4,<9.0",
                "opencv-python-headless",
                "pillow",
                "numpy",
                "pyyaml",
            "tqdm",
            ],
            "extra_index_url": None,
            "apt_packages": ["libgl1", "libglib2.0-0"],
            "timeout": 10 * 60 * 60,
            # NB: cpu НЕ задаємо дефолтом. Бенч 2026-06-30 (реальні рахунки) показав:
            # cache='disk' прискорює ЛИШЕ швидкі картки (A100 3.5→8.1 it/s), а L4/T4
            # compute-bound (L4 з cache=3.0 ≈ без cache 3.1 → cache не діє). cpu=8 на
            # повільній картці МАРНИЙ і ШКІДЛИВИЙ: тягне ~31GB RAM → L4-трен дорожчає
            # $7.3→$10.2. Тому cpu — OPT-IN: `-p cpu=8` лише для A100+/H100, коли час
            # критичний (A100 ~2.7год ~$8.2 vs L4 ~6.5год ~$7.3). Деталі: memory
            # spotter-v35-dataloader-tuning-plan.
        }

    def modal_input_volumes(self, params: dict[str, Any]) -> dict[str, str]:
        # The embedded runner expects datasets/checkpoints under /vol, staged
        # out-of-band via `modal volume put <volume> ...`.
        return {"/vol": str(params.get("modal_volume")
                            or modal_volume_name(self.name, "gpurunner-spotter"))}

    def modal_secrets(self, params: dict[str, Any]) -> list[str]:
        return []  # Modal backend uses its own volume, no Kaggle creds needed

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        mode = str(params.get("mode", "train")).lower()
        if mode != "train":
            raise ValueError(f"yolo_spotter via gpurunner currently only supports mode='train', got {mode!r}")

        dataset = params.get("dataset") or params.get("dataset_name")
        if not dataset:
            raise ValueError(
                "yolo_spotter needs 'dataset' (Kaggle: '<owner>/<slug>'; Modal: '<name>' staged in volume)"
            )
        dataset = str(dataset).strip()

        epochs = int(params.get("epochs", 50))
        if not 1 <= epochs <= 500:
            raise ValueError(f"epochs in [1, 500], got {epochs}")

        batch_size = int(params.get("batch_size", 16))
        if not 1 <= batch_size <= 128:
            raise ValueError(f"batch_size in [1, 128], got {batch_size}")

        imgsz = int(params.get("imgsz", 640))
        if imgsz not in (320, 416, 512, 640, 768, 896, 1024, 1280):
            raise ValueError(f"imgsz must be a YOLO-standard size, got {imgsz}")

        base_model = str(params.get("base_model", "yolov8s.pt")).strip()
        builtin_model = re.fullmatch(r"(?:yolov8[nslmx]|yolo11[nslmx]|yolo26[nslmx])\.pt", base_model)
        volume_checkpoint = base_model.startswith("runs/") and base_model.endswith(".pt")
        if not (builtin_model or volume_checkpoint):
            raise ValueError(
                "base_model must be yolov8{n,s,m,l,x}.pt, yolo11{n,s,m,l,x}.pt, yolo26{n,s,m,l,x}.pt, "
                f"or a Modal-volume checkpoint path like runs/<run>/weights/last.pt; got {base_model!r}"
            )

        val_split = float(params.get("val_split", 0.10))
        if not 0.0 <= val_split <= 0.5:
            raise ValueError(f"val_split in [0, 0.5], got {val_split}")

        seed = int(params.get("seed", 42))
        run_id = str(params.get("run_id", "")).strip() or None
        save_period = int(params.get("save_period", -1))

        # data-loading тюнінг (фікс data-bound боттлнеку на imgsz1280). cache='disk'
        # декодує PNG→.npy раз; 'ram' НЕ для великих датасетів (65GB>RAM); '' (порожній)
        # = вимкнено. workers — паралельні data-loader'и (ефективні лише з cpu↑ у контейнері;
        # на Kaggle CPU фіксований ~2 ядра → тримати низьким).
        cache = str(params.get("cache", "disk")).strip().lower()
        if cache in ("none", "false", "0"):
            cache = ""
        if cache not in ("", "disk", "ram"):
            raise ValueError(f"cache must be one of '', 'disk', 'ram', got {cache!r}")
        workers = int(params.get("workers", 12))
        if not 0 <= workers <= 32:
            raise ValueError(f"workers in [0, 32], got {workers}")

        # device: 'auto' = усі GPU машини (Kaggle-«T4» насправді T4×2 → DDP на обох,
        # 2026-07-21); '0' — форс однієї картки (репро старих тренів / дебаг DDP).
        device = str(params.get("device", "auto")).strip().lower()
        if device not in ("auto", "0"):
            raise ValueError(f"device must be 'auto' or '0', got {device!r}")

        # cpu/memory — OPT-IN ресурси контейнера (Modal-бекенд). Дефолт = Modal-авто
        # (дешево, добре для compute-bound T4/L4). Підіймати ЛИШЕ для A100+/H100, коли
        # час критичний і dataloader-augmentation голодує швидкий GPU. cpu тягне RAM
        # → на повільній картці лише дорожчає (див. бенч-нотатку в modal_image_spec).
        raw_cpu = params.get("cpu")
        cpu = float(raw_cpu) if raw_cpu is not None and raw_cpu not in ("", 0, "0") else None
        if cpu is not None and not 0.5 <= cpu <= 64:
            raise ValueError(f"cpu in [0.5, 64], got {cpu}")
        raw_mem = params.get("memory")
        memory = int(raw_mem) if raw_mem is not None and raw_mem not in ("", 0, "0") else None
        if memory is not None and not 512 <= memory <= 131072:
            raise ValueError(f"memory (MiB) in [512, 131072], got {memory}")
        audit_pack = str(params.get("audit_pack", "")).strip() or None
        audit_every = int(params.get("audit_every", 0))
        if not 0 <= audit_every <= 100:
            raise ValueError(f"audit_every in [0, 100], got {audit_every}")
        audit_conf = float(params.get("audit_conf", 0.05))
        if not 0.0 < audit_conf <= 1.0:
            raise ValueError(f"audit_conf in (0, 1], got {audit_conf}")
        audit_infer_conf = float(params.get("audit_infer_conf", 0.01))
        if not 0.0 < audit_infer_conf <= 1.0:
            raise ValueError(f"audit_infer_conf in (0, 1], got {audit_infer_conf}")
        audit_hard_neg_limit = int(params.get("audit_hard_neg_limit", 20))
        if not 0 <= audit_hard_neg_limit <= 1000:
            raise ValueError(f"audit_hard_neg_limit in [0, 1000], got {audit_hard_neg_limit}")

        return {
            "mode": mode,
            "dataset": dataset,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "base_model": base_model,
            "val_split": val_split,
            "seed": seed,
            "run_id": run_id,
            "save_period": save_period,
            "cache": cache,
            "workers": workers,
            "device": device,
            "cpu": cpu,
            "memory": memory,
            "audit_pack": audit_pack,
            "audit_every": audit_every,
            "audit_conf": audit_conf,
            "audit_infer_conf": audit_infer_conf,
            "audit_hard_neg_limit": audit_hard_neg_limit,
        }

    def estimate_runtime(self, params: dict[str, Any]) -> timedelta:
        normalized = self.validate_params(params)
        batch = normalized["batch_size"]
        n_train = int(params.get("estimated_n_train", self.N_TRAIN_DEFAULT))
        steps = max(1, n_train // batch)
        gpu = str(params.get("_gpu", "T4"))

        # it/s: точний вимір → масштабування від найближчого batch (it/s ≈ const_imgs/batch
        # для насиченого GPU) → груба T4-евристика, якщо GPU невідомий.
        table = self.MEASURED_IT_S.get(gpu)
        if table and batch in table:
            its = table[batch]
        elif table:
            b0 = min(table, key=lambda b: abs(b - batch))
            its = table[b0] * b0 / batch
        else:
            its = self.MEASURED_IT_S["T4"][8] * 8 / batch  # фолбек на T4-профіль

        sec_per_epoch = steps / max(0.1, its) + self.VAL_SEC_PER_EPOCH
        total = sec_per_epoch * normalized["epochs"] + self.ONE_TIME_OVERHEAD_S
        return timedelta(seconds=total)

    def expected_outputs(self, params: dict[str, Any]) -> list[str]:
        return [
            "runs/yolospot/weights/best.pt",
            "runs/yolospot/weights/last.pt",
            "runs/yolospot/results.csv",
            "metrics.json",
        ]

    def dataset_sources(self, params: dict[str, Any]) -> list[str]:
        normalized = self.validate_params(params)
        if "/" in normalized["dataset"]:
            return [normalized["dataset"]]
        return []

    def colab_input_dirs(self, params: dict[str, Any]) -> dict[str, str]:
        """Colab: accept both '<owner>/<slug>' (Kaggle habit) and a bare '<name>'.

        Either way the archive is staged into ``/kaggle/input/<slug>/`` from
        ``MyDrive/gpurunner/data/<slug>/``.
        """
        slug = str(self.validate_params(params)["dataset"]).split("/", 1)[-1]
        return {slug: slug}

    def render_runner_module(self) -> str:
        """Modal backend: ship the embedded runner with ``main(params)`` dispatch."""
        return (
            resources.files("gpurunner._embedded")
            .joinpath("yolo_spotter_modal_runner.py")
            .read_text(encoding="utf-8")
        )

    def render_remote_code(
        self,
        params: dict[str, Any],
        *,
        shard_index: int = 0,
        total_shards: int = 1,
    ) -> str:
        """Kaggle backend: inline notebook body that extracts dataset and trains."""
        if total_shards > 1:
            raise ValueError("yolo_spotter doesn't support sharding")
        normalized = self.validate_params(params)
        dataset_id = normalized["dataset"]
        # Kaggle needs '<owner>/<slug>' (that's what dataset_sources attaches); Colab
        # stages a bare '<name>' from Drive into /kaggle/input/<name>. Both end up as a
        # directory under /kaggle/input, which the extraction step below globs
        # recursively — so a bare name is fine here. A misspelled Kaggle ref surfaces
        # remotely as 'no archive under /kaggle/input' with a directory listing.
        dataset_slug = dataset_id.split("/", 1)[-1]
        params_json = json.dumps(normalized, ensure_ascii=False)

        return f"""# --- gpurunner injected params ---
PARAMS = __import__('json').loads(r'''{params_json}''')
DATASET_SLUG = {dataset_slug!r}

# --- bootstrap ---
import subprocess, sys, json, tarfile, zipfile, shutil
from pathlib import Path

# Pre-install torch 2.4.1+cu121 BEFORE any `import torch`. This build supports
# CUDA capabilities sm_60..sm_90 — covers both Kaggle's P100 (sm_60, default
# on some runs) and T4 (sm_75). Newer torch in the stock Kaggle image only
# supports sm_70+ which crashes on P100.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       '--index-url', 'https://download.pytorch.org/whl/cu121',
                       'torch==2.4.1', 'torchvision==0.19.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'ultralytics>=8.4,<9.0', 'opencv-python-headless'])

import torch
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), 'sm', torch.cuda.get_device_capability(0))

WORK = Path('/kaggle/working')
# Розпаковуємо датасет у /tmp, а НЕ в /kaggle/working — інакше розпакований dataset/images
# (тисячі файлів, ~800МБ) потрапляє в kernel OUTPUT і fetch best.pt захлинається на ньому
# (CLI пагінація + timeout). У /tmp він не в output → output = лише runs/ (ваги, малий).
DATA_ROOT = Path('/tmp/dataset')
DATA_ROOT.mkdir(exist_ok=True, parents=True)

# Kaggle монтує датасет під різними шляхами залежно від платформи:
# /kaggle/input/<slug>/  АБО  /kaggle/input/datasets/<owner>/<slug>/ (нова поведінка
# 2026-05). Тому шукаємо архів РЕКУРСИВНО по всьому /kaggle/input — надійно за будь-якого
# шляху монтування. (Раніше хардкод /kaggle/input/<slug> → no archive.)
INPUT = Path('/kaggle/input')
archives = sorted(list(INPUT.rglob('*.tgz')) + list(INPUT.rglob('*.tar.gz')) + list(INPUT.rglob('*.zip')))
if not archives:
    files = [str(p) for p in INPUT.rglob('*')][:50]
    raise RuntimeError(f'no archive under {{INPUT}}: {{files}}')
archive = archives[0]
print(f'Extracting {{archive.name}} ({{archive.stat().st_size/1e6:.1f}} MB)...', flush=True)
if archive.suffix == '.zip':
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_ROOT)
else:
    with tarfile.open(archive) as tf:
        tf.extractall(DATA_ROOT)

data_yaml_paths = list(DATA_ROOT.glob('**/data.yaml'))
if not data_yaml_paths:
    raise RuntimeError(f'no data.yaml under {{DATA_ROOT}}')
data_yaml = data_yaml_paths[0]
# rewrite 'path' → абсолютна тека data.yaml (Kaggle не робить цього сам; інакше
# train/val-шляхи резолвляться відносно DATASETS_DIR і не знаходяться).
import yaml as _yaml
_dy = _yaml.safe_load(data_yaml.read_text()) or {{}}
_dy['path'] = str(data_yaml.parent)
data_yaml.write_text(_yaml.safe_dump(_dy, allow_unicode=True))
print(f'data.yaml: {{data_yaml}} (path→{{data_yaml.parent}})')
print(data_yaml.read_text())

# train.txt/val.txt містять ВІДНОСНІ шляхи `images/<h>.jpg` (без leading sep).
# ultralytics img2label_paths замінює `{{os.sep}}images{{os.sep}}`→`{{os.sep}}labels{{os.sep}}`,
# а у відносному `images/x.jpg` нема provідного sep → labels НЕ знаходяться
# («No labels found in images.cache»). Тому переписуємо train/val-списки на АБСОЛЮТНІ.
_base = data_yaml.parent
for _lst in ('train', 'val'):
    _p = _dy.get(_lst)
    if not _p:
        continue
    _lp = (_base / _p) if not Path(_p).is_absolute() else Path(_p)
    if _lp.exists() and _lp.suffix == '.txt':
        _lines = [ln.strip() for ln in _lp.read_text().splitlines() if ln.strip()]
        _abs = [str((_base / ln).resolve()) if not Path(ln).is_absolute() else ln for ln in _lines]
        _lp.write_text('\\n'.join(_abs) + '\\n')
        print(f'{{_lst}}.txt -> {{len(_abs)}} absolute paths (e.g. {{_abs[0]}})')

from ultralytics import YOLO
# Kaggle-«T4» = машина з двома T4 (Events показує «GPU T4 x2», 2026-07-21) —
# device='auto' бере ОБИДВІ через ultralytics DDP (batch ділиться навпіл на картку).
_n_gpu = torch.cuda.device_count()
_device = list(range(_n_gpu)) if (PARAMS.get('device', 'auto') == 'auto' and _n_gpu > 1) else 0
print(f'GPUs visible: {{_n_gpu}}, train device: {{_device}}', flush=True)
model = YOLO(PARAMS['base_model'])
results = model.train(
    data=str(data_yaml),
    epochs=PARAMS['epochs'],
    batch=PARAMS['batch_size'],
    imgsz=PARAMS['imgsz'],
    device=_device,
    project=str(WORK / 'runs'),
    name='yolospot',
    exist_ok=True,
    seed=PARAMS['seed'],
    save_period=int(PARAMS.get('save_period', -1)),
    # cache='disk' → декодувати PNG-тайли у .npy раз (прибирає re-decode щоепохи).
    # На Kaggle CPU/RAM ФІКСОВАНІ (~2 ядра, ~13-16GB RAM, не вибираються як на Modal),
    # тож workers не піднімаємо (over-subscription); виграш — лише від cache. УВАГА:
    # Kaggle /tmp обмежений — для дуже великих датасетів disk-cache може впертись у місце.
    cache=(PARAMS.get('cache') or False),
    # АУГМЕНТАЦІЯ (та сама, що Modal): нахил/масштаб/колір/руйнація.
    # fliplr/flipud=0 — текст НЕ дзеркалити (інакше ultralytics дефолт fliplr=0.5!).
    degrees=10.0, scale=0.6, shear=4.0, perspective=0.0005,
    hsv_h=0.10, hsv_s=0.7, hsv_v=0.5, erasing=0.4,
    fliplr=0.0, flipud=0.0,
    verbose=True,
)

# DDP (device=[0,1]) повертає з train() СЛОВНИК метрик, single-GPU — об'єкт
# з .results_dict (зловлено смоуком 5f24a93a, 2026-07-21).
_rd = getattr(results, 'results_dict', results) or {{}}
metrics = {{k: float(v) for k, v in dict(_rd).items()
           if isinstance(v, (int, float))}}
metrics_path = WORK / 'metrics.json'
metrics_path.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

weights_dir = WORK / 'runs' / 'yolospot' / 'weights'
for p in weights_dir.iterdir():
    print(p.name, p.stat().st_size // 1024, 'KB')
"""
