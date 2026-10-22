"""Ваги епох трену в R2 по ходу трену: посилання для боксу, забір, прибирання.

Навіщо: трен на Vast до кінця тримав ваги епох лише на боксі. Бокс, що
самознищився за таймером чи зник разом із хостом, забирав усі натреновані
епохи, хоча кожна з них була готова годинами раніше.

Як:
- `gpurunner run kraken_train|parseq_train -b vast` складає специфікацію
  (`build_spec`): presigned PUT на ключі ``train_out/<handle>/<ім'я>`` для
  файлів із відомими іменами і пул слотів ``train_out/<handle>/slots/slot_NNN``
  для решти (``checkpoint_NN-0.xxxx.ckpt`` kraken: точність у імені наперед
  невідома). Файл їде на бокс у ``REMOTE_ROOT/train_out.json``; бокс
  (`_embedded/_train_out.py`) вантажить кожен завершений файл і після кожного
  перезаливає маніфест ``_manifest.json`` (ім'я → ключ, розмір, sha256).
- `gpurunner fetch <id>` догружає з R2 те, чого бракує (бокс знищено —
  усе), `gpurunner train-out ls|get|prune|convert` — решта.

Ключі R2 на бокс не потрапляють: лише посилання на заздалегідь названі
об'єкти. Строк дії — `url_hours` (≥ 48 год, стеля SigV4 — 7 діб).
Прибирання — лише на явну команду (`train-out prune`, `fetch --prune-r2`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PREFIX = "train_out"
#: Ім'я файла специфікації в `REMOTE_ROOT` боксу. Раннер читає його за
#: шляхом `_TO_SPEC_PATH` у `_embedded/_train_out.py` — тест звіряє обидва боки.
SPEC_NAME = "train_out.json"
MANIFEST_NAME = "_manifest.json"
SLOT_DIR = "slots"
#: Трени, яких це стосується.
TRAIN_JOBS = ("kraken_train", "parseq_train")
#: Мінімальний строк дії посилань. Трен іде до `max_hours`, далі фінальне
#: вивантаження; запас — на повільний старт і забір після самознищення.
URL_HOURS_MIN = 48.0
#: Стеля presigned SigV4 — 7 діб; довше boto3 підпише, а R2 відмовить.
URL_HOURS_MAX = 168.0
#: Стеля епох kraken, коли `-p epochs=-1` (ketos зупиняє лише lag/wall-limit).
#: Найдовший трен в історії — 27 епох (`928fc985`); 80 з запасом.
KRAKEN_EPOCHS_CAP = 80

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._+=-][A-Za-z0-9._+=@-]{0,200}$")
_CKPT = re.compile(r"^checkpoint_(\d+)-[\d.]+\.ckpt$")

_ON = ("1", "true", "yes", "on")
_OFF = ("0", "false", "no", "off")


def mode(params: dict[str, Any]) -> str:
    """`-p r2_epochs=…` → ``auto`` | ``on`` | ``off``."""
    raw = params.get("r2_epochs")
    if raw is None or raw == "":
        return "auto"
    if isinstance(raw, bool):
        return "on" if raw else "off"
    text = str(raw).strip().lower()
    if text in _ON:
        return "on"
    if text in _OFF:
        return "off"
    if text == "auto":
        return "auto"
    raise ValueError(f"r2_epochs: очікую true|false|auto, отримано {raw!r}")


def wanted(job_name: str, params: dict[str, Any]) -> bool:
    """Чи складати посилання для цього заходу. Типово — коли R2 налаштований.

    ``r2_epochs=true`` без налаштованого R2 — помилка (запитано явно), а не
    мовчазне «вимкнено»: краще впасти до оренди, ніж дізнатись після трену.
    """
    if job_name not in TRAIN_JOBS:
        return False
    m = mode(params)
    if m == "off":
        return False
    from gpurunner.htr import r2

    if not r2.configured():
        if m == "on":
            raise RuntimeError("r2_epochs=true, але R2 не налаштований (r2.json або R2_* в оточенні)")
        return False
    return True


def job_prefix(job_id: str) -> str:
    return f"{PREFIX}/{job_id}/"


def url_hours(max_hours: float, autodestroy_hours: float = 0.0) -> float:
    """Строк дії посилань: трен + самознищення + запас, але не менше 48 год і не більше 7 діб."""
    want = float(max_hours) + max(0.0, float(autodestroy_hours)) + 12.0
    return min(URL_HOURS_MAX, max(URL_HOURS_MIN, want))


def plan(job_name: str, normalized: dict[str, Any], raw: dict[str, Any] | None = None
         ) -> tuple[list[str], int]:
    """Відомі наперед імена файлів і скільки слотів під решту.

    Слот — одне посилання на один об'єкт, тож їхня кількість — стеля числа
    файлів з невідомими іменами. Вичерпання бокс пише в лог і продовжує трен.
    """
    raw = raw or {}
    if job_name == "kraken_train":
        epochs = int(normalized.get("epochs", -1))
        cap = int(raw.get("r2_max_epochs") or (epochs if epochs > 0 else KRAKEN_EPOCHS_CAP))
        named = ["ktrain.log", "ktrain_summary.json", "kraken_churro_best.mlmodel",
                 "ktrain_ddp_attempt.log"]
        # чекпойнт + сконвертована епоха (.mlmodel чи .safetensors) + замір
        # епохи на кожну епоху; плюс best_*, база заміру й запас
        return named, 3 * (cap + 1) + 10
    if job_name == "parseq_train":
        epochs = int(normalized.get("epochs", 1))
        named = ["ptrain.log", "ptrain_summary.json", "parseq_best.pt", "parseq_last.pt"]
        named += [f"parseq_ep{i:02d}.pt" for i in range(0, epochs + 1)]
        # замір кожної епохи (xeval_epNN.tsv) і логи рангів DDP
        return named, epochs + 20
    raise ValueError(f"{job_name}: не трен, ваги епох у R2 не передбачені")


def build_spec(job_name: str, normalized: dict[str, Any], raw: dict[str, Any] | None,
               job_id: str, *, hours: float, s3: Any = None) -> dict[str, Any]:
    """Специфікація для боксу: presigned PUT на імена, слоти й маніфест."""
    from gpurunner.htr import r2

    s3 = s3 or r2.client()
    named, n_slots = plan(job_name, normalized, raw)
    base = job_prefix(job_id)

    def entry(key: str) -> dict[str, str]:
        return {"key": key, "url": r2.put_url(key, hours=hours, s3=s3)}

    return {
        "version": 1,
        "job_id": job_id,
        "job": job_name,
        "prefix": base,
        "expires": (datetime.now(tz=UTC) + timedelta(hours=hours)).isoformat(timespec="seconds"),
        "named": {name: entry(base + name) for name in named},
        "slots": [entry(f"{base}{SLOT_DIR}/slot_{i:03d}") for i in range(n_slots)],
        "manifest": entry(base + MANIFEST_NAME),
    }


# ── забір ────────────────────────────────────────────────────────────────────


def read_manifest(job_id: str, *, s3: Any = None) -> dict[str, Any] | None:
    """Маніфест заходу з R2 (`None` — бокс ще нічого не вивантажив або R2 не тримає)."""
    from gpurunner.htr import r2

    s3 = s3 or r2.client()
    try:
        obj = s3.get_object(Bucket=r2.bucket_name(), Key=job_prefix(job_id) + MANIFEST_NAME)
        data = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def listing(job_id: str, *, s3: Any = None) -> dict[str, dict[str, Any]]:
    """Що лежить у R2: ``{ім'я: {key, size, sha256?}}``.

    Ім'я слота береться з маніфесту; слот без запису в маніфесті (маніфест не
    встиг доїхати) ім'я не має — тоді лишається ``slots/slot_NNN``, і
    `fetch` такий не бере.
    """
    from gpurunner.htr import r2

    s3 = s3 or r2.client()
    base = job_prefix(job_id)
    objects = {o["key"]: int(o["size"]) for o in r2.ls(base, s3=s3)}
    manifest = read_manifest(job_id, s3=s3) or {}
    out: dict[str, dict[str, Any]] = {}
    by_key: dict[str, str] = {}
    for name, meta in (manifest.get("files") or {}).items():
        if isinstance(meta, dict) and meta.get("key"):
            by_key[str(meta["key"])] = str(name)
    for key, size in sorted(objects.items()):
        if key == base + MANIFEST_NAME:
            continue
        name = by_key.get(key) or key[len(base):]
        meta = (manifest.get("files") or {}).get(name) or {}
        out[name] = {"key": key, "size": size, "sha256": meta.get("sha256", ""),
                     "expected_size": meta.get("size")}
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def skip_after_box(job_name: str, out_dir: Path) -> Callable[[str], bool]:
    """Що НЕ догружати з R2, коли бокс віддав вихід штатно.

    kraken: `checkpoint_NN-*.ckpt` у вихід боксу не потрапляє (лежить у
    /tmp ketos), але при цілому заборі його вже заступає `epoch_NN.mlmodel`
    чи `.safetensors` — тягнути ~50 МБ на епоху вдруге немає сенсу.
    """
    def skip(name: str) -> bool:
        if job_name != "kraken_train":
            return False
        m = _CKPT.match(name)
        if not m:
            return False
        nn = int(m.group(1))
        return any((out_dir / f"epoch_{nn:02d}{ext}").is_file()
                   for ext in (".mlmodel", ".safetensors"))
    return skip


def skip_superseded(names: set[str]) -> Callable[[str], bool]:
    """Не тягнути `checkpoint_NN-*.ckpt`, коли в R2 уже є `epoch_NN.*` тієї ж епохи.

    Чекпоінт ketos (~330 МБ) потрібен лише як аварійне джерело, якщо конвертація на
    боксі не дійшла; поряд із готовою вагою епохи він — зайва година й гроші забору.
    """
    def skip(name: str) -> bool:
        m = _CKPT.match(name)
        if not m:
            return False
        nn = int(m.group(1))
        return any(f"epoch_{nn:02d}{ext}" in names for ext in (".mlmodel", ".safetensors"))
    return skip


def fetch(job_id: str, out_dir: Path, *, skip: Callable[[str], bool] | None = None,
          replace_mismatch: bool = False, s3: Any = None,
          log: Callable[[str], None] = print) -> dict[str, Any]:
    """Догрузити з R2 у ``out_dir`` файли, яких там немає.

    Наявний файл НЕ перезаписується: з живого боксу він новіший за R2 (лог).
    ``replace_mismatch`` — перезаписати файли, чий розмір розходиться з R2.
    Цілість: розмір і sha256 з маніфесту; розбіжність — файл відкидається.
    """
    from gpurunner.htr import r2

    s3 = s3 or r2.client()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"downloaded": [], "present": [], "skipped": [], "failed": [],
                              "unnamed": [], "manifest": None}
    manifest = read_manifest(job_id, s3=s3)
    report["manifest"] = manifest
    if manifest is not None:
        (out_dir / "train_out_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    bucket = r2.bucket_name()
    items = listing(job_id, s3=s3)
    superseded = skip_superseded(set(items)) if not replace_mismatch else None
    for name, meta in items.items():
        if superseded is not None and superseded(name) and not (out_dir / name).exists():
            report["skipped"].append(name)
            continue
        if "/" in name or not _SAFE_NAME.match(name) or name in (".", ".."):
            report["unnamed"].append(name)
            continue
        target = out_dir / name
        if target.exists() and (not replace_mismatch or target.stat().st_size == meta["size"]):
            report["present"].append(name)
            continue
        if skip is not None and skip(name):
            report["skipped"].append(name)
            continue
        part = target.with_name(target.name + ".r2part")
        try:
            s3.download_file(bucket, meta["key"], str(part))
            size = part.stat().st_size
            if meta.get("expected_size") is not None and size != int(meta["expected_size"]):
                raise RuntimeError(f"розмір {size} ≠ {meta['expected_size']} з маніфесту")
            if meta.get("sha256") and _sha256(part) != meta["sha256"]:
                raise RuntimeError("sha256 не збігся з маніфестом")
            part.replace(target)
            report["downloaded"].append(name)
        except Exception as exc:
            with contextlib.suppress(OSError):
                part.unlink()
            report["failed"].append(f"{name}: {exc}")
            log(f"[r2-epochs] ⚠ {name}: {exc}")
    if report["unnamed"]:
        log(f"[r2-epochs] ⚠ {len(report['unnamed'])} слот(ів) без імені в маніфесті — "
            f"не забрано: {', '.join(report['unnamed'][:3])}…")
    return report


def prune(job_id: str, *, yes: bool = False, s3: Any = None) -> list[tuple[str, int]]:
    """Прибрати ``train_out/<job_id>/``. Без ``yes`` — лише показати, що зникло б."""
    from gpurunner.htr import r2

    if not job_id or "/" in job_id:
        raise ValueError(f"невірний id заходу: {job_id!r}")
    return r2.prune([job_prefix(job_id)], yes=yes, s3=s3)
