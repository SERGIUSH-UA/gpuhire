"""Cloudflare R2: заливка кадрів, presigned-посилання, прибирання.

Канал доставки між домашнім диском і орендованою машиною. Чому саме він, а не
SFTP на бокс: кадри їдуть один раз і лежать, доки справа не дочитана, тож
переоренда не платить за перезаливку вдруге. І чому presigned, а не ключі:
ключі R2 на чужу машину не потрапляють НІКОЛИ — presigned PUT дає боксу право
записати рівно один заздалегідь названий об'єкт і нічого більше.

Доступ береться з оточення (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`,
`CLOUDFLARE_ACCOUNT_ID` або `R2_ENDPOINT`, `R2_BUCKET`), а якщо там порожньо —
з `config_dir()/r2.json`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from gpurunner.config import config_dir

DEFAULT_BUCKET = "gpurunner-htr"

#: Поріг, за яким boto3 ріже файл на частини. 4 ГБ архіву одним PUT не піде.
MULTIPART_MB = 64

#: 🔴 Паралельність заливки. Було 8, і на черзі з 22 справ R2 відмовляв:
#: «ServiceUnavailable: Reduce your concurrent request rate for the same
#: object». Наслідок гірший за саму помилку — план не створювався ВЗАГАЛІ,
#: тобто втрачалась і та частина, що вже залилась (30.08.2026).
DEFAULT_JOBS = 4

#: Коди, на яких має сенс перечекати: це не наша помилка й не брак прав.
_RETRIABLE = ("ServiceUnavailable", "SlowDown", "RequestTimeout",
              "InternalError", "503", "429")
_RETRIES = 5


class R2Error(RuntimeError):
    """Помилка доступу до R2 з поясненням, що робити."""


def _load_env() -> dict[str, str]:
    """Оточення, а за його відсутності — `config_dir()/r2.json`."""
    env: dict[str, str] = {}
    path = config_dir() / "r2.json"
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            env.update({str(k): str(v) for k, v in stored.items() if v})
        except (OSError, ValueError):
            pass
    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "CLOUDFLARE_ACCOUNT_ID",
                "R2_ENDPOINT", "R2_BUCKET"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def configured() -> bool:
    """Чи є взагалі чим користуватись: ключі й бібліотека доступу.

    Питається ДО того, як складати план: відповідь «ні» — не збій, а звичайний
    стан людини, яка просто орендувала машину на годину й бакета не заводила.
    Для неї є транспорт `box`.
    """
    env = _load_env()
    if not (env.get("R2_ACCESS_KEY_ID") and env.get("R2_SECRET_ACCESS_KEY")):
        return False
    if not (env.get("R2_ENDPOINT") or env.get("CLOUDFLARE_ACCOUNT_ID")):
        return False
    try:
        import boto3  # noqa: F401 — перевіряємо саме наявність
    except ImportError:
        return False
    return True


def bucket_name(explicit: str = "") -> str:
    return explicit or _load_env().get("R2_BUCKET") or DEFAULT_BUCKET


def client(env: dict[str, str] | None = None) -> Any:
    """Клієнт S3 під R2. Помилка тут — це відсутній доступ, а не збій мережі."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:  # pragma: no cover
        raise R2Error(
            # 🔴 Синхронізувати ОДИН extra не можна: `uv sync --extra r2`
            # видалить пакети всіх інших (vast, modal, …), і наступний
            # прогін упаде вже на paramiko.
            "немає boto3. Ставити ВСІ extras РАЗОМ, інакше решта зникне:\n"
            "  uv sync --extra modal --extra colab --extra vast "
            "--extra lightning --extra beam --extra saturn --extra web --extra r2"
        ) from e

    env = env if env is not None else _load_env()
    key = env.get("R2_ACCESS_KEY_ID")
    secret = env.get("R2_SECRET_ACCESS_KEY")
    account = env.get("CLOUDFLARE_ACCOUNT_ID")
    if not key or not secret:
        raise R2Error(
            "немає R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY.\n"
            "Cloudflare Dashboard → R2 → Manage R2 API Tokens → Create API token\n"
            "  Permissions: Object Read & Write\n"
            f"Покласти в оточення або у {config_dir() / 'r2.json'}"
        )
    endpoint = env.get("R2_ENDPOINT") or (
        f"https://{account}.r2.cloudflarestorage.com" if account else "")
    if not endpoint:
        raise R2Error("немає ні R2_ENDPOINT, ні CLOUDFLARE_ACCOUNT_ID")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        # R2 ігнорує регіон, але boto3 вимагає його вказати.
        region_name="auto",
        config=Config(signature_version="s3v4",
                      retries={"max_attempts": 5, "mode": "standard"}),
    )


def _is_retriable(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in _RETRIABLE)


class _Progress:
    """Відсотки + МБ/с у stderr. Заливка 4 ГБ мовчки — це нерв, а не робота."""

    def __init__(self, total: int, name: str) -> None:
        self.total, self.name = total, name
        self.seen = 0
        self.t0 = time.time()
        self._lock = threading.Lock()

    def __call__(self, chunk: int) -> None:
        with self._lock:
            self.seen += chunk
            dt = max(0.001, time.time() - self.t0)
            pct = 100.0 * self.seen / max(1, self.total)
            eta = (self.total - self.seen) / max(1.0, self.seen / dt)
            sys.stderr.write(
                f"\r  {self.name}: {pct:5.1f}%  {self.seen / 1e6:7.0f}/"
                f"{self.total / 1e6:.0f} МБ  {self.seen / 1e6 / dt:5.2f} МБ/с"
                f"  ще ~{eta / 60:.0f} хв   ")
            sys.stderr.flush()


def put(path: Path, *, bucket: str = "", prefix: str = "", jobs: int = DEFAULT_JOBS,
        s3: Any = None, quiet: bool = False) -> str:
    """Залити файл. Повертає ключ.

    🔴 Ретрай із наростанням паузи — не косметика. На черзі з 22 справ R2
    відповів `ServiceUnavailable: Reduce your concurrent request rate`, і
    планувальник упав ЦІЛКОМ: файл плану лишився нульовим, тобто втратилась і
    вже залита половина. Обхід тоді був «бити пакетами по 7-8 справ»; тепер
    його не треба пам'ятати.
    """
    bucket = bucket_name(bucket)
    s3 = s3 or client()
    key = f"{prefix.rstrip('/')}/{path.name}" if prefix else path.name
    size = path.stat().st_size
    extra: dict[str, Any] = {}
    try:
        from boto3.s3.transfer import TransferConfig
    except ImportError:
        # Налаштування переносу — суто boto3-івська ручка. Без boto3 реальної
        # заливки все одно немає (клієнт підставлений ззовні), тож і різати на
        # частини нема чого.
        pass
    else:
        extra["Config"] = TransferConfig(
            multipart_threshold=MULTIPART_MB * 1024 * 1024,
            multipart_chunksize=MULTIPART_MB * 1024 * 1024,
            max_concurrency=jobs, use_threads=True)
    for attempt in range(1, _RETRIES + 1):
        try:
            s3.upload_file(str(path), bucket, key, **extra,
                           Callback=None if quiet else _Progress(size, path.name))
            if not quiet:
                sys.stderr.write("\n")
            return key
        except Exception as e:
            if attempt >= _RETRIES or not _is_retriable(e):
                raise
            pause = min(60.0, 2.0 ** attempt)
            print(f"[r2] {path.name}: {type(e).__name__} — чекаю {pause:.0f} с "
                  f"(спроба {attempt}/{_RETRIES})", file=sys.stderr, flush=True)
            time.sleep(pause)
    raise R2Error(f"не залилось після {_RETRIES} спроб: {path}")  # pragma: no cover


def ls(prefix: str = "", *, bucket: str = "", s3: Any = None) -> list[dict[str, Any]]:
    """Об'єкти під префіксом: `key`, `size`, `modified`."""
    bucket = bucket_name(bucket)
    s3 = s3 or client()
    out: list[dict[str, Any]] = []
    kw: dict[str, Any] = {"Bucket": bucket}
    if prefix:
        kw["Prefix"] = prefix
    for page in s3.get_paginator("list_objects_v2").paginate(**kw):
        for obj in page.get("Contents", []):
            out.append({"key": obj["Key"], "size": obj["Size"],
                        "modified": obj.get("LastModified")})
    return out


def get_url(key: str, *, hours: float = 12.0, bucket: str = "", s3: Any = None) -> str:
    """Presigned GET."""
    bucket = bucket_name(bucket)
    s3 = s3 or client()
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(hours * 3600))


def put_urls(prefix: str, count: int, *, hours: float = 24.0, bucket: str = "",
             s3: Any = None) -> list[str]:
    """N presigned PUT — щоб орендований бокс писав чекпоінти без ключів.

    Кожне посилання придатне лише для СВОГО ключа, тож їхня кількість — це
    стеля числа чекпоінтів за прогін. Вичерпання означає мовчазну втрату хвоста.
    """
    bucket = bucket_name(bucket)
    s3 = s3 or client()
    return [
        s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": f"{prefix.rstrip('/')}/ckpt_{i:04d}.tgz"},
            ExpiresIn=int(hours * 3600))
        for i in range(1, count + 1)
    ]


def ckpt_get_urls(prefix: str, count: int, *, hours: float = 24.0, bucket: str = "",
                  s3: Any = None) -> list[str]:
    """Дзеркальні GET на ті самі ключі — без них чекпоінти нічим прочитати."""
    bucket = bucket_name(bucket)
    s3 = s3 or client()
    return [
        get_url(f"{prefix.rstrip('/')}/ckpt_{i:04d}.tgz", hours=hours,
                bucket=bucket, s3=s3)
        for i in range(1, count + 1)
    ]


def prune(prefixes: list[str], *, bucket: str = "", yes: bool = False,
          s3: Any = None) -> list[tuple[str, int]]:
    """Прибрати теки. Без `yes` лише показує, що зникло б.

    🔴🔴 Префікс тут — ТЕКА, а не підрядок. Доти збіг ішов по підрядку, і
    `ckpt/230-1-12` зносив `ckpt/230-1-129` — чекпоінти ЧУЖОГО завершеного
    заходу (17.08.2026). Рятувало лише те, що люди навчились дописувати слеш
    руками; тепер його дописує код.
    """
    bucket = bucket_name(bucket)
    s3 = s3 or client()
    victims: list[tuple[str, int]] = []
    for raw in prefixes:
        prefix = raw if raw.endswith("/") else raw + "/"
        for obj in ls(prefix, bucket=bucket, s3=s3):
            victims.append((obj["key"], obj["size"]))
        # Сам об'єкт із таким іменем (без «теки») теж законна ціль.
        for obj in ls(raw, bucket=bucket, s3=s3):
            if obj["key"] == raw:
                victims.append((obj["key"], obj["size"]))
    if yes:
        for key, _ in victims:
            s3.delete_object(Bucket=bucket, Key=key)
    return victims
