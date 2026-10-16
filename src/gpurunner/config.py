"""Cross-platform paths and runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "gpurunner"


def config_dir() -> Path:
    """Per-user config dir. Windows: %APPDATA%\\gpurunner, Linux: ~/.config/gpurunner."""
    override = os.environ.get("GPURUNNER_CONFIG_DIR")
    if override:
        return Path(override)
    return Path(user_config_dir(APP_NAME, appauthor=False))


def data_dir() -> Path:
    """Per-user data dir for manifest, cached notebook outputs, etc."""
    override = os.environ.get("GPURUNNER_DATA_DIR")
    if override:
        return Path(override)
    return Path(user_data_dir(APP_NAME, appauthor=False))


def manifest_path() -> Path:
    return data_dir() / "manifest.json"


def registry_dir() -> Path:
    """Тека реєстру: бокси, ручні вердикти, калібрування шардів.

    Відрізняється від `data_dir()` призначенням, а не зручністю. У
    `%LOCALAPPDATA%` лежить машинний стан (реєстр прогонів, квоти): він
    відтворюваний і безглуздий в іншій машині. Тут — знання, здобуте
    вимірюванням і оплачене грішми: які орендні бокси брехали про залізо, а які
    відпрацювали. Його варто бачити в дифі, переносити між машинами й читати
    через рік — тому `GPURUNNER_REPO_DATA_DIR` веде в теку ВЛАСНОГО
    git-репозиторію користувача.

    🔴 Реєстр — дані користувача, а не пакета: у ньому ID орендованих машин,
    ціни й назви власних прогонів. Тому в репозиторії gpurunner він не живе, а
    без змінної лягає в `data_dir()/registry`. Адреса відносно цього файла тут
    не годиться взагалі: у встановленому пакеті вона вказує в `site-packages`.
    """
    override = os.environ.get("GPURUNNER_REPO_DATA_DIR")
    if override:
        return Path(override)
    return data_dir() / "registry"


#: Стара назва: реєстр жив у `data/` самого репозиторію.
repo_data_dir = registry_dir


def modal_volume_name(job: str, fallback: str) -> str:
    """Ім'я тому Modal для job'а, якщо його не дали параметром `modal_volume`.

    Том живе в акаунті користувача, і назвати його має право він:
    `GPURUNNER_MODAL_VOLUME_<JOB>` для одного job'а, `GPURUNNER_MODAL_VOLUME` для
    всіх, інакше — нейтральне ім'я пакета. Дві змінні, бо датасети різних
    job'ів зазвичай лежать у різних томах.
    """
    return (os.environ.get(f"GPURUNNER_MODAL_VOLUME_{job.upper()}")
            or os.environ.get("GPURUNNER_MODAL_VOLUME")
            or fallback)


def ensure_dirs() -> None:
    config_dir().mkdir(parents=True, exist_ok=True)
    data_dir().mkdir(parents=True, exist_ok=True)
