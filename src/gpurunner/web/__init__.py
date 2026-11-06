"""Локальний дашборд: баланси, квоти й активні прогони в одному вікні.

Залежить від екстри ``web`` (``uv sync --extra web``). Імпорт ``create_app``
винесено у функцію, щоб ``import gpurunner.web`` не падав там, де FastAPI немає.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    from gpurunner.web.app import create_app as _create_app

    return _create_app(*args, **kwargs)
