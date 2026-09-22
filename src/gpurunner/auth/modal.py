"""Modal credential discovery — stub aligned with the ModalBackend stub.

Real implementation will probe ``~/.modal.toml`` (the file ``modal token new``
writes) and call ``modal.config._read()`` to confirm token validity.
"""

from __future__ import annotations

from pathlib import Path

from gpurunner.core.backend import AuthError


def modal_config_path() -> Path:
    return Path.home() / ".modal.toml"


def verify() -> str:
    """Verify Modal auth. Returns workspace name on success. Currently a stub."""
    if not modal_config_path().exists():
        raise AuthError(
            f"No Modal config at {modal_config_path()}. "
            "Run: uv sync --extra modal && uv run modal token new"
        )
    raise NotImplementedError(
        "modal verify is a stub in v0.1. ModalBackend lands in a later milestone."
    )
