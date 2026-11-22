"""Kaggle credential discovery and verification.

Supports both auth file formats:

  - ``~/.kaggle/access_token``  (KGAT_ token, introduced late 2025)
  - ``~/.kaggle/kaggle.json``   (legacy ``{"username":..., "key":...}``)

We don't re-read the files ourselves — the ``kaggle`` Python package does that
via its ``KaggleApi.authenticate()``. We just probe to give a clearer error
message when something's missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurunner.core.backend import AuthError


def _kaggle_dir() -> Path:
    return Path.home() / ".kaggle"


def access_token_path() -> Path:
    return _kaggle_dir() / "access_token"


def kaggle_json_path() -> Path:
    return _kaggle_dir() / "kaggle.json"


@dataclass
class KaggleCredentials:
    source: str            # "access_token" or "kaggle_json"
    path: Path
    username: str | None   # None if KGAT-style (username resolved server-side)


def discover_credentials() -> KaggleCredentials:
    """Find which credential file is present. Raise AuthError if neither exists."""
    if access_token_path().exists():
        return KaggleCredentials(
            source="access_token",
            path=access_token_path(),
            username=None,
        )
    if kaggle_json_path().exists():
        try:
            data: dict[str, Any] = json.loads(kaggle_json_path().read_text(encoding="utf-8"))
            return KaggleCredentials(
                source="kaggle_json",
                path=kaggle_json_path(),
                username=data.get("username"),
            )
        except (OSError, json.JSONDecodeError) as e:
            raise AuthError(f"Failed to parse {kaggle_json_path()}: {e}") from e
    raise AuthError(
        f"No Kaggle credentials found. Expected one of:\n"
        f"  - {access_token_path()}  (KGAT_… token; recommended)\n"
        f"  - {kaggle_json_path()}   (legacy username+key)\n"
        f"Get one at https://www.kaggle.com/settings → Create API Token."
    )


def verify() -> str:
    """Authenticate and probe a cheap API call. Returns username on success."""
    discover_credentials()  # fail fast with a clear message
    try:
        from kaggle import KaggleApi  # local import to keep CLI startup fast
    except ImportError as e:
        raise AuthError(f"kaggle PyPI package not installed: {e}") from e

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:  # broad: kaggle wraps many errors
        raise AuthError(f"Kaggle authentication failed: {e}") from e

    # Probe with a cheap call that surfaces auth issues immediately.
    try:
        api.kernels_list(mine=True, page_size=1)
    except Exception as e:
        raise AuthError(f"Kaggle API probe failed (token may be revoked): {e}") from e

    username = api.get_config_value("username")
    if not username:
        # KGAT mode: probe `me` endpoint via competitions list which echoes user
        try:
            # Workaround — kaggle 2.x exposes a `get_username()` via api client
            username = getattr(api, "_get_username", lambda: None)() or "<unknown>"
        except Exception:
            username = "<unknown>"
    return username
