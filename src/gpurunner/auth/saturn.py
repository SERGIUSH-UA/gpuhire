"""Saturn Cloud credential discovery.

Saturn Cloud is a hosted JupyterLab/jobs platform with a free tier that includes GPU
instances (the allowance is not published — see the backend docstring). Two values are
needed:

  - **URL** of your Saturn instance, e.g. ``https://app.community.saturnenterprise.io``
  - **API token** — Settings → *User* → API token in the web UI

Sources, in priority order:

  1. ``SATURN_BASE_URL`` + ``SATURN_TOKEN`` environment variables
  2. **gpurunner's own** ``config_dir()/saturn.json`` — written by
     ``gpurunner auth saturn --url … --token …``

Both are exported into the environment before anything touches the SDK:
``saturnfs`` (the fsspec filesystem behind ``sfs://`` paths, which is how inputs
and outputs travel) reads them **at import time** from ``os.environ`` and raises
``KeyError`` if they are absent.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurunner.config import config_dir, ensure_dirs
from gpurunner.core.backend import AuthError

_SETUP_HINT = (
    "Setup (once):\n"
    "  1. create an account at https://saturncloud.io/ (free tier includes GPU instances)\n"
    "  2. web UI → user menu → Settings: copy the API token and the instance URL\n"
    "     (the URL looks like https://app.community.saturnenterprise.io)\n"
    "  3. store them in gpurunner:\n"
    "       gpurunner auth saturn --url <URL> --token <TOKEN>\n"
    "     (env vars SATURN_BASE_URL/SATURN_TOKEN also work and take priority)\n"
    "  4. install the SDK:  uv sync --extra saturn"
)


@dataclass
class SaturnCredentials:
    url: str
    token: str
    source: str


def gpurunner_credentials_path() -> Path:
    """gpurunner's own credential store for Saturn Cloud."""
    return config_dir() / "saturn.json"


def save_credentials(*, url: str, token: str) -> Path:
    """Persist URL + token into gpurunner's config dir. Returns the file path."""
    ensure_dirs()
    path = gpurunner_credentials_path()
    payload = {"url": url.strip().rstrip("/"), "token": token.strip()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with contextlib.suppress(OSError):
        path.chmod(0o600)  # no-op on Windows, meaningful elsewhere
    return path


def _load_gpurunner_file() -> dict[str, str] | None:
    path = gpurunner_credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AuthError(f"Failed to parse {path}: {e}") from e
    if not isinstance(data, dict) or not data.get("url") or not data.get("token"):
        raise AuthError(
            f"{path} has no 'url'/'token' — re-run: gpurunner auth saturn --url … --token …"
        )
    return {str(k): str(v) for k, v in data.items()}


def discover_credentials() -> SaturnCredentials:
    """Locate credentials and export them into the environment for the SDKs."""
    url = os.environ.get("SATURN_BASE_URL")
    token = os.environ.get("SATURN_TOKEN")
    if url and token:
        creds = SaturnCredentials(
            url=url.strip().rstrip("/"), token=token.strip(), source="SATURN_BASE_URL/SATURN_TOKEN env"
        )
    else:
        stored = _load_gpurunner_file()
        if not stored:
            raise AuthError("No Saturn Cloud credentials found.\n" + _SETUP_HINT)
        creds = SaturnCredentials(
            url=stored["url"].rstrip("/"),
            token=stored["token"],
            source=str(gpurunner_credentials_path()),
        )

    # saturnfs reads both at *import* time and raises KeyError when they are
    # missing, so exporting them here is what makes `sfs://` usable at all.
    os.environ["SATURN_BASE_URL"] = creds.url
    os.environ["SATURN_TOKEN"] = creds.token
    return creds


def connect() -> Any:
    """Authenticated ``SaturnConnection``."""
    creds = discover_credentials()
    try:
        from saturn_client import SaturnConnection
    except ImportError as e:
        raise AuthError(
            f"saturn-client is not installed. Run: uv sync --extra saturn ({e})"
        ) from e
    try:
        return SaturnConnection(url=creds.url, api_token=creds.token)
    except Exception as e:
        raise AuthError(f"Saturn Cloud connection failed: {e}") from e


def filesystem() -> Any:
    """``saturnfs`` filesystem instance (``sfs://`` paths), credentials exported."""
    discover_credentials()
    try:
        from saturnfs import SaturnFS
    except ImportError as e:
        raise AuthError(
            f"saturnfs is not installed. Run: uv sync --extra saturn ({e})"
        ) from e
    return SaturnFS()


def verify() -> dict[str, object]:
    """Probe the API. Returns ``{username, org, url}``."""
    conn = connect()
    try:
        user = conn.current_user
        org = conn.primary_org
    except Exception as e:
        raise AuthError(f"Saturn Cloud probe failed: {e}") from e
    return {
        "username": (user or {}).get("username"),
        "org": (org or {}).get("name"),
        "url": conn.url,
    }
