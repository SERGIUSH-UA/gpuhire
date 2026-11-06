"""Beam.cloud credential discovery.

Beam (beam.cloud) is the closest thing to Modal with a *recurring* free tier —
$30 of credit refreshed monthly (Developer plan, no card required). Auth is a
single bearer token.

Sources, in priority order:

  1. ``BEAM_TOKEN`` environment variable
  2. **gpurunner's own** ``config_dir()/beam.json`` — written by
     ``gpurunner auth beam --token …``
  3. ``~/.beam/config.ini`` — written by the official ``beam configure`` CLI

Whatever we find in (2) or (3) is exported as ``BEAM_TOKEN`` before the SDK is
touched. That matters for a boring reason: with no token in sight the SDK's
``get_config_context()`` drops into an **interactive prompt**, which would hang a
non-interactive ``gpurunner run``. We fail with instructions instead.
"""

from __future__ import annotations

import configparser
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurunner.config import config_dir, ensure_dirs
from gpurunner.core.backend import AuthError

#: Written by ``beam configure default --token …``.
BEAM_CONFIG_PATH = Path.home() / ".beam" / "config.ini"

_SETUP_HINT = (
    "Setup (once):\n"
    "  1. create an account at https://beam.cloud/ (free tier: $30 credit, refreshed monthly)\n"
    "  2. Settings → API Keys → copy the token\n"
    "  3. store it in gpurunner:\n"
    "       gpurunner auth beam --token <TOKEN>\n"
    "     (the BEAM_TOKEN env var also works and takes priority)\n"
    "  4. install the SDK:  uv sync --extra beam"
)


@dataclass
class BeamCredentials:
    token: str
    source: str


def gpurunner_credentials_path() -> Path:
    """gpurunner's own credential store for Beam."""
    return config_dir() / "beam.json"


def save_token(token: str) -> Path:
    """Persist the token into gpurunner's config dir. Returns the file path."""
    ensure_dirs()
    path = gpurunner_credentials_path()
    path.write_text(json.dumps({"token": token.strip()}, indent=2), encoding="utf-8")
    with contextlib.suppress(OSError):
        path.chmod(0o600)  # no-op on Windows, meaningful elsewhere
    return path


def _token_from_gpurunner_file() -> str | None:
    path = gpurunner_credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AuthError(f"Failed to parse {path}: {e}") from e
    token = (data or {}).get("token") if isinstance(data, dict) else None
    if not token:
        raise AuthError(f"{path} has no 'token' — re-run: gpurunner auth beam --token <TOKEN>")
    return str(token).strip()


def _token_from_beam_cli() -> str | None:
    """Token out of ``~/.beam/config.ini`` (the official CLI's store)."""
    if not BEAM_CONFIG_PATH.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(BEAM_CONFIG_PATH)
    except configparser.Error as e:
        raise AuthError(f"Failed to parse {BEAM_CONFIG_PATH}: {e}") from e
    for section in [parser.defaults(), *(dict(parser[s]) for s in parser.sections())]:
        token = (section or {}).get("token")
        if token:
            return str(token).strip()
    return None


def discover_credentials() -> BeamCredentials:
    """Locate the token and export it as ``BEAM_TOKEN`` for the SDK."""
    env = os.environ.get("BEAM_TOKEN")
    if env and env.strip():
        return BeamCredentials(token=env.strip(), source="BEAM_TOKEN env")

    token = _token_from_gpurunner_file()
    source = str(gpurunner_credentials_path())
    if not token:
        token = _token_from_beam_cli()
        source = str(BEAM_CONFIG_PATH)
    if not token:
        raise AuthError("No Beam token found.\n" + _SETUP_HINT)

    # The SDK reads BEAM_TOKEN when the config file has no usable context; without
    # this it would prompt on stdin and hang a non-interactive run.
    os.environ["BEAM_TOKEN"] = token
    return BeamCredentials(token=token, source=source)


def import_sdk() -> Any:
    """Import ``beam`` and make its settings re-read the environment.

    ``beta9.config`` caches an ``SDKSettings`` instance the first time anything
    asks for it, and that snapshot is where ``BEAM_TOKEN`` is read from. If the
    token was exported *after* that snapshot (which is exactly what
    ``discover_credentials`` does) the cached settings would still say "no token"
    — so we force a refresh here.
    """
    try:
        import beam
    except ImportError as e:
        raise AuthError(
            f"beam-client is not installed. Run: uv sync --extra beam ({e})"
        ) from e

    from beta9.config import SDKSettings, set_settings

    set_settings(SDKSettings())
    return beam


def service_client() -> Any:
    """Authenticated ``beta9`` gRPC client (volumes, tasks, gateway).

    The context **must** be passed explicitly: `ServiceClient()` defers to
    `get_channel(None)`, which opens an interactive "Context Name [default]:"
    prompt on stdin instead of using the token we just discovered (verified live
    2026-08-02 — a bare `ServiceClient()` hung `gpurunner auth beam --verify`).
    """
    discover_credentials()
    import_sdk()
    from beta9.channel import ServiceClient
    from beta9.config import get_config_context

    return ServiceClient(get_config_context())


def verify() -> dict[str, object]:
    """Probe the API by listing volumes. Returns ``{token_source, volumes}``."""
    creds = discover_credentials()
    from beta9.clients.volume import ListVolumesRequest

    try:
        with service_client() as service:
            res = service.volume.list_volumes(ListVolumesRequest())
    except Exception as e:
        raise AuthError(f"Beam probe failed: {e}") from e
    if not getattr(res, "ok", False):
        raise AuthError(
            f"Beam rejected the token ({getattr(res, 'err_msg', '')}). "
            "Check Settings → API Keys."
        )
    return {"token_source": creds.source, "volumes": len(getattr(res, "volumes", []) or [])}
