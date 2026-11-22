"""Lightning AI credential discovery.

Sources, in priority order:

  1. ``LIGHTNING_USER_ID`` + ``LIGHTNING_API_KEY`` environment variables
  2. **gpurunner's own** ``config_dir()/lightning.json`` — written by
     ``gpurunner auth lightning --user-id … --api-key …``, so the keys live with
     the rest of gpurunner's config instead of in your shell environment
  3. ``~/.lightning/credentials.json``, written by ``lightning login``

Whatever we find in (2) is exported into the environment before the SDK is used —
``lightning-sdk`` authenticates off those two env vars, so this is the seam that
lets gpurunner own the credentials.

We also resolve the **teamspace** — every Job, upload and download is scoped to
one. Order: ``GPURUNNER_LIGHTNING_TEAMSPACE`` / ``LIGHTNING_TEAMSPACE`` env, then
the ``teamspace`` field of ``lightning.json``.

Free tier: ~15 Lightning credits/month ≈ 22 h of T4. Unlike Vast, nothing keeps
billing after a job finishes — jobs are per-run, not rented boxes.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from gpurunner.config import config_dir, ensure_dirs
from gpurunner.core.backend import AuthError

CREDENTIALS_PATH = Path.home() / ".lightning" / "credentials.json"


def gpurunner_credentials_path() -> Path:
    """gpurunner's own credential store for Lightning."""
    return config_dir() / "lightning.json"


def save_credentials(
    *, user_id: str, api_key: str, teamspace: str | None = None
) -> Path:
    """Persist credentials into gpurunner's config dir. Returns the file path."""
    ensure_dirs()
    path = gpurunner_credentials_path()
    payload: dict[str, str] = {"user_id": user_id.strip(), "api_key": api_key.strip()}
    if teamspace:
        payload["teamspace"] = teamspace.strip()
    elif path.exists():
        # keep a previously-saved teamspace when only the keys are rotated
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("teamspace"):
                payload["teamspace"] = str(old["teamspace"])
        except (OSError, json.JSONDecodeError):
            pass
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
    if not isinstance(data, dict) or not data.get("user_id") or not data.get("api_key"):
        raise AuthError(f"{path} has no 'user_id'/'api_key' — re-run: gpurunner auth lightning --user-id … --api-key …")
    return {str(k): str(v) for k, v in data.items()}

_SETUP_HINT = (
    "Setup (once):\n"
    "  1. create an account at https://lightning.ai/ (free tier ≈ 22 h of T4 per month)\n"
    "  2. user menu → Keys → copy USER_ID and API_KEY\n"
    "  3. store them in gpurunner:\n"
    "       gpurunner auth lightning --user-id <ID> --api-key <KEY> --teamspace <NAME>\n"
    "     (env vars LIGHTNING_USER_ID/LIGHTNING_API_KEY also work and take priority)\n"
    "  4. create one Studio in the web UI — jobs borrow its environment"
)


@dataclass
class LightningCredentials:
    source: str
    user_id: str | None
    teamspace: str | None


def default_teamspace() -> str | None:
    env = os.environ.get("GPURUNNER_LIGHTNING_TEAMSPACE") or os.environ.get("LIGHTNING_TEAMSPACE")
    if env:
        return env
    try:
        stored = _load_gpurunner_file()
    except AuthError:
        return None
    return (stored or {}).get("teamspace")


def discover_credentials() -> LightningCredentials:
    """Locate credentials and make sure the SDK can see them.

    The SDK reads ``LIGHTNING_USER_ID``/``LIGHTNING_API_KEY`` from the environment,
    so credentials stored in gpurunner's own file are exported here — that is what
    makes ``gpurunner auth lightning --api-key …`` enough, with no shell setup.
    """
    user_id = os.environ.get("LIGHTNING_USER_ID")
    api_key = os.environ.get("LIGHTNING_API_KEY")
    if user_id and api_key:
        return LightningCredentials(
            source="LIGHTNING_USER_ID/LIGHTNING_API_KEY env",
            user_id=user_id,
            teamspace=default_teamspace(),
        )

    stored = _load_gpurunner_file()
    if stored:
        os.environ["LIGHTNING_USER_ID"] = stored["user_id"]
        os.environ["LIGHTNING_API_KEY"] = stored["api_key"]
        return LightningCredentials(
            source=str(gpurunner_credentials_path()),
            user_id=stored["user_id"],
            teamspace=default_teamspace(),
        )

    if CREDENTIALS_PATH.exists():
        try:
            data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise AuthError(f"Failed to parse {CREDENTIALS_PATH}: {e}") from e
        return LightningCredentials(
            source=str(CREDENTIALS_PATH),
            user_id=data.get("user_id") or data.get("userId"),
            teamspace=default_teamspace(),
        )
    raise AuthError("No Lightning AI credentials found.\n" + _SETUP_HINT)


def import_sdk() -> object:
    """Import ``lightning_sdk``, mapping ImportError to a clear AuthError."""
    try:
        import lightning_sdk
    except ImportError as e:
        raise AuthError(
            f"lightning-sdk not installed: {e}\nRun: uv sync --extra lightning"
        ) from e
    return lightning_sdk


def current_username() -> str | None:
    """Username of the authenticated account (needed to own a Teamspace lookup)."""
    discover_credentials()
    import_sdk()
    try:
        from lightning_sdk.lightning_cloud.rest_client import LightningClient

        return str(LightningClient(retry=False).auth_service_get_user().username)
    except Exception:
        return None


def list_teamspaces() -> list[str]:
    """Names of teamspaces the account is a member of."""
    discover_credentials()
    import_sdk()
    try:
        from lightning_sdk.lightning_cloud.rest_client import LightningClient

        memberships = LightningClient(retry=False).projects_service_list_memberships()
    except Exception as e:
        raise AuthError(f"Could not list Lightning teamspaces: {e}") from e
    return sorted({str(m.name) for m in (memberships.memberships or [])})


def resolve_teamspace(name: str | None = None) -> object:
    """Return a ``lightning_sdk.Teamspace``. Raises AuthError with guidance.

    A teamspace name is ambiguous on its own — the SDK needs to know whether a user
    or an organization owns it ("Neither user or org are specified"). We pass the
    authenticated username, which is what a personal free-tier account needs.
    """
    discover_credentials()
    sdk = import_sdk()
    name = name or default_teamspace()
    owner = os.environ.get("GPURUNNER_LIGHTNING_ORG") or (_load_gpurunner_file() or {}).get("org")
    kwargs: dict[str, str] = {}
    if owner:
        kwargs["org"] = owner
    else:
        user = current_username()
        if user:
            kwargs["user"] = user
    try:
        return sdk.Teamspace(name=name, **kwargs)  # type: ignore[attr-defined]
    except Exception as e:
        hint = ""
        try:
            available = list_teamspaces()
            if available:
                hint = f"\nAvailable teamspaces: {', '.join(available)}"
        except AuthError:
            pass
        raise AuthError(
            f"Could not resolve Lightning teamspace {name or '(default)'}: {e}{hint}\n"
            "Save the right one: gpurunner auth lightning --teamspace <NAME>"
        ) from e


def verify() -> dict[str, object]:
    """Probe the API. Returns ``{user, teamspace, studios}``."""
    creds = discover_credentials()
    ts = resolve_teamspace()
    try:
        studios = [s.name for s in ts.studios]  # type: ignore[attr-defined]
    except Exception as e:
        raise AuthError(f"Lightning probe failed: {e}") from e
    return {
        "user": creds.user_id or "<from credentials.json>",
        "teamspace": getattr(ts, "name", "?"),
        "studios": studios,
    }
