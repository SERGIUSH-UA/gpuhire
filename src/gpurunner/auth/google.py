"""Google (Drive) credential discovery and OAuth desktop flow — used by ColabBackend.

Two files live in ``config_dir()``:

  - ``google_client_secret.json`` — the OAuth *client* JSON downloaded from
    https://console.cloud.google.com/apis/credentials (type: **Desktop app**).
    Put it there yourself once.
  - ``google_token.json`` — the *user* credentials (access + refresh token) written
    by ``login()`` after the browser consent flow.

Scope
-----
We request the FULL ``https://www.googleapis.com/auth/drive`` scope, and that is
load-bearing: the Colab notebook writes its outputs through ``drive.mount``, so
those files are owned by the *user*, not by our OAuth client. With the narrower
``drive.file`` scope (only app-created files) ``fetch_outputs`` would see the
notebook we uploaded but NONE of the results it produced.

Gotcha: while the OAuth consent screen sits in **Testing** publishing status,
Google expires refresh tokens after 7 days. Publish the app ("In production") —
the "unverified app" interstitial stays, but the token stops rotting.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurunner.config import config_dir, ensure_dirs
from gpurunner.core.backend import AuthError

SCOPES: list[str] = ["https://www.googleapis.com/auth/drive"]

#: Root folder created in the user's "My Drive" — everything gpurunner does lives here.
DRIVE_ROOT_FOLDER = "gpurunner"


def client_secret_path() -> Path:
    return config_dir() / "google_client_secret.json"


def token_path() -> Path:
    return config_dir() / "google_token.json"


@dataclass
class GoogleCredentials:
    """Where the two credential files are (token may not exist yet)."""

    client_secret: Path
    token: Path
    has_token: bool


_SETUP_HINT = (
    "Setup (once):\n"
    "  1. console.cloud.google.com → new project → enable the **Google Drive API**\n"
    "  2. OAuth consent screen → External → add yourself as a test user →\n"
    "     **Publish app** (in Testing status refresh tokens expire after 7 days)\n"
    "  3. Credentials → Create OAuth client ID → **Desktop app** → download JSON\n"
    "  4. save it as: {path}\n"
    "  5. run: gpurunner auth google --login"
)


def discover_credentials() -> GoogleCredentials:
    """Locate the credential files. Raise AuthError if the client secret is missing."""
    cs = client_secret_path()
    if not cs.exists():
        raise AuthError(
            f"No Google OAuth client secret at {cs}.\n" + _SETUP_HINT.format(path=cs)
        )
    try:
        data: dict[str, Any] = json.loads(cs.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AuthError(f"Failed to parse {cs}: {e}") from e
    if not ({"installed", "web"} & data.keys()):
        raise AuthError(
            f"{cs} does not look like an OAuth client JSON "
            "(expected an 'installed' or 'web' key). Download the **Desktop app** client."
        )
    if "web" in data and "installed" not in data:
        raise AuthError(
            f"{cs} is a *Web application* OAuth client; the desktop loopback flow needs "
            "an **Desktop app** client. Create one in Credentials → Create OAuth client ID."
        )
    return GoogleCredentials(client_secret=cs, token=token_path(), has_token=token_path().exists())


def _import_libs() -> tuple[Any, Any, Any]:
    """Import the google auth/api libs, mapping ImportError to AuthError."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise AuthError(
            f"Google API libraries not installed: {e}\nRun: uv sync --extra colab"
        ) from e
    return Credentials, InstalledAppFlow, Request


def login(*, port: int = 0) -> str:
    """Run the browser consent flow and persist the token. Returns the account e-mail."""
    creds_files = discover_credentials()
    Credentials, InstalledAppFlow, _ = _import_libs()
    _ = Credentials

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_files.client_secret), SCOPES)
    try:
        creds = flow.run_local_server(port=port, prompt="consent", access_type="offline")
    except Exception as e:
        raise AuthError(f"Google OAuth flow failed: {e}") from e
    if not creds.refresh_token:
        raise AuthError(
            "Google returned no refresh token — re-run and make sure you approve the "
            "consent screen (revoke prior grants at myaccount.google.com/permissions)."
        )
    _save(creds)
    return whoami(creds)


def load_credentials() -> Any:
    """Load stored credentials, refreshing them if expired. Raise AuthError if absent."""
    creds_files = discover_credentials()
    if not creds_files.has_token:
        raise AuthError(
            f"No Google token at {creds_files.token}. Run: gpurunner auth google --login"
        )
    Credentials, _, Request = _import_libs()
    try:
        creds = Credentials.from_authorized_user_file(str(creds_files.token), SCOPES)
    except Exception as e:
        raise AuthError(f"Failed to read {creds_files.token}: {e}") from e

    if creds.valid:
        return creds
    if not creds.refresh_token:
        raise AuthError("Stored Google token has no refresh token. Re-run: gpurunner auth google --login")
    try:
        creds.refresh(Request())
    except Exception as e:
        raise AuthError(
            f"Google token refresh failed: {e}\n"
            "If the OAuth consent screen is still in 'Testing' status, refresh tokens "
            "expire after 7 days — publish the app, then re-run: gpurunner auth google --login"
        ) from e
    _save(creds)
    return creds


def build_drive(creds: Any | None = None) -> Any:
    """Build a Drive v3 service client from stored (or supplied) credentials."""
    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise AuthError(
            f"google-api-python-client not installed: {e}\nRun: uv sync --extra colab"
        ) from e
    if creds is None:
        creds = load_credentials()
    try:
        # cache_discovery=False — the file cache backend is unavailable without
        # oauth2client and logs a warning on every call.
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        raise AuthError(f"Failed to build Drive client: {e}") from e


def whoami(creds: Any | None = None) -> str:
    """Return the e-mail address of the authenticated Drive account."""
    service = build_drive(creds)
    try:
        about = service.about().get(fields="user(emailAddress,displayName)").execute()
    except Exception as e:
        raise AuthError(f"Google Drive probe failed (token may be revoked): {e}") from e
    user = about.get("user") or {}
    return str(user.get("emailAddress") or user.get("displayName") or "<unknown>")


def verify() -> str:
    """Authenticate and probe Drive. Returns the account e-mail on success."""
    return whoami(load_credentials())


def _save(creds: Any) -> None:
    ensure_dirs()
    p = token_path()
    p.write_text(creds.to_json(), encoding="utf-8")
    with contextlib.suppress(OSError):
        p.chmod(0o600)  # no-op on Windows, meaningful elsewhere
