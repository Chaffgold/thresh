"""OAuth2 client-credentials authentication for the OpenSky REST API.

OpenSky retired HTTP Basic authentication; the REST API now accepts only
bearer tokens minted by its Keycloak realm via the ``client_credentials``
grant. Tokens live 30 minutes, after which the API answers ``401`` and the
caller must mint a fresh one and retry.

Credentials resolve in this order (see :func:`load_credentials`):

1. ``OPENSKY_CLIENT_ID`` / ``OPENSKY_CLIENT_SECRET`` environment variables —
   matching the existing ``ADSBX_API_KEY`` convention (``TRAINING.md``).
2. A JSON file, by default ``~/.config/opensky/credentials.json``. This is
   the file OpenSky's web UI hands you when you create an API client, so a
   downloaded ``credentials.json`` can be dropped in unmodified.
3. Nothing — anonymous access, which is quota-limited to roughly 400 credits
   a day (in practice ~55 bbox polls) versus 4,000 for a standard account and
   8,000 for an active feeder.

Secrets are deliberately **not** accepted on the command line: ``argv`` is
world-readable through ``ps`` on a shared machine.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
)
DEFAULT_CREDENTIALS_PATH = Path.home() / ".config" / "opensky" / "credentials.json"
ENV_CLIENT_ID = "OPENSKY_CLIENT_ID"
ENV_CLIENT_SECRET = "OPENSKY_CLIENT_SECRET"

DEFAULT_TOKEN_LIFETIME_S = 1800.0
"""Assumed lifetime when the token response omits ``expires_in`` (30 min)."""

TOKEN_EXPIRY_MARGIN_S = 60.0
"""Refresh this long before nominal expiry so in-flight requests don't 401."""


class OpenSkyAuthError(Exception):
    """Raised when credentials are unusable or a token cannot be minted."""


@dataclass(frozen=True, slots=True)
class OpenSkyCredentials:
    """An OpenSky API client's OAuth2 ``client_credentials`` pair."""

    client_id: str
    client_secret: str

    def __post_init__(self) -> None:
        if not self.client_id or not self.client_secret:
            raise OpenSkyAuthError("client_id and client_secret must both be non-empty")

    def __repr__(self) -> str:
        """Redact the secret so it cannot leak into logs or tracebacks."""
        return f"OpenSkyCredentials(client_id={self.client_id!r}, client_secret=<redacted>)"


def credentials_from_mapping(payload: Mapping[str, Any], *, origin: str) -> OpenSkyCredentials:
    """Build credentials from a parsed JSON object.

    Accepts OpenSky's own ``clientId`` / ``clientSecret`` spelling as well as
    snake_case, so a downloaded ``credentials.json`` works unmodified.
    """
    client_id = payload.get("clientId") or payload.get("client_id")
    client_secret = payload.get("clientSecret") or payload.get("client_secret")
    if not client_id or not client_secret:
        raise OpenSkyAuthError(
            f"{origin} must contain clientId and clientSecret "
            f"(found keys: {sorted(payload)});  re-download it from "
            "https://opensky-network.org/my-opensky/api-client"
        )
    return OpenSkyCredentials(client_id=str(client_id), client_secret=str(client_secret))


def _warn_if_world_readable(path: Path) -> None:
    """Nudge the operator if a secret file is readable beyond its owner."""
    try:
        mode = path.stat().st_mode
    except OSError:  # pragma: no cover - stat failure is not worth failing over
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(
            f"warning: {path} is readable by group/other; run `chmod 600 {path}`",
            file=sys.stderr,
        )


def _credentials_from_file(path: Path) -> OpenSkyCredentials:
    _warn_if_world_readable(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenSkyAuthError(f"cannot read OpenSky credentials from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OpenSkyAuthError(f"{path} must contain a JSON object, got {type(payload).__name__}")
    return credentials_from_mapping(payload, origin=str(path))


def load_credentials(
    *,
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> OpenSkyCredentials | None:
    """Resolve credentials from the environment, then a JSON file.

    Returns ``None`` when neither source is populated, which callers treat as
    "stay anonymous". An explicit ``path`` that does not exist is an error —
    the caller asked for that file specifically.
    """
    env = os.environ if env is None else env
    client_id = env.get(ENV_CLIENT_ID)
    client_secret = env.get(ENV_CLIENT_SECRET)
    if client_id and client_secret:
        return OpenSkyCredentials(client_id=client_id, client_secret=client_secret)
    if bool(client_id) != bool(client_secret):
        missing = ENV_CLIENT_SECRET if client_id else ENV_CLIENT_ID
        raise OpenSkyAuthError(
            f"{missing} is unset; set both {ENV_CLIENT_ID} and {ENV_CLIENT_SECRET} or neither"
        )

    if path is not None:
        if not path.is_file():
            raise OpenSkyAuthError(f"OpenSky credentials file not found: {path}")
        return _credentials_from_file(path)
    if DEFAULT_CREDENTIALS_PATH.is_file():
        return _credentials_from_file(DEFAULT_CREDENTIALS_PATH)
    return None


class OAuth2ClientCredentialsAuth(httpx.Auth):
    """httpx auth flow implementing OpenSky's ``client_credentials`` grant.

    Mints a bearer token on first use, caches it until
    ``TOKEN_EXPIRY_MARGIN_S`` before expiry, and transparently re-mints once
    on a ``401`` (the token expired mid-capture) before replaying the request.
    Token requests travel over the caller's transport, so tests can drive the
    whole flow through a single ``httpx.MockTransport``.
    """

    requires_response_body = True

    def __init__(
        self,
        credentials: OpenSkyCredentials,
        *,
        token_url: str = OPENSKY_TOKEN_URL,
        clock: Any = time.monotonic,
        expiry_margin_s: float = TOKEN_EXPIRY_MARGIN_S,
    ) -> None:
        self._credentials = credentials
        self._token_url = token_url
        self._clock = clock
        self._expiry_margin_s = expiry_margin_s
        self._token: str | None = None
        self._expires_at: float = 0.0

    @property
    def token(self) -> str | None:
        """The cached bearer token, if one has been minted (tests/diagnostics)."""
        return self._token

    def _build_token_request(self) -> httpx.Request:
        return httpx.Request(
            "POST",
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._credentials.client_id,
                "client_secret": self._credentials.client_secret,
            },
        )

    def _store_token(self, response: httpx.Response) -> None:
        if response.status_code != 200:
            detail = response.text[:500]
            raise OpenSkyAuthError(
                f"OpenSky token request failed: HTTP {response.status_code}: {detail}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenSkyAuthError(f"OpenSky token response was not JSON: {exc}") from exc
        token = payload.get("access_token")
        if not token:
            raise OpenSkyAuthError(
                f"OpenSky token response has no access_token (keys: {sorted(payload)})"
            )
        try:
            lifetime_s = float(payload.get("expires_in", DEFAULT_TOKEN_LIFETIME_S))
        except (TypeError, ValueError):
            lifetime_s = DEFAULT_TOKEN_LIFETIME_S
        self._token = str(token)
        self._expires_at = self._clock() + max(lifetime_s - self._expiry_margin_s, 0.0)

    def _is_fresh(self) -> bool:
        return self._token is not None and self._clock() < self._expires_at

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        if not self._is_fresh():
            self._store_token((yield self._build_token_request()))
        request.headers["Authorization"] = f"Bearer {self._token}"
        response = yield request
        if response.status_code == 401:
            # The token expired (or was revoked) mid-capture: re-mint once and
            # replay. A second 401 propagates to the caller as a normal error.
            self._token = None
            self._store_token((yield self._build_token_request()))
            request.headers["Authorization"] = f"Bearer {self._token}"
            yield request


def build_auth(credentials: OpenSkyCredentials | None) -> OAuth2ClientCredentialsAuth | None:
    """Wrap credentials in an auth flow, or ``None`` for anonymous access."""
    if credentials is None:
        return None
    return OAuth2ClientCredentialsAuth(credentials)
