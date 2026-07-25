"""Tests for OpenSky OAuth2 client-credentials authentication.

The whole flow (token mint, caching, 401 re-mint, replay) is driven through a
single ``httpx.MockTransport``, so no network or real credentials are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from acquisition.opensky_auth import (
    ENV_CLIENT_ID,
    ENV_CLIENT_SECRET,
    OAuth2ClientCredentialsAuth,
    OpenSkyAuthError,
    OpenSkyCredentials,
    build_auth,
    credentials_from_mapping,
    load_credentials,
)

TOKEN_URL = "https://auth.example/token"
API_URL = "https://api.example/states/all"
CREDS = OpenSkyCredentials(client_id="cid", client_secret="csec")


class FakeClock:
    """Monotonic clock the tests advance explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _token_body(token: str, expires_in: float = 1800.0) -> dict[str, object]:
    return {"access_token": token, "expires_in": expires_in, "token_type": "bearer"}


def make_transport(
    calls: list[httpx.Request],
    *,
    api_responses: list[httpx.Response] | None = None,
    tokens: list[str] | None = None,
) -> httpx.MockTransport:
    """Mock transport serving the token endpoint and the API endpoint."""
    token_queue = list(tokens or ["tok-1", "tok-2", "tok-3"])
    api_queue = list(api_responses or [])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json=_token_body(token_queue.pop(0)))
        if api_queue:
            return api_queue.pop(0)
        return httpx.Response(200, json={"time": 1, "states": []})

    return httpx.MockTransport(handler)


def build_client(transport: httpx.MockTransport, auth: httpx.Auth) -> httpx.Client:
    return httpx.Client(transport=transport, auth=auth)


# --- credential resolution -------------------------------------------------


def test_env_vars_take_precedence_over_file(tmp_path: Path) -> None:
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps({"clientId": "from-file", "clientSecret": "s"}))
    resolved = load_credentials(
        path=creds_file,
        env={ENV_CLIENT_ID: "from-env", ENV_CLIENT_SECRET: "secret"},
    )
    assert resolved == OpenSkyCredentials(client_id="from-env", client_secret="secret")


def test_loads_opensky_camelcase_credentials_file(tmp_path: Path) -> None:
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps({"clientId": "cid", "clientSecret": "csec"}))
    assert load_credentials(path=creds_file, env={}) == CREDS


def test_loads_snake_case_credentials_file(tmp_path: Path) -> None:
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps({"client_id": "cid", "client_secret": "csec"}))
    assert load_credentials(path=creds_file, env={}) == CREDS


def test_half_set_env_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(OpenSkyAuthError, match=ENV_CLIENT_SECRET):
        load_credentials(path=None, env={ENV_CLIENT_ID: "cid"})


def test_missing_explicit_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(OpenSkyAuthError, match="not found"):
        load_credentials(path=tmp_path / "absent.json", env={})


def test_malformed_credentials_file_names_the_expected_keys(tmp_path: Path) -> None:
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps({"user": "u", "password": "p"}))
    with pytest.raises(OpenSkyAuthError, match="clientId and clientSecret"):
        load_credentials(path=creds_file, env={})


def test_credentials_from_mapping_rejects_empty_values() -> None:
    with pytest.raises(OpenSkyAuthError):
        credentials_from_mapping({"clientId": "", "clientSecret": "s"}, origin="test")


def test_secret_is_redacted_in_repr() -> None:
    assert "csec" not in repr(CREDS)
    assert "cid" in repr(CREDS)


def test_build_auth_returns_none_when_anonymous() -> None:
    assert build_auth(None) is None
    assert isinstance(build_auth(CREDS), OAuth2ClientCredentialsAuth)


# --- token flow ------------------------------------------------------------


def test_mints_token_and_attaches_bearer_header() -> None:
    calls: list[httpx.Request] = []
    auth = OAuth2ClientCredentialsAuth(CREDS, token_url=TOKEN_URL)
    with build_client(make_transport(calls), auth) as client:
        client.get(API_URL)

    assert str(calls[0].url) == TOKEN_URL
    token_form = dict(httpx.QueryParams(calls[0].content.decode()))
    assert token_form["grant_type"] == "client_credentials"
    assert token_form["client_id"] == "cid"
    assert token_form["client_secret"] == "csec"
    assert calls[1].headers["Authorization"] == "Bearer tok-1"


def test_token_is_cached_across_requests() -> None:
    calls: list[httpx.Request] = []
    auth = OAuth2ClientCredentialsAuth(CREDS, token_url=TOKEN_URL, clock=FakeClock())
    with build_client(make_transport(calls), auth) as client:
        client.get(API_URL)
        client.get(API_URL)

    token_calls = [c for c in calls if str(c.url) == TOKEN_URL]
    assert len(token_calls) == 1, "second request should reuse the cached token"


def test_token_is_reminted_after_expiry_margin() -> None:
    calls: list[httpx.Request] = []
    clock = FakeClock()
    auth = OAuth2ClientCredentialsAuth(
        CREDS, token_url=TOKEN_URL, clock=clock, expiry_margin_s=60.0
    )
    with build_client(make_transport(calls), auth) as client:
        client.get(API_URL)
        clock.now = 1741.0  # past 1800 - 60
        client.get(API_URL)

    token_calls = [c for c in calls if str(c.url) == TOKEN_URL]
    assert len(token_calls) == 2
    api_calls = [c for c in calls if str(c.url) != TOKEN_URL]
    assert api_calls[1].headers["Authorization"] == "Bearer tok-2"


def test_401_triggers_remint_and_replay() -> None:
    calls: list[httpx.Request] = []
    transport = make_transport(
        calls,
        api_responses=[
            httpx.Response(401, text="token expired"),
            httpx.Response(200, json={"time": 1, "states": []}),
        ],
    )
    auth = OAuth2ClientCredentialsAuth(CREDS, token_url=TOKEN_URL, clock=FakeClock())
    with build_client(transport, auth) as client:
        response = client.get(API_URL)

    assert response.status_code == 200
    urls = [str(c.url) for c in calls]
    assert urls == [TOKEN_URL, API_URL, TOKEN_URL, API_URL]
    assert calls[3].headers["Authorization"] == "Bearer tok-2"


def test_second_401_propagates_instead_of_looping() -> None:
    calls: list[httpx.Request] = []
    transport = make_transport(
        calls,
        api_responses=[httpx.Response(401, text="nope"), httpx.Response(401, text="nope")],
    )
    auth = OAuth2ClientCredentialsAuth(CREDS, token_url=TOKEN_URL, clock=FakeClock())
    with build_client(transport, auth) as client:
        response = client.get(API_URL)

    assert response.status_code == 401
    assert len([c for c in calls if str(c.url) == TOKEN_URL]) == 2


def test_token_endpoint_failure_raises_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid_client")

    auth = OAuth2ClientCredentialsAuth(CREDS, token_url=TOKEN_URL)
    with build_client(httpx.MockTransport(handler), auth) as client:  # noqa: SIM117
        with pytest.raises(OpenSkyAuthError, match="invalid_client"):
            client.get(API_URL)


def test_token_response_without_access_token_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"expires_in": 1800})

    auth = OAuth2ClientCredentialsAuth(CREDS, token_url=TOKEN_URL)
    with build_client(httpx.MockTransport(handler), auth) as client:  # noqa: SIM117
        with pytest.raises(OpenSkyAuthError, match="no access_token"):
            client.get(API_URL)
