from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx

from rag_demo.production.auth import Principal, TokenVerifier
from rag_demo.production.config import ProductionSettings


class WebAuthenticationError(ValueError):
    pass


class WebSessionUnavailable(RuntimeError):
    pass


class WebSessionStore(Protocol):
    def put_login_state(self, state: str, payload: dict, ttl_seconds: int) -> None:
        ...

    def pop_login_state(self, state: str) -> Optional[dict]:
        ...

    def put_session(self, session_id: str, access_token: str, ttl_seconds: int) -> None:
        ...

    def get_session_token(self, session_id: str) -> Optional[str]:
        ...

    def delete_session(self, session_id: str) -> None:
        ...


class RedisWebSessionStore:
    def __init__(self, client, *, key_prefix: str = "rag:web-auth"):
        self.client = client
        self.key_prefix = key_prefix.rstrip(":")

    @classmethod
    def from_settings(cls, settings: ProductionSettings):
        if not settings.redis_url:
            raise ValueError("RAG_REDIS_URL is required for browser sessions")
        import redis

        return cls(redis.Redis.from_url(settings.redis_url, decode_responses=True))

    def put_login_state(self, state: str, payload: dict, ttl_seconds: int) -> None:
        try:
            stored = self.client.set(
                self._key("login", state),
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                ex=ttl_seconds,
                nx=True,
            )
        except Exception as exc:
            raise WebSessionUnavailable("login state storage is unavailable") from exc
        if not stored:
            raise WebSessionUnavailable("login state collision")

    def pop_login_state(self, state: str) -> Optional[dict]:
        try:
            raw = self.client.getdel(self._key("login", state))
        except Exception as exc:
            raise WebSessionUnavailable("login state storage is unavailable") from exc
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put_session(self, session_id: str, access_token: str, ttl_seconds: int) -> None:
        try:
            self.client.set(
                self._key("session", session_id),
                access_token,
                ex=ttl_seconds,
            )
        except Exception as exc:
            raise WebSessionUnavailable("browser session storage is unavailable") from exc

    def get_session_token(self, session_id: str) -> Optional[str]:
        try:
            value = self.client.get(self._key("session", session_id))
        except Exception as exc:
            raise WebSessionUnavailable("browser session storage is unavailable") from exc
        return str(value) if value else None

    def delete_session(self, session_id: str) -> None:
        try:
            self.client.delete(self._key("session", session_id))
        except Exception as exc:
            raise WebSessionUnavailable("browser session storage is unavailable") from exc

    def _key(self, kind: str, value: str) -> str:
        return f"{self.key_prefix}:{kind}:{value}"


class InMemoryWebSessionStore:
    """Thread-safe test/development implementation with the same expiry semantics."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._lock = threading.Lock()
        self._login_states: dict[str, tuple[float, dict]] = {}
        self._sessions: dict[str, tuple[float, str]] = {}

    def put_login_state(self, state: str, payload: dict, ttl_seconds: int) -> None:
        with self._lock:
            self._purge()
            if state in self._login_states:
                raise WebSessionUnavailable("login state collision")
            self._login_states[state] = (self.clock() + ttl_seconds, dict(payload))

    def pop_login_state(self, state: str) -> Optional[dict]:
        with self._lock:
            self._purge()
            stored = self._login_states.pop(state, None)
            return dict(stored[1]) if stored else None

    def put_session(self, session_id: str, access_token: str, ttl_seconds: int) -> None:
        with self._lock:
            self._purge()
            self._sessions[session_id] = (self.clock() + ttl_seconds, access_token)

    def get_session_token(self, session_id: str) -> Optional[str]:
        with self._lock:
            self._purge()
            stored = self._sessions.get(session_id)
            return stored[1] if stored else None

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _purge(self) -> None:
        now = self.clock()
        self._login_states = {
            key: value for key, value in self._login_states.items() if value[0] > now
        }
        self._sessions = {
            key: value for key, value in self._sessions.items() if value[0] > now
        }


@dataclass(frozen=True)
class CompletedWebLogin:
    session_id: str
    principal: Principal
    max_age_seconds: int


class OidcWebAuth:
    def __init__(
        self,
        *,
        settings: ProductionSettings,
        token_verifier: TokenVerifier,
        store: WebSessionStore,
        token_exchange: Optional[Callable[[dict], dict]] = None,
    ):
        self.settings = settings
        self.token_verifier = token_verifier
        self.store = store
        self.token_exchange = token_exchange or self._exchange_token

    def begin_login(self) -> str:
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = _pkce_challenge(verifier)
        redirect_uri = f"{self.settings.public_base_url}/auth/callback"
        self.store.put_login_state(
            state,
            {"code_verifier": verifier, "redirect_uri": redirect_uri},
            self.settings.web_login_state_ttl_seconds,
        )
        parameters = {
            "response_type": "code",
            "client_id": self.settings.oidc_client_id,
            "redirect_uri": redirect_uri,
            "scope": self.settings.oidc_scopes,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return _append_query(str(self.settings.oidc_authorization_url), parameters)

    def complete_login(self, *, state: str, code: str) -> CompletedWebLogin:
        clean_state = str(state or "").strip()
        clean_code = str(code or "").strip()
        if not clean_state or not clean_code:
            raise WebAuthenticationError("authorization response is incomplete")
        login_state = self.store.pop_login_state(clean_state)
        if not login_state:
            raise WebAuthenticationError("authorization state is invalid or expired")
        token_request = {
            "grant_type": "authorization_code",
            "client_id": self.settings.oidc_client_id,
            "code": clean_code,
            "redirect_uri": str(login_state.get("redirect_uri") or ""),
            "code_verifier": str(login_state.get("code_verifier") or ""),
        }
        secret = (
            self.settings.oidc_client_secret.get_secret_value()
            if self.settings.oidc_client_secret
            else ""
        )
        if secret:
            token_request["client_secret"] = secret
        response = self.token_exchange(token_request)
        access_token = str(response.get("access_token") or "").strip()
        token_type = str(response.get("token_type") or "Bearer").lower()
        if not access_token or token_type != "bearer":
            raise WebAuthenticationError("identity provider did not return a bearer token")
        try:
            principal = self.token_verifier.verify(access_token)
        except Exception as exc:
            raise WebAuthenticationError("identity provider returned an invalid access token") from exc
        try:
            provider_ttl = int(response.get("expires_in") or self.settings.web_session_ttl_seconds)
        except (TypeError, ValueError):
            provider_ttl = self.settings.web_session_ttl_seconds
        max_age = max(60, min(provider_ttl, self.settings.web_session_ttl_seconds))
        session_id = secrets.token_urlsafe(48)
        self.store.put_session(session_id, access_token, max_age)
        return CompletedWebLogin(
            session_id=session_id,
            principal=principal,
            max_age_seconds=max_age,
        )

    def principal_for_session(self, session_id: str) -> Optional[Principal]:
        clean_id = str(session_id or "").strip()
        if not clean_id:
            return None
        access_token = self.store.get_session_token(clean_id)
        if not access_token:
            return None
        try:
            return self.token_verifier.verify(access_token)
        except Exception:
            self.store.delete_session(clean_id)
            return None

    def logout(self, session_id: str) -> None:
        clean_id = str(session_id or "").strip()
        if clean_id:
            self.store.delete_session(clean_id)

    def _exchange_token(self, payload: dict) -> dict:
        try:
            with httpx.Client(http2=True, timeout=10.0) as client:
                response = client.post(
                    str(self.settings.oidc_token_url),
                    data=payload,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WebAuthenticationError("identity provider token exchange failed") from exc
        if not isinstance(body, dict):
            raise WebAuthenticationError("identity provider returned an invalid token response")
        return body


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _append_query(url: str, parameters: dict) -> str:
    parsed = urlsplit(url)
    query = parsed.query
    encoded = urlencode(parameters)
    combined_query = "&".join(filter(None, [query, encoded]))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, combined_query, parsed.fragment)
    )
