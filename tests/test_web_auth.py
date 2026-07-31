import unittest
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import jwt
from fastapi.testclient import TestClient

from rag_demo.production.api import create_app
from rag_demo.production.auth import Hs256TokenVerifier
from rag_demo.production.config import ProductionSettings
from rag_demo.production.web_auth import (
    InMemoryWebSessionStore,
    OidcWebAuth,
    WebAuthenticationError,
)


SECRET = "web-auth-test-secret-with-at-least-32-characters"


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class WebAuthTests(unittest.TestCase):
    def setUp(self):
        self.settings = ProductionSettings(
            RAG_ENV="test",
            RAG_DATABASE_URL="sqlite+pysqlite:///:memory:",
            RAG_AUTH_MODE="dev_hs256",
            RAG_DEV_JWT_SECRET=SECRET,
            RAG_OIDC_AUTHORIZATION_URL="http://127.0.0.1:9999/authorize",
            RAG_OIDC_TOKEN_URL="http://127.0.0.1:9999/token",
            RAG_OIDC_CLIENT_ID="rag-web",
            RAG_PUBLIC_BASE_URL="http://127.0.0.1:8080",
            RAG_WEB_SESSION_TTL_SECONDS=3600,
            RAG_WEB_LOGIN_STATE_TTL_SECONDS=120,
        )
        self.verifier = Hs256TokenVerifier(SECRET)
        self.clock = Clock()
        self.store = InMemoryWebSessionStore(clock=self.clock)
        self.exchanges = []
        self.web_auth = OidcWebAuth(
            settings=self.settings,
            token_verifier=self.verifier,
            store=self.store,
            token_exchange=self._exchange,
        )

    def _token(self):
        now = datetime.now(timezone.utc)
        return jwt.encode(
            {
                "sub": "subject-a",
                "tenant_id": "tenant-a",
                "roles": ["member"],
                "aud": "universal-rag-dev",
                "iat": now,
                "exp": now + timedelta(minutes=30),
            },
            SECRET,
            algorithm="HS256",
        )

    def _exchange(self, payload):
        self.exchanges.append(dict(payload))
        return {"access_token": self._token(), "token_type": "Bearer", "expires_in": 900}

    def test_pkce_state_is_single_use_and_session_resolves_principal(self):
        location = self.web_auth.begin_login()
        query = parse_qs(urlparse(location).query)
        self.assertEqual(query["client_id"], ["rag-web"])
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertNotIn("code_verifier", query)

        completed = self.web_auth.complete_login(state=query["state"][0], code="auth-code")
        self.assertEqual(completed.principal.subject, "subject-a")
        self.assertEqual(completed.max_age_seconds, 900)
        self.assertEqual(
            self.web_auth.principal_for_session(completed.session_id).tenant_id,
            "tenant-a",
        )
        self.assertNotEqual(
            self.exchanges[0]["code_verifier"],
            query["code_challenge"][0],
        )
        with self.assertRaises(WebAuthenticationError):
            self.web_auth.complete_login(state=query["state"][0], code="replay")

    def test_expired_login_state_and_session_are_rejected(self):
        location = self.web_auth.begin_login()
        state = parse_qs(urlparse(location).query)["state"][0]
        self.clock.value += 121
        with self.assertRaises(WebAuthenticationError):
            self.web_auth.complete_login(state=state, code="too-late")

        location = self.web_auth.begin_login()
        state = parse_qs(urlparse(location).query)["state"][0]
        completed = self.web_auth.complete_login(state=state, code="fresh")
        self.clock.value += 901
        self.assertIsNone(self.web_auth.principal_for_session(completed.session_id))

    def test_browser_routes_set_http_only_cookie_and_enforce_logout_origin(self):
        app = create_app(
            settings=self.settings,
            repository=object(),
            pipeline=object(),
            token_verifier=self.verifier,
            web_auth=self.web_auth,
        )
        with TestClient(app, base_url="http://127.0.0.1:8080") as client:
            login = client.get("/auth/login", follow_redirects=False)
            self.assertEqual(login.status_code, 303)
            state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]

            callback = client.get(
                f"/auth/callback?state={state}&code=browser-code",
                follow_redirects=False,
            )
            self.assertEqual(callback.status_code, 303, callback.text)
            self.assertIn("HttpOnly", callback.headers["set-cookie"])
            self.assertIn("SameSite=lax", callback.headers["set-cookie"])
            self.assertTrue(client.get("/auth/status").json()["authenticated"])
            self.assertEqual(client.get("/v1/runtime").status_code, 200)

            denied = client.post(
                "/auth/logout",
                headers={"Origin": "https://attacker.invalid"},
            )
            self.assertEqual(denied.status_code, 403)
            self.assertTrue(client.get("/auth/status").json()["authenticated"])

            logged_out = client.post(
                "/auth/logout",
                headers={"Origin": "http://127.0.0.1:8080"},
            )
            self.assertEqual(logged_out.status_code, 204)
            self.assertFalse(client.get("/auth/status").json()["authenticated"])


if __name__ == "__main__":
    unittest.main()
