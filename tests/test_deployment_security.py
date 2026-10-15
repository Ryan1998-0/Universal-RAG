import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_demo.production.api import create_app
from rag_demo.production.auth import OidcTokenVerifier, Principal
from rag_demo.production.config import ProductionSettings
from rag_demo.production.preflight import check_environment, main as preflight_main
from rag_demo.production.web_auth import InMemoryWebSessionStore, OidcWebAuth, WebAuthenticationError


@pytest.fixture
def settings():
    return ProductionSettings(
        _env_file=None, RAG_ENV="test", RAG_DATABASE_URL="sqlite+pysqlite:///:memory:",
        RAG_AUTH_MODE="dev_hs256", RAG_DEV_JWT_SECRET="a-test-secret-with-at-least-32-characters",
        RAG_PUBLIC_BASE_URL="http://127.0.0.1:8080",
        RAG_OIDC_AUTHORIZATION_URL="http://127.0.0.1:9999/authorize",
        RAG_OIDC_TOKEN_URL="http://127.0.0.1:9999/token", RAG_OIDC_CLIENT_ID="rag-web",
    )


@pytest.fixture
def login_setup(settings):
    verifier = Mock()
    verifier.verify.return_value = Principal("subject-a", "tenant-a", frozenset({"member"}))
    exchange = Mock(return_value={"access_token": "valid-test-token", "expires_in": 900})
    auth = OidcWebAuth(settings=settings, token_verifier=verifier, store=InMemoryWebSessionStore(), token_exchange=exchange)
    app = create_app(settings=settings, repository=object(), pipeline=object(), token_verifier=verifier, web_auth=auth)
    return app, auth, exchange


@pytest.mark.parametrize("binding", ["", "another-browser"])
def test_login_state_is_bound_to_initiating_browser(login_setup, binding):
    _, auth, exchange = login_setup
    state = parse_qs(urlsplit(auth.begin_login("original-browser")).query)["state"][0]
    with pytest.raises(WebAuthenticationError):
        auth.complete_login(state=state, code="attacker-code", browser_binding=binding)
    exchange.assert_not_called()


def test_foreign_browser_callback_cannot_create_session(login_setup):
    app, _, exchange = login_setup
    with TestClient(app, base_url="http://127.0.0.1:8080") as original, TestClient(app, base_url="http://127.0.0.1:8080") as foreign:
        login = original.get("/auth/login", follow_redirects=False)
        state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
        assert "HttpOnly" in login.headers["set-cookie"]
        callback = foreign.get(f"/auth/callback?state={state}&code=stolen", follow_redirects=False)
        assert callback.status_code == 400
        assert not foreign.get("/auth/status").json()["authenticated"]
        exchange.assert_not_called()


def test_cookie_mutations_require_same_origin_bearer_does_not(login_setup):
    app, _, _ = login_setup
    with TestClient(app, base_url="http://127.0.0.1:8080") as client:
        login = client.get("/auth/login", follow_redirects=False)
        state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
        assert client.get(f"/auth/callback?state={state}&code=ok", follow_redirects=False).status_code == 303
        for headers in ({}, {"Origin": "https://evil.test"}, {"Origin": "http://127.0.0.1:8080", "Sec-Fetch-Site": "cross-site"}):
            response = client.post("/v1/models/validate", json={"node": "generation", "model": "ollama:qwen2.5:7b"}, headers=headers)
            assert response.status_code == 403
            assert response.json()["error"]["code"] == "ORIGIN_DENIED"
        for headers in ({"Origin": "http://127.0.0.1:8080"}, {"Sec-Fetch-Site": "same-origin"}, {"Authorization": "Bearer valid-test-token"}):
            assert client.post("/v1/models/validate", json={"node": "generation", "model": "ollama:qwen2.5:7b"}, headers=headers).status_code == 200


def test_oidc_issuer_trailing_slash_is_preserved_and_verified():
    import time
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = OidcTokenVerifier("https://id.test/", "rag-api", "https://id.test/jwks", "tenant_id", "roles")
    verifier.jwks_client = Mock()
    verifier.jwks_client.get_signing_key_from_jwt.return_value = SimpleNamespace(key=key.public_key())
    claims = {"iss": "https://id.test/", "aud": "rag-api", "sub": "u", "tenant_id": "t", "iat": int(time.time()), "exp": int(time.time()) + 60}
    assert verifier.verify(jwt.encode(claims, key, algorithm="RS256")).tenant_id == "t"
    claims["iss"] = "https://id.test"
    from rag_demo.production.auth import AuthenticationError
    with pytest.raises(AuthenticationError):
        verifier.verify(jwt.encode(claims, key, algorithm="RS256"))


@pytest.mark.parametrize("field,value", [
    ("RAG_OIDC_ISSUER", "http://id.test"), ("RAG_OIDC_JWKS_URL", "http://id.test/jwks"),
    ("RAG_OIDC_TOKEN_URL", "http://127.0.0.1.evil.test/token"),
    ("RAG_OIDC_TOKEN_URL", "https://user:secret@id.test/token"),
    ("RAG_OIDC_TOKEN_URL", "https://id.test/token#fragment"),
    ("RAG_PUBLIC_BASE_URL", "https://rag.test/subpath"),
    ("RAG_CORS_ORIGINS", "*"), ("RAG_CORS_ORIGINS", "https://rag.test/path"),
])
def test_unsafe_identity_urls_and_cors_fail_closed(settings, field, value):
    values = settings.model_dump(by_alias=True)
    values[field] = value
    with pytest.raises(ValidationError):
        ProductionSettings(_env_file=None, **values)


@pytest.mark.parametrize("declared_length", [None, "1"])
def test_streaming_body_limit_uses_received_bytes(settings, declared_length):
    settings = settings.model_copy(update={"max_request_body_bytes": 1024})
    app = create_app(settings=settings, repository=object(), pipeline=object())

    @app.post("/read-body")
    async def read_body(request: Request):
        return {"size": len(await request.body())}

    headers = {"X-Request-ID": "stream-test"}
    if declared_length:
        headers["Content-Length"] = declared_length
    with TestClient(app) as client:
        response = client.post("/read-body", content=iter([b"x" * 600, b"x" * 600]), headers=headers)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
        assert response.json()["request_id"] == "stream-test"
        assert response.headers["cache-control"] == "no-store"
        assert client.post("/read-body", content=iter([b"x" * 1024])).json() == {"size": 1024}


def test_error_contract_redacts_validation_inputs_and_unexpected_errors(settings):
    verifier = Mock()
    verifier.verify.return_value = Principal("u", "t", frozenset())
    app = create_app(settings=settings, repository=object(), pipeline=object(), token_verifier=verifier)

    @app.get("/explode")
    async def explode():
        raise RuntimeError("database-password-should-never-leak")

    with TestClient(app) as client:
        responses = [
            client.post("/v1/ask", json={"question": "private-user-input", "unknown": "secret"}, headers={"Authorization": "Bearer test"}),
            client.get("/missing"), client.patch("/v1/runtime"), client.get("/explode"),
        ]
    assert [r.status_code for r in responses] == [422, 404, 405, 500]
    for response in responses:
        assert response.headers["x-request-id"] == response.json()["request_id"]
        assert set(response.json()) == {"error", "request_id"}
        assert response.headers["cache-control"] == "no-store"
        assert "private-user-input" not in response.text
        assert "database-password" not in response.text


def valid_environment():
    return {
        "APP_DOMAIN": "rag.company.test", "ACME_EMAIL": "ops@company.test",
        "POSTGRES_PASSWORD": "a" * 32, "REDIS_PASSWORD": "b" * 32,
        "QDRANT_API_KEY": "c" * 32, "S3_ACCESS_KEY_ID": "access-id",
        "S3_SECRET_ACCESS_KEY": "d" * 32,
        "RAG_OIDC_ISSUER": "https://identity.company.test/",
        "RAG_OIDC_AUDIENCE": "rag-api", "RAG_OIDC_JWKS_URL": "https://identity.company.test/jwks",
        "RAG_OIDC_AUTHORIZATION_URL": "https://identity.company.test/authorize",
        "RAG_OIDC_TOKEN_URL": "https://identity.company.test/token", "RAG_OIDC_CLIENT_ID": "rag-web",
        "RAG_CORS_ORIGINS": "https://rag.company.test", "RAG_OLLAMA_URL": "http://inference:11434",
    }


def test_preflight_rejects_templates_reused_secrets_and_unsupported_presigned():
    values = valid_environment()
    assert check_environment(values) == []
    values.update(POSTGRES_PASSWORD="replace-with-a-long-random-secret", REDIS_PASSWORD="same-secret", QDRANT_API_KEY="same-secret", RAG_UPLOAD_MODE="presigned")
    issues = check_environment(values)
    assert {i["code"] for i in issues} >= {"PLACEHOLDER", "REUSED_SECRET", "SECRET_TOO_SHORT", "COMPOSE_SUPPORTS_PROXY_ONLY"}
    assert "same-secret" not in json.dumps(issues)


def test_preflight_cli_redacts_secrets(tmp_path, capsys, monkeypatch):
    for key in valid_environment():
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "deploy.env"
    path.write_text("\n".join(f"{k}={v}" for k, v in valid_environment().items()), encoding="utf-8")
    assert preflight_main(["--env-file", str(path), "--skip-docker"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed" and not report["docker_checked"]
    assert "a" * 32 not in json.dumps(report)


def test_production_model_overrides_cannot_affect_other_tenants(settings):
    values = settings.model_dump(by_alias=True)
    values.update({
        "RAG_ENV": "production", "RAG_AUTH_MODE": "oidc",
        "RAG_DATABASE_URL": "postgresql+psycopg://user:pass@postgres/rag",
        "RAG_REDIS_URL": "redis://redis:6379", "RAG_QDRANT_URL": "http://qdrant:6333",
        "RAG_S3_ENDPOINT_URL": "http://s3:8333", "RAG_S3_ACCESS_KEY_ID": "test-id",
        "RAG_S3_SECRET_ACCESS_KEY": "test-secret", "RAG_CLAMAV_HOST": "clamav",
        "RAG_OIDC_ISSUER": "https://identity.test/", "RAG_OIDC_AUDIENCE": "rag-api",
        "RAG_OIDC_JWKS_URL": "https://identity.test/jwks",
        "RAG_OIDC_AUTHORIZATION_URL": "https://identity.test/authorize",
        "RAG_OIDC_TOKEN_URL": "https://identity.test/token",
        "RAG_PUBLIC_BASE_URL": "https://rag.test",
    })
    production_settings = ProductionSettings(_env_file=None, **values)
    verifier = Mock()
    verifier.verify.return_value = Principal("tenant-owner", "tenant-a", frozenset({"owner"}))
    app = create_app(
        settings=production_settings, repository=object(), pipeline=object(), token_verifier=verifier,
        rate_limiter=Mock(), vector_repository=Mock(), object_storage=Mock(), inference_probe=Mock(),
        task_dispatcher=Mock(), generation_gate=Mock(), malware_scanner=Mock(), background_health_probe=Mock(), web_auth=Mock(),
    )
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer valid-test"}
        for method in ("PUT", "DELETE"):
            response = client.request(method, "/v1/models/generation", headers=headers, **({"json": {"model": "ollama:qwen2.5:7b"}} if method == "PUT" else {}))
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "MODEL_BINDINGS_READ_ONLY"
        assert client.get("/v1/models", headers=headers).status_code == 200


def test_preflight_docker_validation_never_echoes_invalid_settings(tmp_path, capsys, monkeypatch):
    from rag_demo.production import preflight
    values = valid_environment()
    for key in values:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / "deployment.env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in values.items()), encoding="utf-8")
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "fake-docker")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps({
        "services": {"api": {"environment": {
            "RAG_ENV": "production", "RAG_AUTH_MODE": "dev_hs256",
            "RAG_DATABASE_URL": "postgresql+psycopg://user:never-echo-this-password@postgres/rag",
        }}}
    })))
    assert preflight_main(["--env-file", str(env_file)]) == 1
    output = capsys.readouterr().out
    assert "never-echo-this-password" not in output
    assert json.loads(output)["issues"][0]["code"] == "INVALID_SETTING"
