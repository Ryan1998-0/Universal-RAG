"""Read-only deployment checks. Never print rendered Compose secrets."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import ValidationError

from rag_demo.production.config import ProductionSettings


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "APP_DOMAIN", "ACME_EMAIL", "POSTGRES_PASSWORD", "REDIS_PASSWORD",
    "QDRANT_API_KEY", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY",
    "RAG_OIDC_ISSUER", "RAG_OIDC_AUDIENCE", "RAG_OIDC_JWKS_URL",
    "RAG_OIDC_AUTHORIZATION_URL", "RAG_OIDC_TOKEN_URL", "RAG_OIDC_CLIENT_ID",
    "RAG_CORS_ORIGINS", "RAG_OLLAMA_URL",
)
SECRETS = ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "QDRANT_API_KEY", "S3_SECRET_ACCESS_KEY")


def check_environment(values: dict) -> list[dict]:
    issues = []

    def fail(field: str, code: str):
        issues.append({"field": field, "code": code})

    for field in REQUIRED:
        value = str(values.get(field) or "").strip()
        if not value:
            fail(field, "REQUIRED")
        elif "replace-with" in value.lower() or "example.com" in value.lower():
            fail(field, "PLACEHOLDER")
    for field in SECRETS:
        value = str(values.get(field) or "")
        if len(value) < 24:
            fail(field, "SECRET_TOO_SHORT")
        # These two values are interpolated directly into connection URIs.
        if field in {"POSTGRES_PASSWORD", "REDIS_PASSWORD"} and not re.fullmatch(r"[A-Za-z0-9_-]{24,}", value):
            fail(field, "USE_URL_SAFE_SECRET")
    if len({values.get(field) for field in SECRETS}) != len(SECRETS):
        fail("secrets", "REUSED_SECRET")
    domain = str(values.get("APP_DOMAIN") or "")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", domain) or "." not in domain:
        fail("APP_DOMAIN", "INVALID_HOSTNAME")
    if str(values.get("RAG_CORS_ORIGINS") or "").rstrip("/") != f"https://{domain}":
        fail("RAG_CORS_ORIGINS", "COMPOSE_REQUIRES_SAME_ORIGIN")
    if str(values.get("RAG_UPLOAD_MODE") or "proxy") != "proxy":
        fail("RAG_UPLOAD_MODE", "COMPOSE_SUPPORTS_PROXY_ONLY")
    for field in ("RAG_OIDC_ISSUER", "RAG_OIDC_JWKS_URL", "RAG_OIDC_AUTHORIZATION_URL", "RAG_OIDC_TOKEN_URL"):
        try:
            parsed = urlsplit(str(values.get(field) or ""))
            parsed.port
            invalid = parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment
        except ValueError:
            invalid = True
        if invalid:
            fail(field, "HTTPS_URL_REQUIRED")
    return issues


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.production")
    parser.add_argument("--skip-docker", action="store_true", help="Only check local configuration; does not validate Compose")
    args = parser.parse_args(argv)
    env_path = args.env_file.resolve()
    if not env_path.is_file():
        print(json.dumps({"status": "failed", "issues": [{"field": "env_file", "code": "NOT_FOUND"}]}))
        return 1
    # Compose gives process environment precedence over its --env-file.
    values = {**dotenv_values(env_path), **os.environ}
    issues = check_environment(values)
    checks = ["environment"]
    if not args.skip_docker:
        docker = shutil.which("docker")
        if not docker:
            issues.append({"field": "docker", "code": "NOT_INSTALLED"})
        elif not issues:
            try:
                result = subprocess.run(
                    [docker, "compose", "--project-directory", str(ROOT), "--env-file", str(env_path),
                     "-f", str(ROOT / "compose.yaml"), "config", "--format", "json"],
                    capture_output=True, text=True, encoding="utf-8", timeout=45, check=True,
                )
                rendered = json.loads(result.stdout)
                ProductionSettings(_env_file=None, **rendered["services"]["api"]["environment"])
                checks.append("compose-and-api-settings")
            except ValidationError as exc:
                # Do not stringify ValidationError: it includes original inputs.
                issues.extend({"field": ".".join(map(str, e["loc"])), "code": "INVALID_SETTING"} for e in exc.errors())
            except (subprocess.SubprocessError, OSError, ValueError, KeyError):
                issues.append({"field": "compose", "code": "RENDER_FAILED"})
    print(json.dumps({
        "status": "failed" if issues else "passed", "checks": checks,
        "docker_checked": not args.skip_docker and "compose-and-api-settings" in checks,
        "issues": issues,
    }, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
