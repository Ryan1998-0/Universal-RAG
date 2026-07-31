from dataclasses import dataclass
from typing import FrozenSet, Mapping, Protocol

import jwt
from jwt import PyJWKClient

from rag_demo.production.config import ProductionSettings


class AuthenticationError(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    roles: FrozenSet[str]

    def has_role(self, role: str) -> bool:
        return role in self.roles


class TokenVerifier(Protocol):
    def verify(self, token: str) -> Principal:
        ...


class Hs256TokenVerifier:
    def __init__(self, secret: str, audience: str = "ifrs17-rag-dev"):
        if len(secret) < 32:
            raise ValueError("Development JWT secret must contain at least 32 characters.")
        self.secret = secret
        self.audience = audience

    def verify(self, token: str) -> Principal:
        try:
            claims = jwt.decode(
                token,
                self.secret,
                algorithms=["HS256"],
                audience=self.audience,
                options={"require": ["exp", "iat", "sub", "tenant_id"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError("invalid access token") from exc
        return principal_from_claims(claims, tenant_claim="tenant_id", roles_claim="roles")


class OidcTokenVerifier:
    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_url: str,
        tenant_claim: str,
        roles_claim: str,
    ):
        self.issuer = issuer.rstrip("/")
        self.audience = audience
        self.tenant_claim = tenant_claim
        self.roles_claim = roles_claim
        self.jwks_client = PyJWKClient(jwks_url, cache_keys=True)

    def verify(self, token: str) -> Principal:
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "sub"]},
            )
        except (jwt.PyJWTError, ValueError) as exc:
            raise AuthenticationError("invalid access token") from exc
        return principal_from_claims(
            claims,
            tenant_claim=self.tenant_claim,
            roles_claim=self.roles_claim,
        )


def build_token_verifier(settings: ProductionSettings) -> TokenVerifier:
    if settings.auth_mode == "dev_hs256":
        secret = settings.dev_jwt_secret.get_secret_value()
        return Hs256TokenVerifier(secret=secret)
    return OidcTokenVerifier(
        issuer=str(settings.oidc_issuer),
        audience=str(settings.oidc_audience),
        jwks_url=str(settings.oidc_jwks_url),
        tenant_claim=settings.oidc_tenant_claim,
        roles_claim=settings.oidc_roles_claim,
    )


def principal_from_claims(
    claims: Mapping[str, object],
    tenant_claim: str,
    roles_claim: str,
) -> Principal:
    subject = str(claims.get("sub") or "").strip()
    tenant_id = str(claims.get(tenant_claim) or "").strip()
    if not subject or not tenant_id:
        raise AuthenticationError("access token is missing required identity claims")

    raw_roles = claims.get(roles_claim) or []
    if isinstance(raw_roles, str):
        roles = frozenset(role for role in raw_roles.split() if role)
    elif isinstance(raw_roles, (list, tuple, set)):
        roles = frozenset(str(role).strip() for role in raw_roles if str(role).strip())
    else:
        roles = frozenset()
    return Principal(subject=subject, tenant_id=tenant_id, roles=roles)
