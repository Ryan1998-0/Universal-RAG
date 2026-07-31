from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


Environment = Literal["development", "test", "staging", "production"]
AuthMode = Literal["dev_hs256", "oidc"]
UploadMode = Literal["proxy", "presigned"]
PromptInjectionPolicy = Literal["quarantine", "flag"]


class ProductionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Environment = Field(default="development", alias="RAG_ENV")
    service_name: str = Field(default="ifrs17-rag", alias="RAG_SERVICE_NAME")
    service_version: str = Field(default="dev", alias="RAG_SERVICE_VERSION")
    database_url: str = Field(alias="RAG_DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, alias="RAG_REDIS_URL")
    qdrant_url: Optional[str] = Field(default=None, alias="RAG_QDRANT_URL")
    qdrant_api_key: Optional[SecretStr] = Field(default=None, alias="RAG_QDRANT_API_KEY")
    qdrant_collection: str = Field(default="rag_chunks", alias="RAG_QDRANT_COLLECTION")
    s3_endpoint_url: Optional[str] = Field(default=None, alias="RAG_S3_ENDPOINT_URL")
    s3_public_endpoint_url: Optional[str] = Field(
        default=None,
        alias="RAG_S3_PUBLIC_ENDPOINT_URL",
    )
    s3_access_key_id: Optional[str] = Field(default=None, alias="RAG_S3_ACCESS_KEY_ID")
    s3_secret_access_key: Optional[SecretStr] = Field(default=None, alias="RAG_S3_SECRET_ACCESS_KEY")
    s3_bucket: str = Field(default="rag-documents", alias="RAG_S3_BUCKET")
    s3_region: str = Field(default="us-east-1", alias="RAG_S3_REGION")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="RAG_EMBEDDING_MODEL",
    )
    embedding_dimensions: int = Field(
        default=384,
        ge=1,
        le=65_536,
        alias="RAG_EMBEDDING_DIMENSIONS",
    )
    sparse_embedding_model: str = Field(
        default="Qdrant/bm25",
        alias="RAG_SPARSE_EMBEDDING_MODEL",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        alias="RAG_RERANKER_MODEL",
    )
    inference_base_url: str = Field(
        default="http://127.0.0.1:11434",
        alias="RAG_OLLAMA_URL",
    )

    auth_mode: AuthMode = Field(default="dev_hs256", alias="RAG_AUTH_MODE")
    dev_jwt_secret: Optional[SecretStr] = Field(default=None, alias="RAG_DEV_JWT_SECRET")
    oidc_issuer: Optional[str] = Field(default=None, alias="RAG_OIDC_ISSUER")
    oidc_audience: Optional[str] = Field(default=None, alias="RAG_OIDC_AUDIENCE")
    oidc_jwks_url: Optional[str] = Field(default=None, alias="RAG_OIDC_JWKS_URL")
    oidc_tenant_claim: str = Field(default="tenant_id", alias="RAG_OIDC_TENANT_CLAIM")
    oidc_roles_claim: str = Field(default="roles", alias="RAG_OIDC_ROLES_CLAIM")
    oidc_authorization_url: Optional[str] = Field(
        default=None,
        alias="RAG_OIDC_AUTHORIZATION_URL",
    )
    oidc_token_url: Optional[str] = Field(default=None, alias="RAG_OIDC_TOKEN_URL")
    oidc_client_id: Optional[str] = Field(default=None, alias="RAG_OIDC_CLIENT_ID")
    oidc_client_secret: Optional[SecretStr] = Field(
        default=None,
        alias="RAG_OIDC_CLIENT_SECRET",
    )
    oidc_scopes: str = Field(default="openid profile email", alias="RAG_OIDC_SCOPES")
    public_base_url: Optional[str] = Field(default=None, alias="RAG_PUBLIC_BASE_URL")
    web_session_cookie_name: str = Field(
        default="rag_session",
        alias="RAG_WEB_SESSION_COOKIE_NAME",
    )
    web_session_ttl_seconds: int = Field(
        default=8 * 60 * 60,
        ge=300,
        le=7 * 24 * 60 * 60,
        alias="RAG_WEB_SESSION_TTL_SECONDS",
    )
    web_login_state_ttl_seconds: int = Field(
        default=600,
        ge=60,
        le=1800,
        alias="RAG_WEB_LOGIN_STATE_TTL_SECONDS",
    )

    default_model: str = Field(default="ollama:qwen2.5:7b", alias="RAG_MODEL")
    allowed_models_csv: str = Field(default="", alias="RAG_ALLOWED_MODELS")
    cors_origins_csv: str = Field(default="", alias="RAG_CORS_ORIGINS")
    max_concurrent_generations: int = Field(
        default=5,
        ge=1,
        le=100,
        alias="RAG_MAX_CONCURRENT_GENERATIONS",
    )
    request_timeout_seconds: int = Field(
        default=180,
        ge=5,
        le=900,
        alias="RAG_REQUEST_TIMEOUT_SECONDS",
    )
    generation_queue_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        alias="RAG_GENERATION_QUEUE_TIMEOUT_SECONDS",
    )
    enable_docs: bool = Field(default=False, alias="RAG_ENABLE_API_DOCS")
    enable_metrics: bool = Field(default=True, alias="RAG_ENABLE_METRICS")
    ask_rate_limit_per_minute: int = Field(
        default=30,
        ge=1,
        le=10_000,
        alias="RAG_ASK_RATE_LIMIT_PER_MINUTE",
    )
    max_request_body_bytes: int = Field(
        default=55 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024,
        alias="RAG_MAX_REQUEST_BODY_BYTES",
    )
    max_upload_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1024,
        le=1024 * 1024 * 1024,
        alias="RAG_MAX_UPLOAD_BYTES",
    )
    upload_session_ttl_seconds: int = Field(
        default=900,
        ge=60,
        le=3600,
        alias="RAG_UPLOAD_SESSION_TTL_SECONDS",
    )
    upload_mode: UploadMode = Field(default="proxy", alias="RAG_UPLOAD_MODE")
    parser_version: str = Field(default="canonical-v1", alias="RAG_PARSER_VERSION")
    chunk_schema_version: str = Field(default="parent-child-v1", alias="RAG_CHUNK_SCHEMA_VERSION")
    index_batch_size: int = Field(
        default=64,
        ge=1,
        le=512,
        alias="RAG_INDEX_BATCH_SIZE",
    )
    ocr_languages: str = Field(default="chi_tra+eng", alias="RAG_OCR_LANGUAGES")
    ingestion_lease_seconds: int = Field(
        default=600,
        ge=60,
        le=3600,
        alias="RAG_INGESTION_LEASE_SECONDS",
    )
    ingestion_heartbeat_interval_seconds: int = Field(
        default=60,
        ge=5,
        le=300,
        alias="RAG_INGESTION_HEARTBEAT_INTERVAL_SECONDS",
    )
    clamav_host: Optional[str] = Field(default=None, alias="RAG_CLAMAV_HOST")
    clamav_port: int = Field(default=3310, ge=1, le=65535, alias="RAG_CLAMAV_PORT")
    prompt_injection_policy: PromptInjectionPolicy = Field(
        default="quarantine",
        alias="RAG_PROMPT_INJECTION_POLICY",
    )
    background_heartbeat_ttl_seconds: int = Field(
        default=90,
        ge=30,
        le=600,
        alias="RAG_BACKGROUND_HEARTBEAT_TTL_SECONDS",
    )

    @field_validator("database_url")
    @classmethod
    def database_url_must_be_supported(cls, value: str) -> str:
        clean = value.strip()
        if not clean.startswith(("postgresql+psycopg://", "sqlite+pysqlite://")):
            raise ValueError(
                "RAG_DATABASE_URL must use postgresql+psycopg:// or sqlite+pysqlite://"
            )
        return clean

    @model_validator(mode="after")
    def validate_environment_security(self):
        if self.environment in {"staging", "production"}:
            if self.auth_mode != "oidc":
                raise ValueError("staging and production require RAG_AUTH_MODE=oidc")
            if not self.database_url.startswith("postgresql+psycopg://"):
                raise ValueError("staging and production require PostgreSQL")
            if not str(self.redis_url or "").startswith(("redis://", "rediss://")):
                raise ValueError("staging and production require RAG_REDIS_URL")
            if not str(self.qdrant_url or "").startswith(("http://", "https://")):
                raise ValueError("staging and production require RAG_QDRANT_URL")
            if not str(self.inference_base_url or "").startswith(("http://", "https://")):
                raise ValueError("staging and production require RAG_OLLAMA_URL")
            if not str(self.clamav_host or "").strip():
                raise ValueError("staging and production require RAG_CLAMAV_HOST")
            missing_object_storage = [
                name
                for name, value in (
                    ("RAG_S3_ENDPOINT_URL", self.s3_endpoint_url),
                    ("RAG_S3_ACCESS_KEY_ID", self.s3_access_key_id),
                    ("RAG_S3_SECRET_ACCESS_KEY", self.s3_secret_access_key),
                )
                if not value
            ]
            if missing_object_storage:
                raise ValueError(
                    "S3 configuration is incomplete: "
                    + ", ".join(missing_object_storage)
                )
            if self.upload_mode == "presigned" and not str(
                self.s3_public_endpoint_url or ""
            ).startswith(("http://", "https://")):
                raise ValueError(
                    "production presigned uploads require RAG_S3_PUBLIC_ENDPOINT_URL"
                )
            missing_web_auth = [
                name
                for name, value in (
                    ("RAG_OIDC_AUTHORIZATION_URL", self.oidc_authorization_url),
                    ("RAG_OIDC_TOKEN_URL", self.oidc_token_url),
                    ("RAG_OIDC_CLIENT_ID", self.oidc_client_id),
                    ("RAG_PUBLIC_BASE_URL", self.public_base_url),
                )
                if not str(value or "").strip()
            ]
            if missing_web_auth:
                raise ValueError(
                    "browser OIDC configuration is incomplete: "
                    + ", ".join(missing_web_auth)
                )

        if self.auth_mode == "dev_hs256":
            if self.environment not in {"development", "test"}:
                raise ValueError("dev_hs256 is allowed only in development or test")
            secret = self.dev_jwt_secret.get_secret_value() if self.dev_jwt_secret else ""
            if len(secret) < 32:
                raise ValueError("RAG_DEV_JWT_SECRET must contain at least 32 characters")
        else:
            missing = [
                name
                for name, value in (
                    ("RAG_OIDC_ISSUER", self.oidc_issuer),
                    ("RAG_OIDC_AUDIENCE", self.oidc_audience),
                    ("RAG_OIDC_JWKS_URL", self.oidc_jwks_url),
                )
                if not str(value or "").strip()
            ]
            if missing:
                raise ValueError("OIDC configuration is incomplete: " + ", ".join(missing))
        for field_name, value in (
            ("RAG_OIDC_AUTHORIZATION_URL", self.oidc_authorization_url),
            ("RAG_OIDC_TOKEN_URL", self.oidc_token_url),
            ("RAG_PUBLIC_BASE_URL", self.public_base_url),
        ):
            if value and not str(value).startswith("https://"):
                if not (
                    self.environment in {"development", "test"}
                    and str(value).startswith("http://127.0.0.1")
                ):
                    raise ValueError(f"{field_name} must use HTTPS")
        if self.public_base_url:
            self.public_base_url = str(self.public_base_url).rstrip("/")
        scopes = " ".join(str(self.oidc_scopes or "").split())
        if "openid" not in scopes.split():
            raise ValueError("RAG_OIDC_SCOPES must include openid")
        self.oidc_scopes = scopes
        cookie_name = str(self.web_session_cookie_name or "").strip()
        if not cookie_name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in cookie_name
        ):
            raise ValueError("RAG_WEB_SESSION_COOKIE_NAME contains invalid characters")
        self.web_session_cookie_name = cookie_name
        languages = str(self.ocr_languages or "").strip()
        if not languages or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-"
            for character in languages
        ):
            raise ValueError("RAG_OCR_LANGUAGES contains invalid characters")
        self.ocr_languages = languages
        if self.ingestion_heartbeat_interval_seconds * 3 >= self.ingestion_lease_seconds:
            raise ValueError(
                "RAG_INGESTION_HEARTBEAT_INTERVAL_SECONDS must be less than one third "
                "of RAG_INGESTION_LEASE_SECONDS"
            )
        return self

    @property
    def allowed_models(self) -> List[str]:
        values = [
            item.strip()
            for item in self.allowed_models_csv.split(",")
            if item.strip()
        ]
        if self.default_model not in values:
            values.insert(0, self.default_model)
        return list(dict.fromkeys(values))

    @property
    def cors_origins(self) -> List[str]:
        return [
            item.strip().rstrip("/")
            for item in self.cors_origins_csv.split(",")
            if item.strip()
        ]


@lru_cache(maxsize=1)
def get_production_settings() -> ProductionSettings:
    return ProductionSettings()
