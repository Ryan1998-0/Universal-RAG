import asyncio
import hashlib
import json
import logging
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import List, Optional
from urllib.parse import quote
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from rag_demo.production.auth import (
    AuthenticationError,
    Principal,
    TokenVerifier,
    build_token_verifier,
)
from rag_demo.production.background_health import RedisBackgroundHealth
from rag_demo.production.config import ProductionSettings, get_production_settings
from rag_demo.production.database import create_database_engine, create_session_factory
from rag_demo.production.deletion import SqlAlchemyDeletionRepository
from rag_demo.production.embedding_runtime import FastEmbedRuntime
from rag_demo.production.file_security import ClamAvScanner
from rag_demo.production.generation_gate import (
    GenerationCapacityTimeout,
    GenerationGate,
    build_generation_gate,
)
from rag_demo.production.inference_probe import OllamaInferenceProbe
from rag_demo.production.indexing import (
    IndexBuildValidationError,
    SqlAlchemyIndexingRepository,
)
from rag_demo.production.object_storage import S3ObjectStorage
from rag_demo.production.repository import (
    AccessDeniedError,
    InvalidServiceStateError,
    ResourceNotFoundError,
    SqlAlchemyTenantRepository,
)
from rag_demo.production.rate_limit import RateLimiter, build_rate_limiter
from rag_demo.production.metrics import ProductionMetrics, route_label
from rag_demo.production.retriever import ProductionHybridRetriever
from rag_demo.production.vector_repository import QdrantChunkRepository
from rag_demo.production.task_dispatch import CeleryTaskDispatcher
from rag_demo.production.upload_service import UploadService, UploadValidationError
from rag_demo.production.web_auth import (
    OidcWebAuth,
    RedisWebSessionStore,
    WebAuthenticationError,
    WebSessionUnavailable,
)
from rag_demo.model_gateway import (
    ModelGateway,
    normalize_model_node,
    validate_model_spec,
)
from rag_demo.model_providers import model_request_timeout
from rag_demo.observability import configure_logging, log_event
from rag_demo.rag_pipeline import AGENT_RESPONSE_SCHEMA, RagPipeline, RagPipelineRequest
from rag_demo.retrieval_scope import RetrievalScope


LOGGER = logging.getLogger("rag_demo.production.api")
PIPELINE_VERSION = "canonical-v1"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=20_000)
    knowledge_base_id: str = Field(min_length=1, max_length=64)
    source_ids: Optional[List[str]] = Field(default=None, max_length=500)
    model: Optional[str] = Field(default=None, min_length=1, max_length=255)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    conversation_id: Optional[str] = Field(default=None, min_length=1, max_length=64)

    @field_validator("question", "knowledge_base_id", "model", "conversation_id")
    @classmethod
    def strip_text(cls, value):
        if value is None:
            return value
        clean = str(value).strip()
        if not clean:
            raise ValueError("must not be blank")
        return clean

    @field_validator("source_ids")
    @classmethod
    def normalize_source_ids(cls, value):
        if value is None:
            return None
        normalized = []
        for source_id in value:
            clean = str(source_id).strip()
            if not clean or len(clean) > 255:
                raise ValueError("source IDs must contain 1 to 255 characters")
            if clean not in normalized:
                normalized.append(clean)
        return normalized


class ModelValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=255)
    node: str = Field(default="generation", min_length=1, max_length=64)

    @field_validator("model", "node")
    @classmethod
    def strip_model_fields(cls, value):
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("must not be blank")
        return clean


class ModelBindingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=255)

    @field_validator("model")
    @classmethod
    def strip_model(cls, value):
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("must not be blank")
        return clean


class CreateUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    folder_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    target_document_id: Optional[str] = Field(default=None, min_length=1, max_length=64)

    @field_validator("filename", "content_type", "folder_id", "target_document_id")
    @classmethod
    def strip_upload_text(cls, value):
        if value is None:
            return value
        clean = str(value).strip()
        if not clean:
            raise ValueError("must not be blank")
        return clean


class CreateIndexBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: List[str] = Field(min_length=1, max_length=500)

    @field_validator("document_ids")
    @classmethod
    def normalize_document_ids(cls, value):
        normalized = []
        for document_id in value:
            clean = str(document_id or "").strip()
            if not clean or len(clean) > 64:
                raise ValueError("document IDs must contain 1 to 64 characters")
            if clean not in normalized:
                normalized.append(clean)
        if not normalized:
            raise ValueError("at least one document ID is required")
        return normalized


class CreateKnowledgeBaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    profile: str = Field(default="default", min_length=1, max_length=64)
    visibility: str = Field(default="private", pattern=r"^(private|tenant)$")

    @field_validator("name", "profile", "visibility")
    @classmethod
    def strip_knowledge_base_text(cls, value):
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("must not be blank")
        return clean


class CreateFolderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)

    @field_validator("name")
    @classmethod
    def strip_folder_name(cls, value):
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("must not be blank")
        return clean


class MoveDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_id: Optional[str] = Field(default=None, min_length=1, max_length=64)

    @field_validator("folder_id")
    @classmethod
    def strip_folder_id(cls, value):
        if value is None:
            return None
        clean = str(value).strip()
        if not clean:
            raise ValueError("must not be blank")
        return clean


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_base_id: str = Field(min_length=1, max_length=64)

    @field_validator("knowledge_base_id")
    @classmethod
    def strip_conversation_kb(cls, value):
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("must not be blank")
        return clean


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: Optional[dict] = None,
    ):
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = dict(headers or {})


def _run_pipeline_with_deadline(pipeline, request, timeout_seconds: float):
    with model_request_timeout(timeout_seconds):
        return pipeline.run(request)


def create_app(
    settings: Optional[ProductionSettings] = None,
    repository=None,
    pipeline=None,
    token_verifier: Optional[TokenVerifier] = None,
    rate_limiter: Optional[RateLimiter] = None,
    vector_repository=None,
    object_storage=None,
    inference_probe=None,
    embedding_runtime=None,
    upload_service=None,
    task_dispatcher=None,
    indexing_repository=None,
    deletion_repository=None,
    generation_gate: Optional[GenerationGate] = None,
    malware_scanner=None,
    background_health_probe=None,
    web_auth: Optional[OidcWebAuth] = None,
) -> FastAPI:
    resolved_settings = settings or get_production_settings()
    log_path = configure_logging()
    model_gateway = ModelGateway(
        default_model=resolved_settings.default_model,
        allowed_models=resolved_settings.allowed_models,
    )
    owned_engine = None
    if repository is None:
        owned_engine = create_database_engine(resolved_settings.database_url)
        repository = SqlAlchemyTenantRepository(create_session_factory(owned_engine))
    production_environment = resolved_settings.environment in {"staging", "production"}
    owned_vector_repository = False
    if vector_repository is None and production_environment:
        vector_repository = QdrantChunkRepository.from_settings(resolved_settings)
        owned_vector_repository = True
    if object_storage is None and production_environment:
        object_storage = S3ObjectStorage.from_settings(resolved_settings)
    if (
        inference_probe is None
        and production_environment
        and any(str(model).startswith("ollama:") for model in resolved_settings.allowed_models)
    ):
        inference_probe = OllamaInferenceProbe(
            resolved_settings.inference_base_url,
            resolved_settings.allowed_models,
        )
    if malware_scanner is None and production_environment:
        malware_scanner = ClamAvScanner(
            host=resolved_settings.clamav_host or "",
            port=resolved_settings.clamav_port,
            timeout_seconds=3,
        )
    if background_health_probe is None and production_environment:
        background_health_probe = RedisBackgroundHealth.from_settings(
            resolved_settings
        )
    if task_dispatcher is None and production_environment:
        task_dispatcher = CeleryTaskDispatcher.from_settings(resolved_settings)
    if upload_service is None and object_storage is not None:
        upload_service = UploadService(
            repository=repository,
            object_storage=object_storage,
            settings=resolved_settings,
            dispatcher=task_dispatcher,
        )
    elif upload_service is not None:
        if object_storage is None:
            object_storage = getattr(upload_service, "object_storage", None)
        if task_dispatcher is None:
            task_dispatcher = getattr(upload_service, "dispatcher", None)
    if indexing_repository is None and hasattr(repository, "session_factory"):
        indexing_repository = SqlAlchemyIndexingRepository(repository.session_factory)
    if deletion_repository is None and hasattr(repository, "session_factory"):
        deletion_repository = SqlAlchemyDeletionRepository(repository.session_factory)
    if pipeline is None:
        if production_environment:
            embedding_runtime = embedding_runtime or FastEmbedRuntime.from_settings(
                resolved_settings
            )
            production_retriever = ProductionHybridRetriever(
                vector_repository=vector_repository,
                embedding_runtime=embedding_runtime,
            )
            pipeline = RagPipeline(
                retriever_factory=lambda _profile: production_retriever,
                model_gateway=model_gateway,
            )
        else:
            pipeline = RagPipeline(model_gateway=model_gateway)
    elif getattr(pipeline, "_model_gateway", None) is None:
        # Custom pipelines keep their injected callbacks while still exposing
        # the same runtime model registry to the API contract.
        try:
            setattr(pipeline, "_model_gateway", model_gateway)
        except (AttributeError, TypeError):
            # Some tests/integrations pass opaque callable objects.  They can
            # still use the API model endpoints; only pipeline injection is
            # unavailable for immutable objects.
            pass
    token_verifier = token_verifier or build_token_verifier(resolved_settings)
    rate_limiter = rate_limiter or build_rate_limiter(resolved_settings)
    metrics = ProductionMetrics()
    generation_gate = generation_gate or build_generation_gate(resolved_settings)
    owned_web_session_store = None
    if web_auth is None and production_environment:
        owned_web_session_store = RedisWebSessionStore.from_settings(resolved_settings)
        web_auth = OidcWebAuth(
            settings=resolved_settings,
            token_verifier=token_verifier,
            store=owned_web_session_store,
        )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        if owned_vector_repository:
            close = getattr(vector_repository.client, "close", None)
            if close is not None:
                close()
        if owned_engine is not None:
            owned_engine.dispose()
        if owned_web_session_store is not None:
            close = getattr(owned_web_session_store.client, "close", None)
            if close is not None:
                close()

    docs_url = "/docs" if resolved_settings.enable_docs else None
    openapi_url = "/openapi.json" if resolved_settings.enable_docs else None
    app = FastAPI(
        title="泛用 RAG API",
        version=resolved_settings.service_version,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.repository = repository
    app.state.pipeline = pipeline
    app.state.metrics = metrics
    app.state.vector_repository = vector_repository
    app.state.object_storage = object_storage
    app.state.inference_probe = inference_probe
    app.state.upload_service = upload_service
    app.state.indexing_repository = indexing_repository
    app.state.deletion_repository = deletion_repository
    app.state.generation_gate = generation_gate
    app.state.malware_scanner = malware_scanner
    app.state.background_health_probe = background_health_probe
    app.state.web_auth = web_auth
    app.state.model_gateway = model_gateway
    app.state.log_path = str(log_path)

    if resolved_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "Idempotency-Key",
                "X-Request-ID",
            ],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        started_at = perf_counter()
        metrics.http_inflight.inc()
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_request_id
            if _REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else uuid4().hex
        )
        request.state.request_id = request_id
        content_length = request.headers.get("Content-Length", "")
        if content_length.isdigit() and int(content_length) > resolved_settings.max_request_body_bytes:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": "The request body is too large.",
                    },
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
            _apply_security_headers(response)
            metrics.api_errors.labels(code="REQUEST_TOO_LARGE").inc()
            metrics.http_requests.labels(
                method=request.method,
                route="unmatched",
                status="413",
            ).inc()
            metrics.http_duration.labels(
                method=request.method,
                route="unmatched",
            ).observe((perf_counter() - started_at))
            metrics.http_inflight.dec()
            return response
        try:
            response = await call_next(request)
        except Exception:
            _log_event(
                "request.failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=_elapsed_ms(started_at),
            )
            metrics.http_requests.labels(
                method=request.method,
                route=route_label(request.scope),
                status="500",
            ).inc()
            metrics.http_duration.labels(
                method=request.method,
                route=route_label(request.scope),
            ).observe(perf_counter() - started_at)
            metrics.http_inflight.dec()
            raise
        response.headers["X-Request-ID"] = request_id
        _apply_security_headers(response)
        _log_event(
            "request.completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=_elapsed_ms(started_at),
        )
        route = route_label(request.scope)
        metrics.http_requests.labels(
            method=request.method,
            route=route,
            status=str(response.status_code),
        ).inc()
        metrics.http_duration.labels(
            method=request.method,
            route=route,
        ).observe(perf_counter() - started_at)
        metrics.http_inflight.dec()
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        metrics.api_errors.labels(code=exc.code).inc()
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {"code": exc.code, "message": exc.message},
                "request_id": request.state.request_id,
            },
            headers={
                **({"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else {}),
                **exc.headers,
            },
        )

    security = HTTPBearer(auto_error=False, scheme_name="oidcBearer")

    async def current_principal(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    ) -> Principal:
        if credentials is not None and credentials.scheme.lower() == "bearer":
            try:
                return token_verifier.verify(credentials.credentials)
            except AuthenticationError:
                raise ApiError(401, "INVALID_TOKEN", "The access token is invalid or expired.")
        session_id = request.cookies.get(resolved_settings.web_session_cookie_name, "")
        if web_auth is not None and session_id:
            try:
                principal = await run_in_threadpool(web_auth.principal_for_session, session_id)
            except WebSessionUnavailable:
                raise ApiError(
                    503,
                    "SESSION_UNAVAILABLE",
                    "The login session service is temporarily unavailable.",
                )
            if principal is not None:
                return principal
        raise ApiError(401, "AUTH_REQUIRED", "A valid login session is required.")

    @app.get("/auth/status", include_in_schema=False)
    async def web_auth_status(request: Request):
        session_id = request.cookies.get(resolved_settings.web_session_cookie_name, "")
        if web_auth is None or not session_id:
            return _no_store_json({"authenticated": False})
        try:
            principal = await run_in_threadpool(web_auth.principal_for_session, session_id)
        except WebSessionUnavailable:
            raise ApiError(
                503,
                "SESSION_UNAVAILABLE",
                "The login session service is temporarily unavailable.",
            )
        if principal is None:
            return _no_store_json({"authenticated": False})
        return _no_store_json({
            "authenticated": True,
            "subject": principal.subject,
            "tenant_id": principal.tenant_id,
        })

    @app.get("/auth/login", include_in_schema=False)
    async def web_auth_login():
        if web_auth is None:
            raise ApiError(404, "WEB_LOGIN_DISABLED", "Browser login is not configured.")
        try:
            location = await run_in_threadpool(web_auth.begin_login)
        except WebSessionUnavailable:
            raise ApiError(
                503,
                "SESSION_UNAVAILABLE",
                "The login session service is temporarily unavailable.",
            )
        return RedirectResponse(location, status_code=303, headers={"Cache-Control": "no-store"})

    @app.get("/auth/callback", include_in_schema=False)
    async def web_auth_callback(
        state: str = "",
        code: str = "",
        error: str = "",
    ):
        if web_auth is None:
            raise ApiError(404, "WEB_LOGIN_DISABLED", "Browser login is not configured.")
        if error:
            raise ApiError(400, "OIDC_LOGIN_FAILED", "Identity provider login failed.")
        try:
            completed = await run_in_threadpool(
                web_auth.complete_login,
                state=state,
                code=code,
            )
        except WebAuthenticationError:
            raise ApiError(400, "OIDC_LOGIN_FAILED", "Identity provider login failed.")
        except WebSessionUnavailable:
            raise ApiError(
                503,
                "SESSION_UNAVAILABLE",
                "The login session service is temporarily unavailable.",
            )
        response = RedirectResponse("/", status_code=303, headers={"Cache-Control": "no-store"})
        response.set_cookie(
            resolved_settings.web_session_cookie_name,
            completed.session_id,
            max_age=completed.max_age_seconds,
            path="/",
            secure=production_environment,
            httponly=True,
            samesite="lax",
        )
        return response

    @app.post("/auth/logout", include_in_schema=False)
    async def web_auth_logout(request: Request):
        configured_origin = str(resolved_settings.public_base_url or "").rstrip("/")
        request_origin = str(request.headers.get("Origin") or "").rstrip("/")
        if configured_origin and request_origin and request_origin != configured_origin:
            raise ApiError(403, "ORIGIN_DENIED", "The request origin is not allowed.")
        session_id = request.cookies.get(resolved_settings.web_session_cookie_name, "")
        if web_auth is not None and session_id:
            try:
                await run_in_threadpool(web_auth.logout, session_id)
            except WebSessionUnavailable:
                raise ApiError(
                    503,
                    "SESSION_UNAVAILABLE",
                    "The login session service is temporarily unavailable.",
                )
        response = Response(status_code=204, headers={"Cache-Control": "no-store"})
        response.delete_cookie(
            resolved_settings.web_session_cookie_name,
            path="/",
            secure=production_environment,
            httponly=True,
            samesite="lax",
        )
        return response

    @app.get("/health/live", include_in_schema=False)
    async def health_live():
        return {
            "status": "live",
            "service": resolved_settings.service_name,
            "version": resolved_settings.service_version,
        }

    @app.get("/health/ready", include_in_schema=False)
    async def health_ready(request: Request):
        dependencies = {}
        try:
            await run_in_threadpool(repository.ping)
            dependencies["database"] = "ready"
        except Exception:
            _log_event(
                "health.not_ready",
                request_id=request.state.request_id,
                dependency="database",
            )
            dependencies["database"] = "unavailable"
        limiter_ping = getattr(rate_limiter, "ping", None)
        if limiter_ping is not None:
            try:
                await run_in_threadpool(limiter_ping)
                dependencies["redis"] = "ready"
            except Exception:
                dependencies["redis"] = "unavailable"
        for dependency_name, dependency in (
            ("qdrant", vector_repository),
            ("object_storage", object_storage),
            ("inference", inference_probe),
            ("malware_scanner", malware_scanner),
            ("background_workers", background_health_probe),
        ):
            if dependency is None:
                continue
            try:
                await run_in_threadpool(dependency.ping)
                dependencies[dependency_name] = "ready"
            except Exception:
                _log_event(
                    "health.not_ready",
                    request_id=request.state.request_id,
                    dependency=dependency_name,
                )
                dependencies[dependency_name] = "unavailable"
        if "unavailable" in dependencies.values():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "dependencies": dependencies},
            )
        return {"status": "ready", "dependencies": dependencies}

    if resolved_settings.enable_metrics:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        @app.get("/metrics", include_in_schema=False)
        async def prometheus_metrics():
            return Response(
                content=generate_latest(metrics.registry),
                media_type=CONTENT_TYPE_LATEST,
            )

    @app.get("/v1/knowledge-bases")
    async def list_knowledge_bases(
        principal: Principal = Depends(current_principal),
    ):
        try:
            return {
                "items": await run_in_threadpool(
                    repository.list_knowledge_bases,
                    principal,
                )
            }
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")

    @app.get("/v1/runtime")
    async def runtime_config(
        principal: Principal = Depends(current_principal),
    ):
        del principal
        return {
            "default_model": resolved_settings.default_model,
            "allowed_models": resolved_settings.allowed_models,
            "max_upload_bytes": resolved_settings.max_upload_bytes,
            "upload_mode": resolved_settings.upload_mode,
            "supported_extensions": [
                "pdf",
                "png",
                "jpg",
                "jpeg",
                "webp",
                "tif",
                "tiff",
                "bmp",
                "heic",
                "docx",
                "txt",
                "md",
                "markdown",
                "json",
            ],
            "interfaces": {
                "data_input": {
                    "question": "POST /v1/ask",
                    "document_upload": "POST /v1/knowledge-bases/{knowledge_base_id}/uploads",
                    "document_content": "PUT /v1/uploads/{upload_id}/content",
                    "document_complete": "POST /v1/uploads/{upload_id}/complete",
                },
                "data_output": {
                    "answer": "POST /v1/ask",
                    "run": "GET /v1/answer-runs/{run_id}",
                    "timings": "POST /v1/ask -> timings.stages",
                },
                "model": {
                    "list": "GET /v1/models",
                    "validate": "POST /v1/models/validate",
                    "override": "PUT /v1/models/{node}",
                    "clear_override": "DELETE /v1/models/{node}",
                },
            },
        }

    @app.get("/v1/models")
    async def list_model_bindings(
        principal: Principal = Depends(current_principal),
    ):
        del principal
        description = model_gateway.describe()
        description["components"] = {
            "embedding": {
                "model": resolved_settings.embedding_model,
                "env_var": "RAG_EMBEDDING_MODEL",
                "replaceable": True,
                "runtime_override": False,
            },
            "sparse_embedding": {
                "model": resolved_settings.sparse_embedding_model,
                "env_var": "RAG_SPARSE_EMBEDDING_MODEL",
                "replaceable": True,
                "runtime_override": False,
            },
            "reranker": {
                "model": resolved_settings.reranker_model,
                "env_var": "RAG_RERANKER_MODEL",
                "replaceable": True,
                "runtime_override": False,
            },
        }
        description["observability"] = {
            "log_file": Path(str(log_path)).name,
            "timing_field": "timings.stages",
        }
        return description

    @app.post("/v1/models/validate")
    async def validate_model_binding(
        payload: ModelValidationRequest,
        principal: Principal = Depends(current_principal),
    ):
        del principal
        try:
            node = normalize_model_node(payload.node)
            validation = validate_model_spec(payload.model, resolved_settings.allowed_models)
        except ValueError as exc:
            raise ApiError(400, "INVALID_MODEL", str(exc))
        validation["node"] = node
        validation["binding"] = model_gateway.binding(
            node,
            requested_model=payload.model,
        ).as_dict()
        return validation

    def require_model_admin(principal: Principal) -> None:
        if production_environment:
            raise ApiError(
                403,
                "MODEL_MANAGEMENT_FORBIDDEN",
                "Runtime model replacement is disabled in staging and production.",
            )
        if not ({"owner", "admin"} & set(principal.roles)):
            raise ApiError(403, "MODEL_MANAGEMENT_FORBIDDEN", "Model replacement requires owner or admin role.")

    @app.put("/v1/models/{node}")
    async def override_model_binding(
        node: str,
        payload: ModelBindingRequest,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        require_model_admin(principal)
        try:
            binding = model_gateway.set_override(node, payload.model)
        except ValueError as exc:
            raise ApiError(400, "INVALID_MODEL", str(exc))
        await _record_audit_safely(
            repository,
            principal,
            "model.override",
            "model_node",
            binding.node,
            request.state.request_id,
            "success",
        )
        return {
            "binding": binding.as_dict(),
            "runtime_override": True,
            "scope": "process",
            "persistent": False,
            "restart_required_for_env_change": True,
        }

    @app.delete("/v1/models/{node}")
    async def clear_model_binding(
        node: str,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        require_model_admin(principal)
        try:
            binding = model_gateway.clear_override(node)
        except ValueError as exc:
            raise ApiError(400, "INVALID_MODEL", str(exc))
        await _record_audit_safely(
            repository,
            principal,
            "model.override.clear",
            "model_node",
            binding.node,
            request.state.request_id,
            "success",
        )
        return {"binding": binding.as_dict(), "runtime_override": False}

    @app.post("/v1/knowledge-bases", status_code=201)
    async def create_knowledge_base(
        payload: CreateKnowledgeBaseRequest,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        try:
            result = await run_in_threadpool(
                repository.create_knowledge_base,
                principal=principal,
                name=payload.name,
                profile=payload.profile,
                visibility=payload.visibility,
            )
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        except ValueError:
            raise ApiError(400, "INVALID_KNOWLEDGE_BASE", "The knowledge base is invalid.")
        except InvalidServiceStateError:
            raise ApiError(409, "KNOWLEDGE_BASE_CONFLICT", "The knowledge base already exists.")
        await _record_audit_safely(
            repository,
            principal,
            "knowledge_base.create",
            "knowledge_base",
            result["id"],
            request.state.request_id,
            "success",
        )
        return result

    @app.put("/v1/knowledge-bases/{knowledge_base_id}/folders/{folder_id}")
    async def rename_folder(
        knowledge_base_id: str,
        folder_id: str,
        payload: CreateFolderRequest,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                knowledge_base_id,
                None,
            )
            result = await run_in_threadpool(
                repository.rename_folder,
                principal=principal,
                authorized=authorized,
                folder_id=folder_id,
                name=payload.name,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "FOLDER_NOT_FOUND", "Folder not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        except ValueError:
            raise ApiError(400, "INVALID_FOLDER", "The folder is invalid.")
        except InvalidServiceStateError:
            raise ApiError(409, "FOLDER_CONFLICT", "The folder already exists.")
        await _record_audit_safely(
            repository,
            principal,
            "folder.rename",
            "folder",
            folder_id,
            request.state.request_id,
            "success",
        )
        return result

    @app.delete("/v1/knowledge-bases/{knowledge_base_id}/folders/{folder_id}")
    async def delete_folder(
        knowledge_base_id: str,
        folder_id: str,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                knowledge_base_id,
                None,
            )
            result = await run_in_threadpool(
                repository.delete_folder,
                principal=principal,
                authorized=authorized,
                folder_id=folder_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "FOLDER_NOT_FOUND", "Folder not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        await _record_audit_safely(
            repository,
            principal,
            "folder.delete",
            "folder",
            folder_id,
            request.state.request_id,
            "success",
        )
        return result

    @app.get("/v1/knowledge-bases/{knowledge_base_id}/folders")
    async def list_folders(
        knowledge_base_id: str,
        principal: Principal = Depends(current_principal),
    ):
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                knowledge_base_id,
                None,
            )
            items = await run_in_threadpool(
                repository.list_folders,
                principal=principal,
                authorized=authorized,
            )
            return {"items": items}
        except ResourceNotFoundError:
            raise ApiError(404, "KNOWLEDGE_BASE_NOT_FOUND", "Knowledge base not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")

    @app.post("/v1/knowledge-bases/{knowledge_base_id}/folders", status_code=201)
    async def create_folder(
        knowledge_base_id: str,
        payload: CreateFolderRequest,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                knowledge_base_id,
                None,
            )
            result = await run_in_threadpool(
                repository.create_folder,
                principal=principal,
                authorized=authorized,
                name=payload.name,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "KNOWLEDGE_BASE_NOT_FOUND", "Knowledge base not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        except ValueError:
            raise ApiError(400, "INVALID_FOLDER", "The folder is invalid.")
        except InvalidServiceStateError:
            raise ApiError(409, "FOLDER_CONFLICT", "The folder already exists.")
        await _record_audit_safely(
            repository,
            principal,
            "folder.create",
            "folder",
            result["id"],
            request.state.request_id,
            "success",
        )
        return result

    @app.get("/v1/knowledge-bases/{knowledge_base_id}/documents")
    async def list_documents(
        knowledge_base_id: str,
        principal: Principal = Depends(current_principal),
    ):
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                knowledge_base_id,
                None,
            )
            items = await run_in_threadpool(
                repository.list_documents,
                principal=principal,
                authorized=authorized,
            )
            return {"items": items}
        except ResourceNotFoundError:
            raise ApiError(404, "KNOWLEDGE_BASE_NOT_FOUND", "Knowledge base not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")

    @app.put(
        "/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}/folder"
    )
    async def move_document(
        knowledge_base_id: str,
        document_id: str,
        payload: MoveDocumentRequest,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                knowledge_base_id,
                None,
            )
            result = await run_in_threadpool(
                repository.move_document,
                principal=principal,
                authorized=authorized,
                document_id=document_id,
                folder_id=payload.folder_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "RESOURCE_NOT_FOUND", "Document or folder not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        await _record_audit_safely(
            repository,
            principal,
            "document.move",
            "document",
            document_id,
            request.state.request_id,
            "success",
        )
        return result

    @app.get("/v1/conversations")
    async def list_conversations(
        principal: Principal = Depends(current_principal),
    ):
        try:
            return {
                "items": await run_in_threadpool(
                    repository.list_conversations,
                    principal=principal,
                )
            }
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")

    @app.post("/v1/conversations", status_code=201)
    async def create_conversation(
        payload: CreateConversationRequest,
        principal: Principal = Depends(current_principal),
    ):
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                payload.knowledge_base_id,
                None,
            )
            return await run_in_threadpool(
                repository.ensure_conversation,
                principal=principal,
                authorized=authorized,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "KNOWLEDGE_BASE_NOT_FOUND", "Knowledge base not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")

    @app.get("/v1/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: str,
        principal: Principal = Depends(current_principal),
    ):
        try:
            return await run_in_threadpool(
                repository.get_conversation,
                principal=principal,
                conversation_id=conversation_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "CONVERSATION_NOT_FOUND", "Conversation not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")

    @app.get("/v1/answer-runs/{run_id}")
    async def get_answer_run(
        run_id: str,
        principal: Principal = Depends(current_principal),
    ):
        try:
            result = await run_in_threadpool(
                repository.get_answer_run,
                principal=principal,
                run_id=run_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "ANSWER_RUN_NOT_FOUND", "Answer run not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        for citation in result.get("citations") or []:
            version_id = (
                str(citation.get("document_version_id") or "")
                if citation.get("available") is True
                else ""
            )
            citation["source_url"] = (
                f"/v1/document-versions/{version_id}/content"
                if version_id
                else None
            )
        return result

    @app.get("/v1/document-versions/{document_version_id}/content")
    async def download_document_version(
        document_version_id: str,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        if object_storage is None:
            raise ApiError(503, "SOURCE_UNAVAILABLE", "Source content is unavailable.")
        try:
            source = await run_in_threadpool(
                repository.resolve_document_version_download,
                principal=principal,
                document_version_id=document_version_id,
            )
            await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                source["knowledge_base_id"],
                None,
            )
            payload = await run_in_threadpool(
                object_storage.download_bytes,
                source["object_key"],
                resolved_settings.max_upload_bytes,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "SOURCE_NOT_FOUND", "Source content not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        except Exception as exc:
            _log_event(
                "source.download.failed",
                request_id=request.state.request_id,
                tenant_id=principal.tenant_id,
                document_version_id=document_version_id,
                error_class=exc.__class__.__name__,
            )
            raise ApiError(503, "SOURCE_UNAVAILABLE", "Source content is unavailable.")
        filename = Path(str(source["filename"] or "document")).name
        ascii_filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "document"
        await _record_audit_safely(
            repository,
            principal,
            "document.download",
            "document_version",
            document_version_id,
            request.state.request_id,
            "success",
        )
        return Response(
            content=payload,
            media_type=str(source["mime_type"] or "application/octet-stream"),
            headers={
                "Content-Disposition": (
                    f"inline; filename=\"{ascii_filename}\"; "
                    f"filename*=UTF-8''{quote(filename)}"
                ),
                "Cache-Control": "private, no-store",
            },
        )
    @app.delete("/v1/conversations/{conversation_id}")
    async def delete_conversation(
        conversation_id: str,
        principal: Principal = Depends(current_principal),
    ):
        try:
            await run_in_threadpool(
                repository.delete_conversation,
                principal=principal,
                conversation_id=conversation_id,
            )
            return {"deleted": True, "id": conversation_id}
        except ResourceNotFoundError:
            raise ApiError(404, "CONVERSATION_NOT_FOUND", "Conversation not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")

    @app.post("/v1/ask")
    async def ask(
        payload: AskRequest,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        model = payload.model or resolved_settings.default_model
        if model not in resolved_settings.allowed_models:
            raise ApiError(400, "MODEL_NOT_ALLOWED", "The requested model is not allowed.")

        try:
            rate_decision = await run_in_threadpool(rate_limiter.allow, principal, "rag.ask")
        except Exception:
            _log_event(
                "rate_limit.failed",
                request_id=request.state.request_id,
                tenant_id=principal.tenant_id,
            )
            raise ApiError(503, "DEPENDENCY_UNAVAILABLE", "The service is temporarily unavailable.")
        if not rate_decision.allowed:
            raise ApiError(
                429,
                "RATE_LIMITED",
                "Too many requests. Please retry later.",
                headers={"Retry-After": str(rate_decision.retry_after_seconds)},
            )

        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                payload.knowledge_base_id,
                payload.source_ids,
            )
        except ResourceNotFoundError:
            await _record_audit_safely(
                repository,
                principal,
                "rag.ask",
                "knowledge_base",
                payload.knowledge_base_id,
                request.state.request_id,
                "not_found",
            )
            raise ApiError(404, "KNOWLEDGE_BASE_NOT_FOUND", "Knowledge base not found.")
        except AccessDeniedError:
            await _record_audit_safely(
                repository,
                principal,
                "rag.ask",
                "knowledge_base",
                payload.knowledge_base_id,
                request.state.request_id,
                "denied",
            )
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        except Exception:
            _log_event(
                "authorization.failed",
                request_id=request.state.request_id,
                tenant_id=principal.tenant_id,
            )
            raise ApiError(503, "DEPENDENCY_UNAVAILABLE", "The service is temporarily unavailable.")

        if authorized.active_index_version_id and any((
            authorized.embedding_model != resolved_settings.embedding_model,
            authorized.sparse_embedding_model
            != resolved_settings.sparse_embedding_model,
            authorized.embedding_dimensions != resolved_settings.embedding_dimensions,
            authorized.chunk_schema_version != resolved_settings.chunk_schema_version,
            authorized.qdrant_collection != resolved_settings.qdrant_collection,
        )):
            _log_event(
                "retrieval.index_configuration_mismatch",
                request_id=request.state.request_id,
                tenant_id=principal.tenant_id,
                knowledge_base_id=authorized.id,
                index_version_id=authorized.active_index_version_id,
            )
            raise ApiError(
                503,
                "INDEX_CONFIGURATION_MISMATCH",
                "The active index is incompatible with the configured retrieval runtime.",
            )

        conversation_id = str(payload.conversation_id or "")
        history = []
        if conversation_id:
            try:
                conversation = await run_in_threadpool(
                    repository.ensure_conversation,
                    principal=principal,
                    authorized=authorized,
                    conversation_id=conversation_id,
                )
                conversation_id = conversation["id"]
                history = await run_in_threadpool(
                    repository.recent_conversation_messages,
                    principal=principal,
                    authorized=authorized,
                    conversation_id=conversation_id,
                    limit=10,
                )
            except ResourceNotFoundError:
                raise ApiError(404, "CONVERSATION_NOT_FOUND", "Conversation not found.")
            except AccessDeniedError:
                raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")

        pipeline_request = RagPipelineRequest(
            question=payload.question,
            model=model,
            profile=authorized.profile,
            source_ids=authorized.source_ids,
            top_k=payload.top_k,
            persist_conversation=False,
            conversation_id=conversation_id,
            history=history,
            request_id=request.state.request_id,
            retrieval_scope=RetrievalScope(
                tenant_id=principal.tenant_id,
                knowledge_base_id=authorized.id,
                index_version_id=authorized.active_index_version_id or "",
            ),
        )
        try:
            metrics.generation_waiting.inc()
            try:
                generation_token = await generation_gate.acquire(
                    resolved_settings.generation_queue_timeout_seconds
                )
            finally:
                metrics.generation_waiting.dec()
        except GenerationCapacityTimeout:
            raise ApiError(
                503,
                "GENERATION_CAPACITY_FULL",
                "Generation capacity is currently full. Please retry shortly.",
                headers={"Retry-After": "1"},
            )

        release_generation_token = True
        try:
            metrics.generation_active.inc()
            try:
                result = await asyncio.wait_for(
                    run_in_threadpool(
                        _run_pipeline_with_deadline,
                        pipeline,
                        pipeline_request,
                        resolved_settings.request_timeout_seconds - 1,
                    ),
                    timeout=resolved_settings.request_timeout_seconds,
                )
            finally:
                metrics.generation_active.dec()
        except asyncio.TimeoutError:
            release_generation_token = False
            await _record_audit_safely(
                repository,
                principal,
                "rag.ask",
                "knowledge_base",
                authorized.id,
                request.state.request_id,
                "timeout",
            )
            raise ApiError(504, "MODEL_TIMEOUT", "The answer request timed out.")
        except Exception:
            _log_event(
                "pipeline.failed",
                request_id=request.state.request_id,
                tenant_id=principal.tenant_id,
                knowledge_base_id=authorized.id,
            )
            await _record_audit_safely(
                repository,
                principal,
                "rag.ask",
                "knowledge_base",
                authorized.id,
                request.state.request_id,
                "failed",
            )
            raise ApiError(500, "ANSWER_FAILED", "The answer could not be generated.")
        finally:
            if release_generation_token:
                try:
                    await generation_gate.release(generation_token)
                except Exception:
                    _log_event(
                        "generation_lease.release_failed",
                        request_id=request.state.request_id,
                        tenant_id=principal.tenant_id,
                    )

        metrics.observe_pipeline_timings(result.get("timings") or {})

        if not conversation_id:
            try:
                conversation = await run_in_threadpool(
                    repository.ensure_conversation,
                    principal=principal,
                    authorized=authorized,
                )
                conversation_id = conversation["id"]
            except Exception:
                _log_event(
                    "conversation.create_failed",
                    request_id=request.state.request_id,
                    tenant_id=principal.tenant_id,
                )
                raise ApiError(
                    503,
                    "RESULT_NOT_PERSISTED",
                    "The result could not be persisted safely.",
                )

        try:
            await run_in_threadpool(
                repository.record_answer_run,
                principal,
                authorized,
                payload.question,
                model,
                result,
                PIPELINE_VERSION,
                request.state.request_id,
                conversation_id,
            )
            await run_in_threadpool(
                repository.record_audit_event,
                principal,
                "rag.ask",
                "knowledge_base",
                authorized.id,
                request.state.request_id,
                "success",
                {"run_id": result.get("run_id")},
            )
        except Exception:
            _log_event(
                "persistence.failed",
                request_id=request.state.request_id,
                tenant_id=principal.tenant_id,
                run_id=result.get("run_id"),
            )
            raise ApiError(503, "RESULT_NOT_PERSISTED", "The result could not be persisted safely.")

        return _public_result(
            result,
            knowledge_base_id=authorized.id,
            conversation_id=conversation_id,
        )

    @app.post("/v1/knowledge-bases/{knowledge_base_id}/uploads", status_code=201)
    async def reserve_upload(
        knowledge_base_id: str,
        payload: CreateUploadRequest,
        request: Request,
        principal: Principal = Depends(current_principal),
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ):
        if upload_service is None:
            raise ApiError(503, "UPLOAD_UNAVAILABLE", "Document upload is temporarily unavailable.")
        if not idempotency_key:
            raise ApiError(400, "IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required.")
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                knowledge_base_id,
                None,
            )
            result = await run_in_threadpool(
                upload_service.reserve,
                principal=principal,
                authorized=authorized,
                idempotency_key=idempotency_key,
                filename=payload.filename,
                content_type=payload.content_type,
                size_bytes=payload.size_bytes,
                sha256=payload.sha256,
                folder_id=payload.folder_id,
                target_document_id=payload.target_document_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "RESOURCE_NOT_FOUND", "The requested resource was not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        except UploadValidationError:
            raise ApiError(400, "INVALID_UPLOAD", "The upload request is invalid.")
        except InvalidServiceStateError:
            raise ApiError(409, "UPLOAD_CONFLICT", "The upload request conflicts with existing state.")
        except Exception:
            _log_event(
                "upload.reserve.failed",
                request_id=request.state.request_id,
                knowledge_base_id=knowledge_base_id,
            )
            raise ApiError(
                503,
                "UPLOAD_DEPENDENCY_UNAVAILABLE",
                "Document upload is temporarily unavailable.",
            )
        await _record_audit_safely(
            repository,
            principal,
            "document.upload.reserve",
            "upload_session",
            result["upload_id"],
            request.state.request_id,
            "success",
        )
        return result

    @app.put("/v1/uploads/{upload_id}/content", status_code=201)
    async def upload_content(
        upload_id: str,
        request: Request,
        principal: Principal = Depends(current_principal),
        content_type: Optional[str] = Header(default=None, alias="Content-Type"),
    ):
        if upload_service is None:
            raise ApiError(503, "UPLOAD_UNAVAILABLE", "Document upload is temporarily unavailable.")
        try:
            reservation = await run_in_threadpool(
                upload_service.get_reservation,
                principal=principal,
                upload_id=upload_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "UPLOAD_NOT_FOUND", "Upload session not found.")
        except InvalidServiceStateError:
            raise ApiError(409, "UPLOAD_NOT_WRITABLE", "The upload cannot receive content.")

        digest = hashlib.sha256()
        observed_size = 0
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as body:
            async for block in request.stream():
                if not block:
                    continue
                observed_size += len(block)
                if (
                    observed_size > reservation.expected_size_bytes
                    or observed_size > resolved_settings.max_upload_bytes
                ):
                    raise ApiError(
                        413,
                        "UPLOAD_TOO_LARGE",
                        "The uploaded content exceeds the reserved size.",
                    )
                digest.update(block)
                body.write(block)
            body.seek(0)
            try:
                result = await run_in_threadpool(
                    upload_service.store_proxy_content,
                    principal=principal,
                    reservation=reservation,
                    body=body,
                    size_bytes=observed_size,
                    sha256=digest.hexdigest(),
                    content_type=content_type or "",
                )
            except UploadValidationError:
                raise ApiError(400, "UPLOAD_CONTENT_MISMATCH", "The uploaded content is invalid.")
            except ResourceNotFoundError:
                raise ApiError(404, "UPLOAD_NOT_FOUND", "Upload session not found.")
            except InvalidServiceStateError:
                raise ApiError(409, "UPLOAD_NOT_WRITABLE", "The upload cannot receive content.")
            except Exception:
                _log_event(
                    "upload.content.failed",
                    request_id=request.state.request_id,
                    upload_id=upload_id,
                )
                raise ApiError(
                    503,
                    "UPLOAD_DEPENDENCY_UNAVAILABLE",
                    "Document upload is temporarily unavailable.",
                )
        await _record_audit_safely(
            repository,
            principal,
            "document.upload.content",
            "upload_session",
            upload_id,
            request.state.request_id,
            "success",
        )
        return result

    @app.post("/v1/uploads/{upload_id}/complete", status_code=202)
    async def complete_upload(
        upload_id: str,
        request: Request,
        principal: Principal = Depends(current_principal),
    ):
        if upload_service is None:
            raise ApiError(503, "UPLOAD_UNAVAILABLE", "Document upload is temporarily unavailable.")
        try:
            result = await run_in_threadpool(
                upload_service.complete,
                principal=principal,
                upload_id=upload_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "UPLOAD_NOT_FOUND", "Upload session not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        except InvalidServiceStateError:
            raise ApiError(409, "UPLOAD_NOT_COMPLETABLE", "The upload cannot be completed.")
        except Exception:
            _log_event(
                "upload.complete.failed",
                request_id=request.state.request_id,
                upload_id=upload_id,
            )
            raise ApiError(503, "UPLOAD_DEPENDENCY_UNAVAILABLE", "Upload verification is unavailable.")
        await _record_audit_safely(
            repository,
            principal,
            "document.upload.complete",
            "document_version",
            result["document_version_id"],
            request.state.request_id,
            "success",
        )
        return result

    @app.get("/v1/ingestion-jobs/{job_id}")
    async def ingestion_job_status(
        job_id: str,
        principal: Principal = Depends(current_principal),
    ):
        try:
            return await run_in_threadpool(
                repository.get_ingestion_job,
                principal=principal,
                job_id=job_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "INGESTION_JOB_NOT_FOUND", "Ingestion job not found.")

    @app.post(
        "/v1/knowledge-bases/{knowledge_base_id}/index-builds",
        status_code=202,
    )
    async def create_index_build(
        knowledge_base_id: str,
        payload: CreateIndexBuildRequest,
        request: Request,
        principal: Principal = Depends(current_principal),
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ):
        if indexing_repository is None:
            raise ApiError(503, "INDEX_BUILD_UNAVAILABLE", "Index building is unavailable.")
        if not idempotency_key:
            raise ApiError(400, "IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required.")
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                knowledge_base_id,
                None,
            )
            reservation = await run_in_threadpool(
                indexing_repository.reserve_build,
                principal=principal,
                authorized=authorized,
                idempotency_key=idempotency_key,
                document_ids=payload.document_ids,
                settings=resolved_settings,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "RESOURCE_NOT_FOUND", "The requested resource was not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        except IndexBuildValidationError:
            raise ApiError(400, "INVALID_INDEX_BUILD", "The index build request is invalid.")
        except InvalidServiceStateError:
            raise ApiError(409, "INDEX_BUILD_CONFLICT", "The index build cannot be created.")

        dispatch_state = "already_queued" if reservation.duplicate else "pending"
        if task_dispatcher is not None and not reservation.duplicate:
            try:
                task_id = await run_in_threadpool(
                    task_dispatcher.dispatch_index_build,
                    reservation.job_id,
                )
                await run_in_threadpool(
                    indexing_repository.record_dispatch,
                    job_id=reservation.job_id,
                    task_id=task_id,
                )
                dispatch_state = "queued"
            except Exception:
                _log_event(
                    "index_build.dispatch_failed",
                    request_id=request.state.request_id,
                    job_id=reservation.job_id,
                )
        await _record_audit_safely(
            repository,
            principal,
            "index.build.create",
            "index_version",
            reservation.index_version_id,
            request.state.request_id,
            "success",
        )
        return {
            "job_id": reservation.job_id,
            "index_version_id": reservation.index_version_id,
            "status": reservation.status,
            "duplicate": reservation.duplicate,
            "dispatch_state": dispatch_state,
        }

    @app.get("/v1/index-build-jobs/{job_id}")
    async def index_build_status(
        job_id: str,
        principal: Principal = Depends(current_principal),
    ):
        if indexing_repository is None:
            raise ApiError(503, "INDEX_BUILD_UNAVAILABLE", "Index building is unavailable.")
        try:
            return await run_in_threadpool(
                indexing_repository.get_status,
                principal=principal,
                job_id=job_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "INDEX_BUILD_JOB_NOT_FOUND", "Index build job not found.")

    @app.delete(
        "/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}",
        status_code=202,
    )
    async def delete_document(
        knowledge_base_id: str,
        document_id: str,
        request: Request,
        principal: Principal = Depends(current_principal),
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ):
        if deletion_repository is None:
            raise ApiError(503, "DELETION_UNAVAILABLE", "Document deletion is unavailable.")
        if not idempotency_key:
            raise ApiError(400, "IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required.")
        try:
            authorized = await run_in_threadpool(
                repository.authorize_knowledge_base,
                principal,
                knowledge_base_id,
                None,
            )
            reservation = await run_in_threadpool(
                deletion_repository.tombstone_document,
                principal=principal,
                authorized=authorized,
                document_id=document_id,
                idempotency_key=idempotency_key,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "DOCUMENT_NOT_FOUND", "Document not found.")
        except AccessDeniedError:
            raise ApiError(403, "ACCESS_DENIED", "Access to this resource is denied.")
        except ValueError:
            raise ApiError(400, "INVALID_DELETION", "The deletion request is invalid.")
        except InvalidServiceStateError:
            raise ApiError(409, "DELETION_CONFLICT", "The deletion request conflicts with existing state.")
        await _record_audit_safely(
            repository,
            principal,
            "document.delete",
            "document",
            document_id,
            request.state.request_id,
            "accepted",
        )
        return {
            "outbox_id": reservation.outbox_id,
            "document_id": reservation.document_id,
            "state": reservation.state,
            "duplicate": reservation.duplicate,
        }

    @app.get("/v1/deletion-jobs/{outbox_id}")
    async def deletion_status(
        outbox_id: str,
        principal: Principal = Depends(current_principal),
    ):
        if deletion_repository is None:
            raise ApiError(503, "DELETION_UNAVAILABLE", "Document deletion is unavailable.")
        try:
            return await run_in_threadpool(
                deletion_repository.get_status,
                principal=principal,
                outbox_id=outbox_id,
            )
        except ResourceNotFoundError:
            raise ApiError(404, "DELETION_JOB_NOT_FOUND", "Deletion job not found.")

    return app


async def _record_audit_safely(
    repository,
    principal: Principal,
    action: str,
    resource_type: str,
    resource_id: str,
    request_id: str,
    outcome: str,
) -> None:
    try:
        await run_in_threadpool(
            repository.record_audit_event,
            principal,
            action,
            resource_type,
            resource_id,
            request_id,
            outcome,
        )
    except Exception:
        _log_event(
            "audit.write_failed",
            request_id=request_id,
            tenant_id=principal.tenant_id,
            outcome=outcome,
        )


def _apply_security_headers(response) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"


def _no_store_json(content: dict, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=content,
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )


def _public_result(
    result: dict,
    knowledge_base_id: str,
    conversation_id: str = "",
) -> dict:
    retrieval = dict(result.get("retrieval") or {})
    evidence = retrieval.get("evidence_evaluation")
    public_evidence = None
    if isinstance(evidence, dict):
        public_evidence = {
            "sufficient": bool(evidence.get("sufficient")),
            "confidence": str(evidence.get("confidence") or ""),
            "reason": str(evidence.get("reason") or ""),
        }
    validation = result.get("evidence_validation")
    public_validation = None
    if isinstance(validation, dict):
        public_validation = {
            "sufficient": bool(validation.get("sufficient")),
            "status": str(validation.get("status") or ""),
            "valid_citations": [
                int(rank)
                for rank in validation.get("valid_citations") or []
                if str(rank).isdigit()
            ],
            "invalid_citations": [
                int(rank)
                for rank in validation.get("invalid_citations") or []
                if str(rank).isdigit()
            ],
            "uncited_claims": [
                str(claim)[:500]
                for claim in validation.get("uncited_claims") or []
                if str(claim).strip()
            ][:20],
            "unsupported_claims": [
                str(claim)[:500]
                for claim in validation.get("unsupported_claims") or []
                if str(claim).strip()
            ][:20],
            "reason": str(validation.get("reason") or ""),
        }
    return {
        "schema_version": str(result.get("schema_version") or AGENT_RESPONSE_SCHEMA),
        "run_id": str(result.get("run_id") or ""),
        "request_id": str(
            result.get("request_id")
            or (result.get("timing_trace") or {}).get("requestId")
            or ""
        ),
        "knowledge_base_id": knowledge_base_id,
        "conversation_id": conversation_id,
        "answer": str(result.get("answer") or ""),
        "confidence": str(result.get("confidence") or "medium"),
        "citations": _public_citations(result),
        "grounding_warnings": list(result.get("grounding_warnings") or []),
        "evidence_validation": public_validation,
        "retrieval": {
            "server_generated": True,
            "needed": bool(retrieval.get("needed")),
            "reason": str(retrieval.get("reason") or ""),
            "query": str(retrieval.get("query") or ""),
            "evidence_evaluation": public_evidence,
            "timings": dict(retrieval.get("timings") or {}),
        },
        "model": dict(result.get("model") or {}),
        "timings": dict(result.get("timings") or {}),
        "timing_trace": dict(result.get("timing_trace") or {}),
    }


def _public_citations(result: dict) -> list[dict]:
    run_id = str(result.get("run_id") or "")
    citations = []
    for raw in result.get("citations") or []:
        citation = dict(raw)
        version_id = str(
            citation.get("document_version_id")
            or citation.get("documentVersionId")
            or ""
        )
        citation["details_url"] = f"/v1/answer-runs/{run_id}" if run_id else None
        citation["source_url"] = (
            f"/v1/document-versions/{version_id}/content"
            if version_id
            else None
        )
        citations.append(citation)
    return citations


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _log_event(event: str, **fields) -> None:
    LOGGER.info(json.dumps({"event": event, **fields}, ensure_ascii=True, sort_keys=True))
