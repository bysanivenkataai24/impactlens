"""FastAPI application: ``GET /health``, ``POST /predict`` and the static review UI.

This module knows HTTP. It does not import any model library: the model is
reached only through :class:`impactlens.adapters.SegmentationAdapter`.

Order of checks in ``POST /predict`` (cheapest and least revealing first):

1. rate limit            -> 429 ``rate_limited``
2. authentication        -> 401 ``unauthenticated``
3. model readiness       -> 503 ``model_unavailable``
4. request media type    -> 415 ``unsupported_media_type``
5. declared size         -> 413 ``payload_too_large``
6. capacity (admission)  -> 429 ``server_busy``
7. body read (bounded)   -> 413 / 408 / 422
8. image decode          -> 415 / 413 / 422
9. inference             -> 504 ``inference_timeout`` / 500 ``inference_failed``

No branch substitutes canned detections for an error.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from .. import __version__
from ..config import Settings
from ..imaging import ImageRejected, configure_pillow
from ..schemas import (
    LIMITATIONS,
    SCHEMA_VERSION,
    AuthCheckResponse,
    ErrorResponse,
    HealthResponse,
    ImageInfo,
    LatencyMs,
    PredictResponse,
)
from .runtime import (
    InferenceFailed,
    InferenceService,
    InferenceTimeout,
    ModelLoader,
    ModelState,
    build_model_state,
    default_model_loader,
)
from .security import (
    SecurityHeadersMiddleware,
    SlidingWindowRateLimiter,
    api_key_matches,
    client_address,
    presented_api_key,
)

log = logging.getLogger("impactlens.service")

#: Allowance for multipart boundaries, part headers and the threshold field.
MULTIPART_OVERHEAD_BYTES = 16 * 1024
_ALLOWED_FIELDS = {"image", "score_threshold"}

NO_REGION_MESSAGE = (
    "No supported damage region was returned for this photo at the requested score "
    "threshold. This does not mean the vehicle is undamaged."
)


class ApiError(Exception):
    """An error that maps to one documented HTTP status and error code."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        reasons: list[str] | None = None,
        retry_after_s: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.reasons = reasons
        self.retry_after_s = retry_after_s
        self.headers = dict(headers or {})
        if retry_after_s is not None:
            self.headers.setdefault("Retry-After", str(retry_after_s))


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_response(request: Request, error: ApiError) -> JSONResponse:
    body = ErrorResponse(
        error={
            "code": error.code,
            "message": error.message,
            "request_id": _request_id(request),
            "reasons": error.reasons,
            "retry_after_s": error.retry_after_s,
        }
    )
    return JSONResponse(
        body.model_dump(mode="json", exclude_none=True),
        status_code=error.status,
        headers=error.headers,
    )


def _parse_threshold(raw: object, settings: Settings) -> float:
    if raw is None:
        return settings.default_score_threshold
    low, high = settings.min_score_threshold, settings.max_score_threshold
    problem = ApiError(
        422,
        "invalid_request",
        f"score_threshold must be a number between {low} and {high}.",
    )
    if not isinstance(raw, str) or len(raw) > 32:
        raise problem
    try:
        value = float(raw.strip())
    except ValueError:
        raise problem from None
    if not math.isfinite(value) or not low <= value <= high:
        raise problem
    return value


async def _read_upload(request: Request, settings: Settings) -> tuple[bytes, float]:
    """Read the multipart body into memory with a hard byte limit."""
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise ApiError(
            415,
            "unsupported_media_type",
            "Send the photo as multipart/form-data with a file field named 'image'.",
        )

    limit = settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
    too_large = ApiError(
        413,
        "payload_too_large",
        f"The upload exceeds the limit of {settings.max_upload_bytes} bytes.",
    )
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                raise too_large
        except ValueError:
            raise ApiError(422, "invalid_request", "Invalid Content-Length header.") from None

    async def bounded_stream():
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            # Enforced on bytes actually received, so chunked uploads and wrong
            # Content-Length headers cannot bypass it.
            if received > limit:
                raise too_large
            yield chunk

    parser = MultiPartParser(
        request.headers, bounded_stream(), max_files=1, max_fields=2, max_part_size=1024
    )
    # Keep the upload in memory: never spool visitor bytes to a temporary file.
    parser.spool_max_size = limit + 1
    form = None
    try:
        form = await asyncio.wait_for(parser.parse(), timeout=settings.request_body_timeout_s)
        unknown = set(form.keys()) - _ALLOWED_FIELDS
        if unknown:
            raise ApiError(
                422, "invalid_request", f"Unexpected form field(s): {sorted(unknown)[:5]}."
            )
        upload = form.get("image")
        if not isinstance(upload, UploadFile):
            raise ApiError(
                422, "invalid_request", "A file field named 'image' is required."
            )
        if len(form.getlist("image")) != 1:
            raise ApiError(422, "invalid_request", "Send exactly one 'image' file per request.")
        threshold = _parse_threshold(form.get("score_threshold"), settings)
        data = await upload.read()
    except TimeoutError:
        raise ApiError(
            408, "request_timeout", "The upload was not received in time."
        ) from None
    except MultiPartException as exc:
        raise ApiError(422, "invalid_request", f"Malformed multipart body: {exc.message}") from None
    finally:
        if form is not None:
            await form.close()

    if len(data) > settings.max_upload_bytes:
        raise too_large
    return data, threshold


def create_app(
    settings: Settings | None = None, *, model_loader: ModelLoader | None = None
) -> FastAPI:
    """Application factory.

    ``model_loader`` exists so that contract tests can inject a synthetic
    adapter. Production code never passes it.
    """
    settings = settings or Settings.from_env()
    loader = model_loader or default_model_loader
    logging.getLogger("impactlens").setLevel(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_pillow(settings.max_image_pixels)
        # Loading is blocking work (file hashing, checkpoint deserialization, warm-up).
        state: ModelState = await anyio.to_thread.run_sync(build_model_state, settings, loader)
        app.state.model = state
        app.state.inference = (
            InferenceService(state.adapter, settings) if state.adapter is not None else None
        )
        log.info("startup complete: ready=%s reasons=%s", state.ready, state.reasons)
        try:
            yield
        finally:
            if app.state.inference is not None:
                await app.state.inference.aclose()
            if state.adapter is not None:
                state.adapter.close()

    app = FastAPI(
        title="ImpactLens inference service",
        version=__version__,
        description=(
            "Decision-support vehicle damage segmentation. Every result is a suggestion "
            "pending human review. Response schema version " + SCHEMA_VERSION + "."
        ),
        lifespan=lifespan,
        docs_url=None,  # the default docs pages load scripts from a CDN, which the CSP forbids
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute, 60.0)

    # --- middleware (last added = outermost) -----------------------------
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "X-API-Key", "Content-Type"],
            allow_credentials=False,
            expose_headers=["X-Request-ID", "Retry-After"],
            max_age=600,
        )
    app.add_middleware(SecurityHeadersMiddleware, frame_ancestors=settings.frame_ancestors)

    # --- error handlers ----------------------------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _error_response(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "invalid_request")
        message = {404: "Not found.", 405: "Method not allowed."}.get(
            exc.status_code, "The request could not be processed."
        )
        return _error_response(
            request, ApiError(exc.status_code, code, message, headers=dict(exc.headers or {}))
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return _error_response(request, ApiError(422, "invalid_request", "Invalid request."))

    # --- guards ------------------------------------------------------------
    def guard(request: Request) -> None:
        """Rate limit, then authenticate. Used by every protected route."""
        retry_after = app.state.limiter.hit(client_address(request, settings.trusted_proxy_hops))
        if retry_after is not None:
            raise ApiError(
                429,
                "rate_limited",
                "Too many requests from this client. Retry later.",
                retry_after_s=retry_after,
            )
        if not api_key_matches(presented_api_key(request), settings.api_key):
            raise ApiError(
                401,
                "unauthenticated",
                "A valid service key is required (Authorization: Bearer <key>).",
                headers={"WWW-Authenticate": 'Bearer realm="impactlens"'},
            )

    # --- routes ------------------------------------------------------------
    def health_payload() -> HealthResponse:
        state: ModelState = app.state.model
        info = state.adapter.info if state.adapter is not None else None
        return HealthResponse(
            ready=state.ready,
            reasons=list(dict.fromkeys(state.reasons)),
            model_id=state.model_id,
            model_revision=state.model_revision,
            checkpoint_sha256=state.checkpoint_sha256,
            task=info.task if info else None,
            classes=list(info.class_names) if info else [],
            device=info.device if info else None,
            limits=settings.public_summary(),
            service_version=__version__,
            source_url=settings.source_url,
        )

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    async def health() -> JSONResponse:
        """Liveness plus readiness detail. Always 200 while the process is serving."""
        return JSONResponse(health_payload().model_dump(mode="json"))

    @app.get(
        "/ready",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        tags=["service"],
    )
    async def ready() -> JSONResponse:
        """Same body as /health, but 503 when real inference is not possible."""
        payload = health_payload()
        return JSONResponse(payload.model_dump(mode="json"), status_code=200 if payload.ready else 503)

    @app.get(
        "/auth/check",
        response_model=AuthCheckResponse,
        responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
        tags=["service"],
    )
    async def auth_check(request: Request) -> JSONResponse:
        """Lets a client verify its service key without uploading a photo."""
        guard(request)
        return JSONResponse(AuthCheckResponse().model_dump(mode="json"))

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            code: {"model": ErrorResponse} for code in (401, 408, 413, 415, 422, 429, 500, 503, 504)
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["image"],
                            "properties": {
                                "image": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "One JPEG, PNG or WebP photograph.",
                                },
                                "score_threshold": {
                                    "type": "number",
                                    "description": "Minimum model score. See /health limits.",
                                },
                            },
                        }
                    }
                },
            }
        },
        tags=["inference"],
    )
    async def predict(request: Request) -> JSONResponse:
        started = time.perf_counter()
        guard(request)

        state: ModelState = app.state.model
        service: InferenceService | None = app.state.inference
        if not state.ready or service is None:
            raise ApiError(
                503,
                "model_unavailable",
                "No usable model is loaded, so no prediction was made. See /health.",
                reasons=list(dict.fromkeys(state.reasons)),
            )

        if not service.try_admit():
            raise ApiError(
                429,
                "server_busy",
                "The service is at capacity. Retry shortly.",
                retry_after_s=max(1, math.ceil(settings.inference_timeout_s / 4)),
            )
        try:
            data, threshold = await _read_upload(request, settings)
            output = await service.run(data, threshold)
        except ImageRejected as exc:
            raise ApiError(exc.status, exc.code, exc.message) from None
        except InferenceTimeout:
            log.warning("request %s timed out", _request_id(request))
            raise ApiError(
                504,
                "inference_timeout",
                "The request ran out of time. No result was produced.",
                retry_after_s=max(1, math.ceil(settings.inference_timeout_s / 4)),
            ) from None
        except InferenceFailed as exc:
            log.error(
                "request %s: inference failed", _request_id(request), exc_info=exc.__cause__ or exc
            )
            raise ApiError(
                500, "inference_failed", "Inference failed. No result was produced."
            ) from None
        except ApiError:
            raise
        except ClientDisconnect:
            # Nobody is listening any more; answer in a defined way and keep the log quiet.
            raise ApiError(
                400, "invalid_request", "The client disconnected during the upload."
            ) from None
        except Exception:
            log.exception("request %s: unexpected error", _request_id(request))
            raise ApiError(500, "internal_error", "Unexpected server error.") from None
        finally:
            service.leave()

        image = output.image
        total_ms = (time.perf_counter() - started) * 1000.0
        response = PredictResponse(
            request_id=_request_id(request),
            model_id=state.model_id,
            model_revision=state.model_revision,
            checkpoint_sha256=state.checkpoint_sha256,
            image=ImageInfo(
                width=image.width,
                height=image.height,
                format=image.format,
                exif_orientation=image.exif_orientation,
                orientation_applied=image.orientation_applied,
                sha256=image.sha256,
            ),
            score_threshold=threshold,
            outcome="detections_returned" if output.detections else "no_supported_region_returned",
            message=(
                f"{len(output.detections)} region(s) returned for human review."
                if output.detections
                else NO_REGION_MESSAGE
            ),
            detections=output.detections,
            warnings=output.warnings,
            latency_ms=LatencyMs(
                queue=round(output.queue_ms, 2),
                decode=round(output.decode_ms, 2),
                inference=round(output.inference_ms, 2),
                postprocess=round(output.postprocess_ms, 2),
                total=round(total_ms, 2),
            ),
            limitations=list(LIMITATIONS),
        )
        # Operational log line: sizes and timings only. No image content, key or address.
        log.info(
            "predict request_id=%s bytes=%d image=%dx%d detections=%d threshold=%.3f "
            "queue_ms=%.1f decode_ms=%.1f inference_ms=%.1f total_ms=%.1f",
            response.request_id,
            len(data),
            image.width,
            image.height,
            len(output.detections),
            threshold,
            output.queue_ms,
            output.decode_ms,
            output.inference_ms,
            total_ms,
        )
        return JSONResponse(response.model_dump(mode="json"))

    # --- static review UI (registered last so API routes win) ---------------
    dist = settings.frontend_dist or Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
        log.info("serving review UI from %s", dist)
    else:
        log.warning("review UI not found at %s; only the API is served", dist)

        @app.get("/", include_in_schema=False)
        async def no_ui() -> Response:
            return JSONResponse(
                {
                    "service": "impactlens",
                    "detail": "The review UI is not built. Run `npm run build` in frontend/ "
                    "or set FRONTEND_DIST. API: GET /health, POST /predict.",
                }
            )

    return app
