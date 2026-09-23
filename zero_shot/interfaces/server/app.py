from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ...core.classifier import classify
from ...core.config import Config, load_config
from ...core.image_utils import decode_image, downscale_to_byte_limit
from ...core.scorer import get_scorer, loaded_scorer
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    HealthResponse,
    Usage,
)

STATIC_DIR = Path(__file__).parent / "static"
CONFIG = load_config()
logger = logging.getLogger("uvicorn.error")
_RESOLVED_DEVICE: str | None = None

# Target size for image uploads: anything larger is downscaled + re-encoded as
# JPEG until it fits (`downscale_to_byte_limit`). MAX_UPLOAD_BYTES is the hard
# ceiling rejected outright, so one request can't make us read/decode unbounded
# data. The decode itself is also guarded by Pillow's decompression-bomb limit.
# Override with ZERO_SHOT_MAX_IMAGE_MB / ZERO_SHOT_MAX_UPLOAD_MB.
MAX_IMAGE_BYTES = int(float(os.environ.get("ZERO_SHOT_MAX_IMAGE_MB", "16")) * 1024 * 1024)
MAX_UPLOAD_BYTES = int(float(os.environ.get("ZERO_SHOT_MAX_UPLOAD_MB", "64")) * 1024 * 1024)
# Whole-request ceiling: the image plus the JSON `questions`/`state` fields and
# multipart overhead. Rejected before the body is read.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024

# Error shapes documented in OpenAPI (the endpoints return them as JSONResponse).
ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Invalid request or classification error"},
    413: {"model": ErrorResponse, "description": "Upload or request body too large"},
    422: {"model": ErrorResponse, "description": "Malformed request body"},
}


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content=ErrorResponse(error=message).model_dump()
    )


def _too_large_response() -> JSONResponse:
    return _error(
        413,
        f"`file` exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit "
        "(set ZERO_SHOT_MAX_UPLOAD_MB to raise it)",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _RESOLVED_DEVICE
    logger.info(
        "Loading model %s on device=%s (gpu=%s, kv_cache=%s, quantize=%s) ...",
        CONFIG.model,
        CONFIG.device,
        CONFIG.gpu,
        CONFIG.kv_cache,
        CONFIG.quantize,
    )
    try:
        scorer = get_scorer(
            CONFIG.model,
            save_to=CONFIG.resolve_save_to(),
            device=CONFIG.device,
            gpu=CONFIG.gpu,
            quantize=CONFIG.quantize,
        )
        _RESOLVED_DEVICE = scorer.device
        logger.info(
            "Model ready: %s on %s (multimodal=%s)",
            type(scorer._model).__name__,
            scorer.device,
            scorer.multimodal,
        )
    except Exception as exc:  # noqa: BLE001 - keep serving; retry on first request
        logger.warning("Model preload failed (%s: %s); will load on first request.", type(exc).__name__, exc)
    yield


app = FastAPI(title="Zero-Shot Classifier", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def limit_request_body(request, call_next):
    """Reject oversized bodies from Content-Length before the parser reads them."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            too_large = int(content_length) > MAX_REQUEST_BYTES
        except ValueError:
            too_large = False
        if too_large:
            return _error(
                413,
                f"Request body exceeds the {MAX_REQUEST_BYTES // (1024 * 1024)} MB limit "
                "(set ZERO_SHOT_MAX_UPLOAD_MB to raise it)",
            )
    return await call_next(request)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> HealthResponse:
    scorer = loaded_scorer(CONFIG.model)
    return HealthResponse(
        ok=True,
        model_id=CONFIG.model,
        device=_RESOLVED_DEVICE or CONFIG.device,
        gpu=CONFIG.gpu,
        quantize=CONFIG.quantize,
        kv_cache=CONFIG.kv_cache,
        temperature=CONFIG.temperature,
        calibrate=CONFIG.calibrate,
        model_loaded=scorer is not None,
        multimodal=scorer.multimodal if scorer is not None else None,
    )


def _run_classify(
    questions: dict[str, Any],
    state: Any,
    *,
    image: Any = None,
    model: str | None = None,
    temperature: float | None = None,
    kv_cache: bool | None = None,
    calibrate: bool | None = None,
    calibration_context: str | None = None,
) -> JSONResponse:
    model_id = model or CONFIG.model
    started = time.perf_counter()
    try:
        results = classify(
            questions,
            state,
            model_id=model_id,
            save_to=CONFIG.save_to_for(model_id),
            device=CONFIG.device,
            gpu=CONFIG.gpu,
            quantize=CONFIG.quantize,
            temperature=temperature if temperature is not None else CONFIG.temperature,
            use_kv_cache=kv_cache if kv_cache is not None else CONFIG.kv_cache,
            calibrate=calibrate if calibrate is not None else CONFIG.calibrate,
            calibration_context=calibration_context or CONFIG.calibration_context,
            image=image,
        )
    except (ValueError, RuntimeError) as exc:
        return _error(400, str(exc))
    elapsed_ms = (time.perf_counter() - started) * 1000

    answers = {r.name: r.to_dict() for r in results}
    payload = ClassifyResponse(
        model=model_id,
        answers=answers,
        results=list(answers.values()),
        usage=Usage(
            input_tokens=sum(r.input_tokens for r in results),
            output_tokens=sum(r.output_tokens for r in results),
            timing_ms=elapsed_ms,
        ),
    )
    return JSONResponse(content=payload.model_dump())


def _text_response(request: ClassifyRequest) -> JSONResponse:
    questions = request.questions if request.questions is not None else request.question
    if not questions:
        return _error(422, "`questions` is required")
    return _run_classify(
        questions,
        request.state,
        model=request.model,
        temperature=request.temperature,
        kv_cache=request.kv_cache,
        calibrate=request.calibrate,
        calibration_context=request.calibration_context,
    )


@app.post("/v1/classify/text", response_model=ClassifyResponse, responses=ERROR_RESPONSES)
def classify_text_endpoint(request: ClassifyRequest) -> JSONResponse:
    """Text classification from a JSON body."""
    return _text_response(request)


@app.post("/v1/systemone", response_model=ClassifyResponse, responses=ERROR_RESPONSES)
def systemone_endpoint(request: ClassifyRequest) -> JSONResponse:
    """TypeSafe-compatible JSON endpoint (text-only)."""
    return _text_response(request)


@app.post("/v1/classify/image", response_model=ClassifyResponse, responses=ERROR_RESPONSES)
@app.post("/v1/systemone/image", response_model=ClassifyResponse, responses=ERROR_RESPONSES)
async def classify_image_endpoint(
    file: UploadFile = File(...),
    questions: str = Form(...),
    state: str | None = Form(None),
    model: str | None = Form(None),
    temperature: float | None = Form(None),
    kv_cache: bool | None = Form(None),
    calibrate: bool | None = Form(None),
    calibration_context: str | None = Form(None),
) -> JSONResponse:
    """Image classification: multipart `file` stream plus JSON `questions`/`state`.

    Mounted at `/v1/classify/image` and the TypeSafe-compatible
    `/v1/systemone/image`.
    """
    try:
        questions_payload = json.loads(questions)
    except json.JSONDecodeError as exc:
        return _error(422, f"`questions` is not valid JSON: {exc}")
    state_payload: Any = None
    if state:
        try:
            state_payload = json.loads(state)
        except json.JSONDecodeError as exc:
            return _error(422, f"`state` is not valid JSON: {exc}")

    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        return _too_large_response()
    image = await file.read()
    if not image:
        return _error(422, "`file` is empty")
    if len(image) > MAX_UPLOAD_BYTES:
        return _too_large_response()
    if len(image) > MAX_IMAGE_BYTES:
        try:
            image = await run_in_threadpool(
                downscale_to_byte_limit, image, MAX_IMAGE_BYTES
            )
        except ValueError as exc:
            return _error(400, str(exc))
    # Decode at the edge so corrupt uploads fail as 400 here, not 500 deeper in
    # the scorer. The decoded image is passed straight through.
    try:
        image = await run_in_threadpool(decode_image, image)
    except ValueError as exc:
        return _error(400, str(exc))

    return await run_in_threadpool(
        _run_classify,
        questions_payload,
        state_payload,
        image=image,
        model=model,
        temperature=temperature,
        kv_cache=kv_cache,
        calibrate=calibrate,
        calibration_context=calibration_context,
    )


def configure(config: Config) -> Config:
    """Swap the active config at runtime (used by ``zero-shot-serve --config``).

    Endpoints read the module-level :data:`CONFIG` on each request, so the new
    config takes effect without rebuilding the app. The model is loaded lazily on
    the first request (or by the lifespan hook on the next start).
    """
    global CONFIG, _RESOLVED_DEVICE
    CONFIG = config
    _RESOLVED_DEVICE = None
    return CONFIG
