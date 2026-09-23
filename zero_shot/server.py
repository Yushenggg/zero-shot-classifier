from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .classifier import classify
from .config import load_config
from .scorer import get_scorer, is_loaded, loaded_scorer

STATIC_DIR = Path(__file__).parent / "static"
CONFIG = load_config()
logger = logging.getLogger("uvicorn.error")
_RESOLVED_DEVICE: str | None = None


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
        logger.info("Model ready: %s on %s", type(scorer._model).__name__, scorer.device)
    except Exception as exc:  # noqa: BLE001 - keep serving; retry on first request
        logger.warning("Model preload failed (%s: %s); will load on first request.", type(exc).__name__, exc)
    yield


app = FastAPI(title="Zero-Shot Classifier", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ClassifyRequest(BaseModel):
    state: Any = None
    questions: dict[str, Any] | None = None
    question: dict[str, Any] | None = None  # legacy alias for `questions`
    model: str | None = None
    temperature: float | None = None
    kv_cache: bool | None = None
    calibrate: bool | None = None
    calibration_context: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    scorer = loaded_scorer(CONFIG.model)
    return {
        "ok": True,
        "model_id": CONFIG.model,
        "device": _RESOLVED_DEVICE or CONFIG.device,
        "gpu": CONFIG.gpu,
        "quantize": CONFIG.quantize,
        "kv_cache": CONFIG.kv_cache,
        "temperature": CONFIG.temperature,
        "calibrate": CONFIG.calibrate,
        "model_loaded": scorer is not None,
        # None until the model is loaded, so the UI can hide/disable image mode.
        "multimodal": scorer.multimodal if scorer is not None else None,
    }


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
        return JSONResponse(status_code=400, content={"error": str(exc)})
    elapsed_ms = (time.perf_counter() - started) * 1000

    answers = {r.name: r.to_dict() for r in results}
    usage = {
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "timing_ms": elapsed_ms,
    }
    return JSONResponse(
        content={
            "model": model_id,
            "answers": answers,
            "results": list(answers.values()),
            "usage": usage,
        }
    )


def _text_response(request: ClassifyRequest) -> JSONResponse:
    questions = request.questions if request.questions is not None else request.question
    if not questions:
        return JSONResponse(status_code=422, content={"error": "`questions` is required"})
    return _run_classify(
        questions,
        request.state,
        model=request.model,
        temperature=request.temperature,
        kv_cache=request.kv_cache,
        calibrate=request.calibrate,
        calibration_context=request.calibration_context,
    )


@app.post("/v1/classify/text")
def classify_text_endpoint(request: ClassifyRequest) -> JSONResponse:
    """Text classification from a JSON body."""
    return _text_response(request)


@app.post("/v1/systemone")
def systemone_endpoint(request: ClassifyRequest) -> JSONResponse:
    """TypeSafe-compatible JSON endpoint (text-only)."""
    return _text_response(request)


@app.post("/v1/classify/image")
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
    """Image classification: multipart `file` stream plus JSON `questions`/`state`."""
    try:
        questions_payload = json.loads(questions)
    except json.JSONDecodeError as exc:
        return JSONResponse(
            status_code=422, content={"error": f"`questions` is not valid JSON: {exc}"}
        )
    state_payload: Any = None
    if state:
        try:
            state_payload = json.loads(state)
        except json.JSONDecodeError as exc:
            return JSONResponse(
                status_code=422, content={"error": f"`state` is not valid JSON: {exc}"}
            )

    image = await file.read()
    if not image:
        return JSONResponse(status_code=422, content={"error": "`file` is empty"})

    return _run_classify(
        questions_payload,
        state_payload,
        image=image,
        model=model,
        temperature=temperature,
        kv_cache=kv_cache,
        calibrate=calibrate,
        calibration_context=calibration_context,
    )


def _cpu_low_config_path() -> Path:
    """Locate the bundled config.smollm.toml (project root or cwd)."""
    candidate = Path(__file__).resolve().parent.parent / "config.smollm.toml"
    return candidate if candidate.exists() else Path.cwd() / "config.smollm.toml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="zero-shot-serve",
        description="Serve the zero-shot classifier web UI and HTTP API.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file to serve, e.g. config.vlm.toml for image classification.",
    )
    parser.add_argument(
        "--cpu-low",
        action="store_true",
        help="Serve the small CPU model (config.smollm.toml) instead of the configured one.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: $ZERO_SHOT_PORT or 8000).",
    )
    args = parser.parse_args(argv)

    global CONFIG
    if args.config:
        CONFIG = load_config(args.config)
    elif args.cpu_low:
        CONFIG = load_config(_cpu_low_config_path())

    host = os.environ.get("ZERO_SHOT_HOST", "127.0.0.1")
    port = args.port if args.port is not None else int(os.environ.get("ZERO_SHOT_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
