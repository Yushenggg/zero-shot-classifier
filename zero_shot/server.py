from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .classifier import classify
from .config import load_config
from .scorer import get_scorer, is_loaded

STATIC_DIR = Path(__file__).parent / "static"
CONFIG = load_config()
logger = logging.getLogger("uvicorn.error")
_RESOLVED_DEVICE: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _RESOLVED_DEVICE
    logger.info(
        "Loading model %s on device=%s (gpu=%s, kv_cache=%s) ...",
        CONFIG.model,
        CONFIG.device,
        CONFIG.gpu,
        CONFIG.kv_cache,
    )
    try:
        scorer = get_scorer(
            CONFIG.model,
            save_to=CONFIG.resolve_save_to(),
            device=CONFIG.device,
            gpu=CONFIG.gpu,
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
    return {
        "ok": True,
        "model_id": CONFIG.model,
        "device": _RESOLVED_DEVICE or CONFIG.device,
        "gpu": CONFIG.gpu,
        "kv_cache": CONFIG.kv_cache,
        "temperature": CONFIG.temperature,
        "calibrate": CONFIG.calibrate,
        "model_loaded": is_loaded(CONFIG.model),
    }


@app.post("/api/classify")
@app.post("/v1/systemone")
def classify_endpoint(request: ClassifyRequest) -> JSONResponse:
    questions = request.questions if request.questions is not None else request.question
    if not questions:
        return JSONResponse(status_code=422, content={"error": "`questions` is required"})

    model_id = request.model or CONFIG.model
    try:
        results = classify(
            questions,
            request.state,
            model_id=model_id,
            save_to=CONFIG.save_to_for(model_id),
            device=CONFIG.device,
            gpu=CONFIG.gpu,
            temperature=request.temperature if request.temperature is not None else CONFIG.temperature,
            use_kv_cache=request.kv_cache if request.kv_cache is not None else CONFIG.kv_cache,
            calibrate=request.calibrate if request.calibrate is not None else CONFIG.calibrate,
            calibration_context=request.calibration_context or CONFIG.calibration_context,
        )
    except (ValueError, RuntimeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    answers = {r.name: r.to_dict() for r in results}
    usage = {
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
    }
    return JSONResponse(
        content={
            "model": model_id,
            "answers": answers,
            "results": list(answers.values()),
            "usage": usage,
        }
    )


def main() -> None:
    uvicorn.run("zero_shot.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
