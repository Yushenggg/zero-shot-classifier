from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .classifier import classify
from .config import load_config
from .scorer import get_scorer, is_loaded

STATIC_DIR = Path(__file__).parent / "static"
CONFIG = load_config()
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
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
        logger.info(
            "Model ready: %s on %s", type(scorer._model).__name__, scorer.device
        )
    except Exception as exc:  # noqa: BLE001 - keep serving; retry on first request
        logger.warning("Model preload failed (%s: %s); will load on first request.", type(exc).__name__, exc)
    yield


app = FastAPI(title="Zero-Shot Classifier", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    question: Any
    state: Any = None
    temperature: float = 1.0
    kv_cache: bool | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_id": CONFIG.model,
        "device": CONFIG.device,
        "gpu": CONFIG.gpu,
        "kv_cache": CONFIG.kv_cache,
        "model_loaded": is_loaded(CONFIG.model),
    }


@app.post("/api/classify")
def classify_endpoint(request: ClassifyRequest) -> JSONResponse:
    try:
        results = classify(
            request.question,
            request.state,
            model_id=CONFIG.model,
            save_to=CONFIG.resolve_save_to(),
            device=CONFIG.device,
            gpu=CONFIG.gpu,
            temperature=request.temperature,
            use_kv_cache=request.kv_cache if request.kv_cache is not None else CONFIG.kv_cache,
        )
    except (ValueError, RuntimeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return JSONResponse(content={"results": [r.to_dict() for r in results]})


def main() -> None:
    uvicorn.run("zero_shot.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
