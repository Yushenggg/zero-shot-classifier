from __future__ import annotations

from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .classifier import classify
from .config import load_config
from .scorer import is_loaded

STATIC_DIR = Path(__file__).parent / "static"
CONFIG = load_config()

app = FastAPI(title="Zero-Shot Classifier")


class ClassifyRequest(BaseModel):
    question: Any
    state: Any = None
    model_id: str | None = None
    temperature: float = 1.0


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health(model_id: str | None = None) -> dict[str, Any]:
    model_id = model_id or CONFIG.model
    return {"ok": True, "model_id": model_id, "model_loaded": is_loaded(model_id)}


@app.post("/api/classify")
def classify_endpoint(request: ClassifyRequest) -> JSONResponse:
    try:
        results = classify(
            request.question,
            request.state,
            model_id=request.model_id or CONFIG.model,
            save_to=CONFIG.resolve_save_to(),
            temperature=request.temperature,
        )
    except (ValueError, RuntimeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return JSONResponse(content={"results": [r.to_dict() for r in results]})


def main() -> None:
    uvicorn.run("zero_shot.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
