"""Local research demo API; uploaded image bytes are processed in memory only."""
from __future__ import annotations
import io
import os
from pathlib import Path
from functools import lru_cache

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from .data import image_transform
from .model import SUPPORTED_ARCHITECTURES
from .train import load_checkpoint

ROOT = Path(os.environ.get("TB_CXR_ROOT", Path(__file__).resolve().parents[2])).resolve()
WEB_FILE = ROOT / "web/index.html"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def available_models() -> list[dict]:
    models = []
    for path in sorted((ROOT / "artifacts/runs").glob("train_*/*/best_model.pt")):
        source = path.parents[1].name.removeprefix("train_")
        arch = path.parent.name
        models.append({"id": f"{source}:{arch}", "train_source": source,
                       "architecture": arch, "path": str(path)})
    return models


@lru_cache(maxsize=4)
def _load(path_string: str):
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_checkpoint(path, device="cpu")


def create_app() -> FastAPI:
    app = FastAPI(title="TB CXR Research Demo", version="1.0.0",
                  description="Research-only binary chest radiograph classifier; not for diagnosis.")

    @app.get("/", include_in_schema=False)
    def index():
        if not WEB_FILE.exists():
            raise HTTPException(status_code=404, detail="Demo page is missing")
        return FileResponse(WEB_FILE)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "models_available": len(available_models()),
                "clinical_use": False}

    @app.get("/api/models")
    def models():
        return [{k: v for k, v in item.items() if k != "path"} for item in available_models()]

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...), model_id: str | None = None):
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image.")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Maximum upload size is 8 MB.")
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")
        choices = available_models()
        if model_id:
            selected = next((m for m in choices if m["id"] == model_id), None)
        else:
            selected = choices[0] if choices else None
        if selected is None:
            raise HTTPException(status_code=503, detail="No trained checkpoint is available. Run the experiment first.")
        try:
            model, checkpoint = _load(selected["path"])
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Selected checkpoint is unavailable.")
        transform = image_transform(checkpoint["image_size"], train=False)
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            probability = float(torch.softmax(model(x), dim=1)[0, 1].item())
        prediction = "abnormal_tb_associated" if probability >= 0.5 else "normal"
        return {
            "model": {"architecture": selected["architecture"], "trained_on": selected["train_source"]},
            "prediction": prediction,
            "score": probability,
            "threshold": 0.5,
            "warning": "Research demonstration only. This score is not a calibrated probability, TB diagnosis, or substitute for radiologist review and confirmatory testing.",
        }
    return app

app = create_app()
