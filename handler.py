import os
import time
import base64
import io
from typing import List, Dict, Any

import runpod
from PIL import Image
from pdf2image import convert_from_bytes

# Force Surya to use torch backend (no Docker, no vLLM)
os.environ.setdefault("SURYA_INFERENCE_BACKEND", "torch")
os.environ.setdefault("SURYA_MODEL_CHECKPOINT", "/workspace/models/bina")

from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

# Lazy-load once
_MANAGER = None
_PREDICTOR = None

def _get_predictor():
    global _MANAGER, _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    _MANAGER = SuryaInferenceManager()
    _PREDICTOR = RecognitionPredictor(_MANAGER)
    return _PREDICTOR

def _decode_to_images(file_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
    try:
        images = convert_from_bytes(file_bytes, dpi=dpi)
        return [im.convert("RGB") for im in images]
    except Exception:
        return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

def handler(job):
    t0 = time.perf_counter()

    job_input = job.get("input", {})
    b64 = job_input.get("file_base64")
    if not b64:
        return {"error": "Missing 'file_base64' in input"}

    params = job_input.get("params", {})
    dpi = params.get("dpi", 150)

    file_bytes = base64.b64decode(b64)
    images = _decode_to_images(file_bytes, dpi=dpi)

    predictor = _get_predictor()
    predictions = predictor(images)

    pages = []
    for page_pred in predictions:
        page_lines = []
        if hasattr(page_pred, "text_lines"):
            for line in page_pred.text_lines:
                page_lines.append({
                    "text": getattr(line, "text", ""),
                    "bbox": getattr(line, "bbox", None)
                })
        else:
            for block in getattr(page_pred, "blocks", []):
                page_lines.append({
                    "text": getattr(block, "text", getattr(block, "html", "")),
                    "bbox": getattr(block, "bbox", None)
                })
        pages.append(page_lines)

    elapsed = time.perf_counter() - t0
    return {
        "pages": pages,
        "elapsed_sec": round(elapsed, 3)
    }

runpod.serverless.start({"handler": handler})