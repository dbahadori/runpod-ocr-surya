import os
import time
import base64
import io
from typing import List, Dict, Any

import runpod
from PIL import Image
from pdf2image import convert_from_bytes

from surya.model.detection import segformer
from surya.model.recognition import vllm
from surya.ocr import run_ocr

MODEL_PATH = os.getenv("BINA_MODEL_PATH", "/workspace/models/bina")

# Load models once (cold start)
DETECTOR = segformer.load_model()
RECOGNIZER = vllm.load_model(
    checkpoint=MODEL_PATH,
    backend="vllm"          # in-process, no Docker
)

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

    predictions = run_ocr(
        images,
        [img.size for img in images],
        DETECTOR,
        RECOGNIZER,
        batch_size=4
    )

    pages = []
    for page_pred in predictions:
        page_lines = []
        for line in page_pred.text_lines:
            page_lines.append({
                "text": line.text,
                "bbox": line.bbox
            })
        pages.append(page_lines)

    elapsed = time.perf_counter() - t0
    return {
        "pages": pages,
        "elapsed_sec": round(elapsed, 3)
    }

runpod.serverless.start({"handler": handler})