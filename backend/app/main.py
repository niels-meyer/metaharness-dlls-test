import os
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .image_ops import (
    FIXED_PARAMS,
    ImageStore,
    decode_image_bytes,
    difference_heatmap,
    downscale_image,
    encode_png_base64,
    evaluate_with_regions,
    run_api_meta_harness_pipeline,
    run_fixed_harness_pipeline,
    run_meta_harness_pipeline,
    upscale_bicubic,
    upscale_enhanced,
    HarnessParams,
)


app = FastAPI(title="DLSS Meta-Harness Experiment API", version="1.0.0")

load_dotenv()

DEFAULT_LOCAL_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
if cors_origins_env.strip():
    allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    allowed_origins = DEFAULT_LOCAL_ORIGINS

allow_localhost_regex = os.getenv("CORS_ALLOW_LOCALHOST_REGEX", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
max_image_pixels = int(os.getenv("MAX_IMAGE_PIXELS", str(12_000_000)))
max_image_width = int(os.getenv("MAX_IMAGE_WIDTH", "6000"))
max_image_height = int(os.getenv("MAX_IMAGE_HEIGHT", "6000"))
max_store_images = int(os.getenv("MAX_STORE_IMAGES", "80"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$" if allow_localhost_regex else None,
)

store = ImageStore(max_images=max_store_images)


class ImageRequest(BaseModel):
    image_id: str


class DownscaleRequest(ImageRequest):
    scaling_factor: int = Field(default=2, ge=2, le=8)


class UpscaleRequest(ImageRequest):
    method: str = Field(default="enhanced", pattern="^(baseline|enhanced)$")
    sharpening_strength: float = Field(default=0.8, ge=0.1, le=3.0)
    edge_strength: float = Field(default=0.2, ge=0.0, le=1.0)
    detail_boost: float = Field(default=1.0, ge=0.2, le=3.0)
    adversarial: bool = False


class EvaluateRequest(ImageRequest):
    predicted_image_id: Optional[str] = None


class HarnessRequest(ImageRequest):
    adversarial: bool = False


class MetaHarnessRequest(HarnessRequest):
    iterations: int = Field(default=8, ge=2, le=30)


class ApiMetaHarnessRequest(MetaHarnessRequest):
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    prompt: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file too large ({len(content)} bytes). Max allowed is {max_upload_bytes} bytes.",
        )

    try:
        image = decode_image_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    height, width = image.shape[:2]
    if width > max_image_width or height > max_image_height:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image dimensions too large ({width}x{height}). "
                f"Max allowed is {max_image_width}x{max_image_height}."
            ),
        )

    if width * height > max_image_pixels:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image has too many pixels ({width * height}). "
                f"Max allowed is {max_image_pixels}."
            ),
        )

    image_id = store.put(image)
    return {
        "image_id": image_id,
        "shape": list(image.shape),
    }


@app.post("/downscale")
def downscale(req: DownscaleRequest) -> dict:
    try:
        image = store.get(req.image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    low_res = downscale_image(image, req.scaling_factor)
    low_res_id = store.put(low_res)

    return {
        "image_id": low_res_id,
        "image_base64": encode_png_base64(low_res),
        "shape": list(low_res.shape),
    }


@app.post("/upscale")
def upscale(req: UpscaleRequest) -> dict:
    try:
        image = store.get(req.image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if req.adversarial:
        from .image_ops import apply_adversarial_inputs

        image = apply_adversarial_inputs(image)

    if req.method == "baseline":
        # Use a fixed x2 upscale target for direct endpoint usage.
        out = upscale_bicubic(image, (image.shape[0] * 2, image.shape[1] * 2))
    else:
        params = HarnessParams(
            sharpening_strength=req.sharpening_strength,
            edge_strength=req.edge_strength,
            detail_boost=req.detail_boost,
        )
        out = upscale_enhanced(image, (image.shape[0] * 2, image.shape[1] * 2), params)

    output_id = store.put(out)
    return {
        "image_id": output_id,
        "image_base64": encode_png_base64(out),
        "shape": list(out.shape),
    }


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    try:
        gt = store.get(req.image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"ground truth missing: {exc}") from exc

    if not req.predicted_image_id:
        raise HTTPException(status_code=400, detail="predicted_image_id is required")

    try:
        pred = store.get(req.predicted_image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"predicted image missing: {exc}") from exc

    if gt.shape != pred.shape:
        pred = upscale_bicubic(pred, gt.shape[:2])

    metrics = evaluate_with_regions(gt, pred)
    heat = difference_heatmap(gt, pred)

    return {
        "metrics": metrics,
        "error_heatmap_base64": encode_png_base64(heat),
    }


@app.post("/run-fixed-harness")
def run_fixed_harness(req: HarnessRequest) -> dict:
    try:
        gt = store.get(req.image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = run_fixed_harness_pipeline(gt, adversarial=req.adversarial)
    return result


@app.post("/run-meta-harness")
def run_meta_harness(req: MetaHarnessRequest) -> dict:
    try:
        gt = store.get(req.image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = run_meta_harness_pipeline(gt, iterations=req.iterations, adversarial=req.adversarial)
    return result


@app.post("/run-api-meta-harness")
def run_api_meta_harness(req: ApiMetaHarnessRequest) -> dict:
    try:
        gt = store.get(req.image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        result = run_api_meta_harness_pipeline(
            gt,
            iterations=req.iterations,
            adversarial=req.adversarial,
            api_url=req.api_url,
            api_key=req.api_key,
            model=req.model,
            prompt=req.prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model API request failed: {exc}") from exc

    return result


@app.get("/default-fixed-parameters")
def default_fixed_parameters() -> dict:
    return FIXED_PARAMS.to_dict()


@app.get("/store-stats")
def store_stats() -> dict:
    return store.stats()
