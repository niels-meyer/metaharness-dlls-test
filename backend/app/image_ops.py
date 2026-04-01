import base64
import uuid
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class HarnessParams:
    sharpening_strength: float
    edge_strength: float
    detail_boost: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class ImageStore:
    def __init__(self) -> None:
        self._store: Dict[str, np.ndarray] = {}

    def put(self, image: np.ndarray) -> str:
        image_id = str(uuid.uuid4())
        self._store[image_id] = image
        return image_id

    def get(self, image_id: str) -> np.ndarray:
        if image_id not in self._store:
            raise KeyError(f"image_id not found: {image_id}")
        return self._store[image_id]


FIXED_PARAMS = HarnessParams(sharpening_strength=0.55, edge_strength=0.1, detail_boost=0.9)


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.astype(np.uint8), mode="RGB")


def _to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _to_gray(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _conv2d(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(gray, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(gray, dtype=np.float32)

    for y in range(kh):
        for x in range(kw):
            out += kernel[y, x] * padded[y : y + gray.shape[0], x : x + gray.shape[1]]

    return out


def _gaussian_kernel(size: int = 7, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _blur_gray(gray: np.ndarray, size: int = 7, sigma: float = 1.5) -> np.ndarray:
    return _conv2d(gray.astype(np.float32), _gaussian_kernel(size=size, sigma=sigma))


def decode_image_bytes(content: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive parsing branch.
        raise ValueError("Failed to decode image") from exc

    if image is None:
        raise ValueError("Failed to decode image")
    return _to_np(image)


def encode_png_base64(image: np.ndarray) -> str:
    buffer = BytesIO()
    _to_pil(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def downscale_image(image: np.ndarray, scaling_factor: int = 2) -> np.ndarray:
    h, w = image.shape[:2]
    target = (max(1, w // scaling_factor), max(1, h // scaling_factor))
    return _to_np(_to_pil(image).resize(target, Image.Resampling.BOX))


def upscale_bicubic(low_res: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    return _to_np(_to_pil(low_res).resize((target_w, target_h), Image.Resampling.BICUBIC))


def _unsharp_mask(image: np.ndarray, amount: float) -> np.ndarray:
    blurred = _to_np(_to_pil(image).filter(ImageFilter.GaussianBlur(radius=1.2))).astype(np.float32)
    sharp = image.astype(np.float32) * (1.0 + amount) - blurred * amount
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _normalized_gradient_map(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = _conv2d(gray, kx)
    gy = _conv2d(gray, ky)
    grad = np.sqrt(gx * gx + gy * gy)
    gmin = float(np.min(grad))
    gmax = float(np.max(grad))
    if gmax - gmin < 1e-8:
        return np.zeros_like(grad, dtype=np.float32)
    return ((grad - gmin) / (gmax - gmin)).astype(np.float32)


def upscale_enhanced(low_res: np.ndarray, target_shape: Tuple[int, int], params: HarnessParams) -> np.ndarray:
    base = upscale_bicubic(low_res, target_shape)
    grad_map = _normalized_gradient_map(base)

    sharpened = _unsharp_mask(base, params.sharpening_strength)

    lap_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    gray = _to_gray(base)
    lap_abs = np.abs(_conv2d(gray, lap_k))
    lap_abs = np.clip(lap_abs, 0, 255)
    edge_boosted = base.astype(np.float32) + lap_abs[..., None] * params.edge_strength

    grad_3 = np.repeat(grad_map[..., None], 3, axis=2)
    detail_mask = np.clip(grad_3 * params.detail_boost, 0.0, 1.0)

    mixed = (edge_boosted.astype(np.float32) * detail_mask) + (
        sharpened.astype(np.float32) * (1.0 - detail_mask)
    )

    # Blend back toward bicubic to avoid harsh artifacts from aggressive filters.
    alpha = float(np.clip(0.35 + 0.2 * params.detail_boost, 0.3, 0.75))
    stabilized = base.astype(np.float32) * (1.0 - alpha) + mixed * alpha
    return np.clip(stabilized, 0, 255).astype(np.uint8)


def apply_adversarial_inputs(image: np.ndarray, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy().astype(np.float32)

    noise = rng.normal(loc=0.0, scale=11.0, size=out.shape)
    out += noise

    h, w = out.shape[:2]
    stripe = np.zeros((h, w), dtype=np.float32)
    stripe[:, ::6] = 18.0
    stripe[::6, :] += 18.0
    out[..., 1] += stripe * 0.35
    out[..., 2] += stripe * 0.15

    overlay_img = _to_pil(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay_img)
    draw.text((10, max(20, h // 6)), "ADVERSARIAL", fill=(250, 220, 40))
    return _to_np(overlay_img)


def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    diff = gt.astype(np.float32) - pred.astype(np.float32)
    return float(np.mean(np.square(diff)))


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    err = mse(gt, pred)
    if err <= 1e-10:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(err)))


def ssim_approx(gt: np.ndarray, pred: np.ndarray) -> float:
    x = _to_gray(gt).astype(np.float32)
    y = _to_gray(pred).astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ux = _blur_gray(x, size=7, sigma=1.5)
    uy = _blur_gray(y, size=7, sigma=1.5)

    uxx = _blur_gray(x * x, size=7, sigma=1.5)
    uyy = _blur_gray(y * y, size=7, sigma=1.5)
    uxy = _blur_gray(x * y, size=7, sigma=1.5)

    sx = uxx - ux * ux
    sy = uyy - uy * uy
    sxy = uxy - ux * uy

    num = (2 * ux * uy + c1) * (2 * sxy + c2)
    den = (ux * ux + uy * uy + c1) * (sx + sy + c2)
    score = np.mean(num / (den + 1e-8))
    return float(np.clip(score, 0.0, 1.0))


def difference_heatmap(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    diff = np.abs(gt.astype(np.float32) - pred.astype(np.float32))
    gray = np.mean(diff, axis=2) / 255.0
    gray = np.clip(gray, 0.0, 1.0)

    r = np.clip(1.8 * gray, 0.0, 1.0)
    g = np.clip((gray - 0.25) * 1.6, 0.0, 1.0)
    b = np.clip((gray - 0.55) * 2.0, 0.0, 1.0)

    heat = np.stack([r, g, b], axis=2)
    return np.clip(heat * 255.0, 0, 255).astype(np.uint8)


def evaluate_with_regions(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    base_mse = mse(gt, pred)
    base_psnr = psnr(gt, pred)
    base_ssim = ssim_approx(gt, pred)

    grad = _normalized_gradient_map(gt)
    detail_mask = grad > np.percentile(grad, 70)
    smooth_mask = ~detail_mask

    sq_err = np.mean((gt.astype(np.float32) - pred.astype(np.float32)) ** 2, axis=2)

    detail_err = float(np.mean(sq_err[detail_mask])) if np.any(detail_mask) else base_mse
    smooth_err = float(np.mean(sq_err[smooth_mask])) if np.any(smooth_mask) else base_mse

    return {
        "mse": base_mse,
        "psnr": base_psnr,
        "ssim": base_ssim,
        "detail_error": detail_err,
        "smooth_error": smooth_err,
    }


def adapt_params(prev: HarnessParams, metrics: Dict[str, float], prev_psnr: float | None) -> HarnessParams:
    next_params = HarnessParams(
        sharpening_strength=prev.sharpening_strength,
        edge_strength=prev.edge_strength,
        detail_boost=prev.detail_boost,
    )

    detail_err = metrics["detail_error"]
    smooth_err = metrics["smooth_error"]

    if detail_err > smooth_err:
        next_params.sharpening_strength += 0.03
        next_params.edge_strength += 0.015
        next_params.detail_boost += 0.03
    else:
        next_params.sharpening_strength -= 0.025
        next_params.edge_strength -= 0.01
        next_params.detail_boost -= 0.02

    if prev_psnr is not None and metrics["psnr"] < prev_psnr:
        next_params.sharpening_strength *= 0.97
        next_params.edge_strength *= 0.97

    next_params.sharpening_strength = float(np.clip(next_params.sharpening_strength, 0.35, 2.2))
    next_params.edge_strength = float(np.clip(next_params.edge_strength, 0.05, 0.85))
    next_params.detail_boost = float(np.clip(next_params.detail_boost, 0.65, 2.0))

    return next_params


def run_fixed_harness_pipeline(gt_image: np.ndarray, adversarial: bool) -> Dict:
    low_res = downscale_image(gt_image, scaling_factor=2)
    low_res_eval = apply_adversarial_inputs(low_res) if adversarial else low_res

    baseline = upscale_bicubic(low_res_eval, gt_image.shape[:2])
    enhanced = upscale_enhanced(low_res_eval, gt_image.shape[:2], FIXED_PARAMS)

    baseline_metrics = evaluate_with_regions(gt_image, baseline)
    enhanced_metrics = evaluate_with_regions(gt_image, enhanced)

    heatmap = difference_heatmap(gt_image, enhanced)

    return {
        "mode": "fixed",
        "iterations": 1,
        "parameters": FIXED_PARAMS.to_dict(),
        "history": [
            {
                "iteration": 1,
                "params": FIXED_PARAMS.to_dict(),
                "metrics": enhanced_metrics,
            }
        ],
        "images": {
            "ground_truth": encode_png_base64(gt_image),
            "low_res": encode_png_base64(low_res_eval),
            "baseline_upscale": encode_png_base64(baseline),
            "enhanced_upscale": encode_png_base64(enhanced),
            "error_heatmap": encode_png_base64(heatmap),
        },
        "metrics": {
            "baseline": baseline_metrics,
            "enhanced": enhanced_metrics,
        },
    }


def run_meta_harness_pipeline(gt_image: np.ndarray, iterations: int, adversarial: bool) -> Dict:
    low_res = downscale_image(gt_image, scaling_factor=2)
    low_res_eval = apply_adversarial_inputs(low_res) if adversarial else low_res

    baseline = upscale_bicubic(low_res_eval, gt_image.shape[:2])
    baseline_metrics = evaluate_with_regions(gt_image, baseline)

    current_params = HarnessParams(sharpening_strength=0.45, edge_strength=0.1, detail_boost=0.9)
    history: List[Dict] = []

    best_image = None
    best_metrics = None
    best_iteration = 0
    best_psnr = -1e9
    prev_psnr = None

    for i in range(iterations):
        enhanced = upscale_enhanced(low_res_eval, gt_image.shape[:2], current_params)
        metrics = evaluate_with_regions(gt_image, enhanced)

        history.append(
            {
                "iteration": i + 1,
                "params": current_params.to_dict(),
                "metrics": metrics,
            }
        )

        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            best_image = enhanced
            best_metrics = metrics
            best_iteration = i + 1

        current_params = adapt_params(current_params, metrics, prev_psnr)
        prev_psnr = metrics["psnr"]

    if best_image is None:
        best_image = baseline

    heatmap = difference_heatmap(gt_image, best_image)

    return {
        "mode": "meta",
        "iterations": iterations,
        "final_parameters": current_params.to_dict(),
        "best_iteration": best_iteration,
        "history": history,
        "images": {
            "ground_truth": encode_png_base64(gt_image),
            "low_res": encode_png_base64(low_res_eval),
            "baseline_upscale": encode_png_base64(baseline),
            "enhanced_upscale": encode_png_base64(best_image),
            "error_heatmap": encode_png_base64(heatmap),
        },
        "metrics": {
            "baseline": baseline_metrics,
            "enhanced": best_metrics if best_metrics else baseline_metrics,
        },
    }
