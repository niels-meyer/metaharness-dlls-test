# DLSS Harness Experiment Web App

This project simulates a simplified DLSS-style upscaling pipeline and compares:

- **Fixed Harness** (static, single-pass evaluation)
- **Meta-Harness (Evolving)** (recursive, adaptive loop)

## Stack

- Frontend: React + Vite + Recharts
- Backend: FastAPI
- Image processing: Pillow + NumPy

## Environment

The project is now configured around Google Gemini by default.

- Backend reads `backend/.env`
- Frontend reads `frontend/.env`

Backend variables:

- `MODEL_API_URL`
- `MODEL_API_MODEL`
- `MODEL_API_KEY`
- `CORS_ALLOW_ORIGINS`
- `CORS_ALLOW_LOCALHOST_REGEX`
- `MAX_UPLOAD_BYTES`
- `MAX_IMAGE_PIXELS`
- `MAX_IMAGE_WIDTH`
- `MAX_IMAGE_HEIGHT`
- `MAX_STORE_IMAGES`

Frontend variables:

- `VITE_API_BASE`
- `VITE_DEFAULT_MODEL_API_URL`
- `VITE_DEFAULT_MODEL_API_MODEL`

## Features Implemented

- Image upload
- Mode switch (Fixed Harness vs Meta-Harness)
- Iteration control for meta-harness
- Adversarial input toggle (noise + grid/stripes + text overlay)
- Side-by-side images:
  - Original
  - Low-resolution
  - Baseline bicubic upscale
  - Enhanced pseudo-DLSS upscale
  - Error heatmap
- Metrics:
  - MSE
  - PSNR
  - SSIM (approximation)
- Meta-harness history chart across iterations
- Parameter evolution returned by backend and plotted in UI

## Backend API Endpoints

- `POST /upload`
- `POST /downscale`
- `POST /upscale`
- `POST /evaluate`
- `POST /run-fixed-harness`
- `POST /run-meta-harness`
- `POST /run-api-meta-harness`
- `GET /store-stats`

## Security And Stability Hardening

- CORS is restricted to local origins by default (`localhost` / `127.0.0.1`) instead of wildcard `*`.
- Uploads are guarded by file-size and image-dimension limits to avoid out-of-memory crashes.
- The in-memory image store is bounded (`MAX_STORE_IMAGES`) and evicts old entries automatically.
- Temporary build/wheel artifacts are ignored and cleaned up for better repository hygiene.

## Fixed Harness Behavior

- Uses fixed parameters (`FIXED_PARAMS`)
- Runs exactly once
- No adaptation

## Meta-Harness Behavior

Each iteration:

1. Upscales using current parameters
2. Evaluates MSE/PSNR/SSIM
3. Computes detail-region vs smooth-region error
4. Adapts sharpening/edge/detail parameters
5. Stores iteration in history
6. Feeds evolved parameters into the next iteration

No hardcoded improvement is used: outputs are recomputed each iteration with new parameters.

## Model-API Harness Behavior

The project also supports a third mode that calls an external image model API over multiple iterations.

Per iteration:

1. Send low-res input + prompt (+ previous API output) to the model API
2. Evaluate output with MSE/PSNR/SSIM
3. Adapt local enhancement parameters and prompt hints
4. Repeat and keep the best-scoring iteration

Expected external API request/response:

- Request JSON fields:
  - `image_base64` (required)
  - `target_width` (required)
  - `target_height` (required)
  - `prompt` (required)
  - `model` (optional)
  - `iteration` (optional)
  - `previous_output_base64` (optional)
- Response JSON fields:
  - `image_base64` (required)

You can pass API config in request body (`api_url`, `api_key`, `model`, `prompt`) or set environment variables:

- `MODEL_API_URL`
- `MODEL_API_KEY`
- `MODEL_API_MODEL`

### Direct Google Gemini Usage

The backend now supports Google Gemini image endpoints directly (no proxy required).

Use these values in the UI for Model-API mode:

- `Model API URL`: `https://generativelanguage.googleapis.com/v1beta`
- `Model Name`: `gemini-2.5-flash-image`
- `API Key`: your Google AI Studio key

You can also provide a full endpoint URL like:

`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent`

## Run Locally

### 1) Backend

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8010
```

Edit `backend/.env` to set your Google API key.

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Edit `frontend/.env` if you want to change the default API base or prefill a different model URL/model name.

Open the frontend URL printed by Vite (default: `http://127.0.0.1:5173`).

If you want to use another backend URL/port, set `VITE_API_BASE` before starting the frontend.

## Notes

- This is a conceptual simulation inspired by DLSS, not an implementation of proprietary DLSS internals.
- For accurate comparison, use the same uploaded image and toggle modes/adversarial setting to observe static vs adaptive behavior.
