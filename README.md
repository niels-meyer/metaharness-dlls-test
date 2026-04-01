# DLSS Harness Experiment Web App

This project simulates a simplified DLSS-style upscaling pipeline and compares:

- **Fixed Harness** (static, single-pass evaluation)
- **Meta-Harness (Evolving)** (recursive, adaptive loop)

## Stack

- Frontend: React + Vite + Recharts
- Backend: FastAPI
- Image processing: Pillow + NumPy

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

## Run Locally

### 1) Backend

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8010
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the frontend URL printed by Vite (default: `http://127.0.0.1:5173`).

If you want to use another backend URL/port, set `VITE_API_BASE` before starting the frontend.

## Notes

- This is a conceptual simulation inspired by DLSS, not an implementation of proprietary DLSS internals.
- For accurate comparison, use the same uploaded image and toggle modes/adversarial setting to observe static vs adaptive behavior.
