import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8010";

function MetricCard({ title, value }) {
  return (
    <div className="metric-card">
      <p>{title}</p>
      <strong>{value}</strong>
    </div>
  );
}

function ImagePanel({ title, base64 }) {
  return (
    <div className="image-panel">
      <h4>{title}</h4>
      {base64 ? <img src={`data:image/png;base64,${base64}`} alt={title} /> : <div className="placeholder">No image</div>}
    </div>
  );
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageId, setImageId] = useState("");
  const [mode, setMode] = useState("fixed");
  const [iterations, setIterations] = useState(10);
  const [adversarial, setAdversarial] = useState(false);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("Upload an image and run the harness.");
  const [busy, setBusy] = useState(false);

  const chartData = useMemo(() => {
    if (!result?.history) {
      return [];
    }
    return result.history.map((h) => ({
      iteration: h.iteration,
      psnr: Number(h.metrics.psnr.toFixed(3)),
      mse: Number(h.metrics.mse.toFixed(3)),
      ssim: Number(h.metrics.ssim.toFixed(4)),
      sharpening: Number(h.params.sharpening_strength.toFixed(3)),
      edge: Number(h.params.edge_strength.toFixed(3)),
    }));
  }, [result]);

  async function uploadImage() {
    if (!selectedFile) {
      throw new Error("Select an image first");
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Upload failed (${res.status})`);
    }

    const data = await res.json();
    setImageId(data.image_id);
    return data.image_id;
  }

  async function runHarness(forcedMode = mode, existingImageId = imageId) {
    setBusy(true);
    try {
      const effectiveImageId = existingImageId || (await uploadImage());
      const endpoint = forcedMode === "fixed" ? "/run-fixed-harness" : "/run-meta-harness";

      const payload = {
        image_id: effectiveImageId,
        adversarial,
      };

      if (forcedMode === "meta") {
        payload.iterations = iterations;
      }

      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Harness run failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
      setStatus(`Run completed in ${forcedMode.toUpperCase()} mode.`);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!imageId) {
      return;
    }
    runHarness(mode, imageId).catch((err) => {
      setStatus(`Mode switch rerun failed: ${err.message}`);
    });
    // Re-run pipeline on mode switch to make behavior differences explicit.
  }, [mode]);

  async function handleRunClick() {
    try {
      setStatus("Running harness pipeline...");
      await runHarness(mode, imageId);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Simulation Lab</p>
          <h1>DLSS Harness Experiment</h1>
          <p className="subtitle">
            Compare static evaluation against a recursive, self-improving meta-harness.
          </p>
        </div>
        <div className="status-pill">{status}</div>
      </header>

      <section className="control-grid">
        <label className="control card">
          <span>Upload Ground Truth</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              const f = e.target.files?.[0] || null;
              setSelectedFile(f);
              setImageId("");
              setResult(null);
              setStatus("Image selected. Run harness to evaluate.");
            }}
          />
        </label>

        <div className="control card">
          <span>Harness Mode</span>
          <div className="toggle-row">
            <button
              className={mode === "fixed" ? "active" : ""}
              onClick={() => setMode("fixed")}
            >
              Fixed Harness
            </button>
            <button
              className={mode === "meta" ? "active" : ""}
              onClick={() => setMode("meta")}
            >
              Meta-Harness (Evolving)
            </button>
          </div>
        </div>

        <label className="control card">
          <span>Meta Iterations</span>
          <input
            type="range"
            min={2}
            max={25}
            value={iterations}
            disabled={mode !== "meta"}
            onChange={(e) => setIterations(Number(e.target.value))}
          />
          <strong>{iterations}</strong>
        </label>

        <label className="control card check">
          <input
            type="checkbox"
            checked={adversarial}
            onChange={(e) => setAdversarial(e.target.checked)}
          />
          <span>Enable adversarial inputs (noise + patterns + text)</span>
        </label>
      </section>

      <div className="actions">
        <button disabled={busy || !selectedFile} onClick={handleRunClick}>
          {busy ? "Running..." : "Run Pipeline"}
        </button>
      </div>

      {result && (
        <>
          <section className="metric-grid">
            <MetricCard title="Mode" value={result.mode} />
            <MetricCard title="Iterations" value={result.iterations} />
            <MetricCard
              title="Baseline PSNR"
              value={result.metrics?.baseline?.psnr?.toFixed(3)}
            />
            <MetricCard
              title="Enhanced PSNR"
              value={result.metrics?.enhanced?.psnr?.toFixed(3)}
            />
            <MetricCard
              title="Enhanced SSIM"
              value={result.metrics?.enhanced?.ssim?.toFixed(4)}
            />
          </section>

          <section className="image-grid">
            <ImagePanel title="Original" base64={result.images?.ground_truth} />
            <ImagePanel title="Low-Resolution" base64={result.images?.low_res} />
            <ImagePanel title="Baseline Upscale (Bicubic)" base64={result.images?.baseline_upscale} />
            <ImagePanel title="Enhanced Upscale (Pseudo-DLSS)" base64={result.images?.enhanced_upscale} />
            <ImagePanel title="Error Heatmap" base64={result.images?.error_heatmap} />
          </section>

          {mode === "meta" && chartData.length > 0 && (
            <section className="chart-card">
              <h3>Meta-Harness Iteration History</h3>
              <p>
                The graph below proves recursive adaptation: parameters evolve from prior evaluation results,
                and quality metrics are recomputed each iteration.
              </p>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2d4f54" />
                    <XAxis dataKey="iteration" stroke="#d6eceb" />
                    <YAxis yAxisId="left" stroke="#d6eceb" />
                    <YAxis yAxisId="right" orientation="right" stroke="#ffd4a3" />
                    <Tooltip />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="psnr" stroke="#f4a259" strokeWidth={2} />
                    <Line yAxisId="left" type="monotone" dataKey="ssim" stroke="#8bd3dd" strokeWidth={2} />
                    <Line yAxisId="right" type="monotone" dataKey="sharpening" stroke="#ff6b6b" strokeWidth={2} />
                    <Line yAxisId="right" type="monotone" dataKey="edge" stroke="#c7f464" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
