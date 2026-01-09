import argparse
import csv
import json
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse, parse_qs

import numpy as np
from stable_baselines3 import PPO

from train_pong_ppo import SB3PongEnv
from pong import simple_tracking_policy


ROOT = Path.cwd()
_HEATMAP_CACHE: Optional[List[List[int]]] = None
_HEATMAP_TS = 0.0


def _latest_report(log_dir: Path) -> Optional[Path]:
    reports = sorted(log_dir.glob("run_report_*.json"))
    return reports[-1] if reports else None


def _read_metrics(metrics_csv: Path) -> dict:
    if not metrics_csv.exists():
        return {}
    best = None
    with metrics_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                reward = float(row.get("avg_reward", 0.0))
            except Exception:
                reward = float("-inf")
            if best is None or reward > float(best.get("avg_reward", float("-inf"))):
                best = row
    return best or {}


def _heatmap_from_model(model_path: Path, steps: int = 1500, bins: int = 40) -> List[List[int]]:
    global _HEATMAP_CACHE, _HEATMAP_TS
    now = time.time()
    if _HEATMAP_CACHE is not None and (now - _HEATMAP_TS) < 10:
        return _HEATMAP_CACHE
    if not model_path.exists():
        return []
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        obs, _ = env.reset()
        heat = np.zeros((bins, bins), dtype=np.int32)
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            bx, by = obs[0], obs[1]
            x = min(bins - 1, max(0, int(bx * bins)))
            y = min(bins - 1, max(0, int(by * bins)))
            heat[y, x] += 1
            if terminated or truncated:
                obs, _ = env.reset()
        _HEATMAP_CACHE = heat.tolist()
        _HEATMAP_TS = now
        return _HEATMAP_CACHE
    finally:
        env.close()


class DashboardHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, content: bytes, content_type: str = "text/html") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, _dashboard_html().encode("utf-8"))
            return
        if parsed.path == "/api/status":
            log_dir = ROOT / "logs"
            metrics = _read_metrics(ROOT / "logs" / "metrics.csv")
            report_path = _latest_report(log_dir)
            report = json.loads(report_path.read_text()) if report_path else {}
            payload = {
                "metrics": metrics,
                "report": report,
            }
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
            return
        if parsed.path == "/api/metrics":
            metrics_csv = ROOT / "logs" / "metrics.csv"
            series = []
            err = None
            if metrics_csv.exists():
                with metrics_csv.open("r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        try:
                            series.append(
                                {
                                    "cycle": int(row.get("cycle", 0)),
                                    "model_id": row.get("model_id", ""),
                                    "avg_reward": float(row.get("avg_reward", 0.0)),
                                    "win_rate": float(row.get("win_rate", 0.0)),
                                    "avg_rally_length": float(row.get("avg_rally_length", 0.0)),
                                    "avg_return_rate": float(row.get("avg_return_rate", 0.0)),
                                    "delta_reward": float(row.get("delta_reward", 0.0)) if row.get("delta_reward") else 0.0,
                                }
                            )
                        except Exception:
                            continue
            else:
                err = "metrics.csv not found"
            self._send(200, json.dumps({"series": series, "error": err}).encode("utf-8"), "application/json")
            return
        if parsed.path == "/api/heatmap":
            model_path = ROOT / "models" / "ppo_pong_custom_latest.zip"
            best = _read_metrics(ROOT / "logs" / "metrics.csv")
            if best and best.get("model_id"):
                model_path = ROOT / "models" / f"{best['model_id']}_latest.zip"
            try:
                heat = _heatmap_from_model(model_path)
                self._send(200, json.dumps({"heatmap": heat}).encode("utf-8"), "application/json")
            except Exception as exc:
                self._send(200, json.dumps({"heatmap": [], "error": str(exc)}).encode("utf-8"), "application/json")
            return
        if parsed.path == "/file":
            qs = parse_qs(parsed.query)
            raw = qs.get("path", [""])[0]
            target = (ROOT / raw).resolve()
            if not str(target).startswith(str(ROOT)):
                self._send(403, b"forbidden", "text/plain")
                return
            if not target.exists():
                self._send(404, b"not found", "text/plain")
                return
            content_type = "application/octet-stream"
            if target.suffix.lower() == ".mp4":
                content_type = "video/mp4"
            self._send(200, target.read_bytes(), content_type)
            return
        self._send(404, b"not found", "text/plain")


def _dashboard_html() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Pong Trainer Dashboard</title>
  <style>
    :root {
      --bg: #0b0f14;
      --panel: #121820;
      --accent: #14f195;
      --accent2: #f5a623;
      --text: #e9f1ff;
    }
    body {
      margin: 0; font-family: 'Segoe UI', sans-serif; background: radial-gradient(circle at top, #151b24, #0b0f14);
      color: var(--text);
    }
    .layout { display: grid; grid-template-columns: 220px 1fr; min-height: 100vh; }
    .sidebar { background: #0f141b; padding: 16px; border-right: 1px solid #1f2a36; }
    .sidebar h2 { font-size: 12px; letter-spacing: 2px; color: #8aa3c5; text-transform: uppercase; }
    .menu button { width: 100%; margin: 6px 0; padding: 10px; background: #121820; color: var(--text); border: 1px solid #1f2a36; border-radius: 8px; cursor: pointer; }
    .menu button.active { border-color: var(--accent); color: var(--accent); }
    .wrap { padding: 20px; max-width: 1200px; margin: 0 auto; }
    h1 { letter-spacing: 2px; text-transform: uppercase; font-size: 20px; color: var(--accent); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .card { background: var(--panel); border: 1px solid #1f2a36; border-radius: 12px; padding: 16px; }
    .stat { font-size: 28px; color: var(--accent2); }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    canvas { width: 100%; height: auto; background: #0f141b; border-radius: 8px; }
    video { width: 100%; border-radius: 8px; border: 1px solid #203040; }
    .label { font-size: 12px; text-transform: uppercase; color: #8aa3c5; letter-spacing: 1px; }
    .panel { display: none; }
    .panel.active { display: block; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 6px 8px; border-bottom: 1px solid #1f2a36; text-align: left; }
  </style>
</head>
<body>
    <div class="layout">
      <div class="sidebar">
        <h2>Dashboard</h2>
        <div class="menu">
          <button data-panel="overview" class="active">Overview</button>
          <button data-panel="charts">Charts</button>
          <button data-panel="videos">Videos</button>
          <button data-panel="table">Recent Metrics</button>
        </div>
        <div class="label" id="statusLine" style="margin-top:12px;">Status: --</div>
        <div class="label" id="dataLine">Data: --</div>
      </div>
    <div class="wrap">
      <h1>Live Pong Training Dashboard</h1>
      <div id="panel-overview" class="panel active">
        <div class="grid">
          <div class="card">
            <div class="label">Best Model</div>
            <div id="bestModel" class="stat">--</div>
            <div class="label">Avg Reward</div>
            <div id="bestReward">--</div>
            <div class="label">Win Rate</div>
            <div id="bestWin">--</div>
          </div>
          <div class="card">
            <div class="label">Latest Run</div>
            <div id="runId">--</div>
            <div class="label">Stop Reason</div>
            <div id="stopReason">--</div>
          </div>
          <div class="card">
            <div class="label">Animated Heatmap</div>
            <canvas id="heatmap" width="400" height="400"></canvas>
            <div id="heatmapNote" class="label" style="margin-top:6px;">--</div>
          </div>
        </div>
      </div>
      <div id="panel-charts" class="panel">
        <div class="card">
          <div class="label">Training Charts</div>
          <div class="split">
            <div>
              <div class="label">Avg Reward (per cycle)</div>
              <canvas id="chartReward" width="500" height="200"></canvas>
            </div>
            <div>
              <div class="label">Win Rate (per cycle)</div>
              <canvas id="chartWin" width="500" height="200"></canvas>
            </div>
          </div>
          <div class="split" style="margin-top:12px;">
            <div>
              <div class="label">Delta Reward (per cycle)</div>
              <canvas id="chartDelta" width="500" height="200"></canvas>
            </div>
            <div>
              <div class="label">Avg Rally Length (per cycle)</div>
              <canvas id="chartRally" width="500" height="200"></canvas>
            </div>
          </div>
        </div>
      </div>
      <div id="panel-videos" class="panel">
        <div class="card">
          <div class="label">Comparative Split</div>
          <div class="split">
            <div>
              <div class="label">Latest Combined</div>
              <video id="vidCombined" controls muted></video>
            </div>
            <div>
              <div class="label">Extended Eval</div>
              <video id="vidEval" controls muted></video>
            </div>
          </div>
        </div>
      </div>
      <div id="panel-table" class="panel">
        <div class="card">
          <div class="label">Recent Metrics</div>
          <table>
            <thead>
              <tr>
                <th>Cycle</th>
                <th>Model</th>
                <th>Avg Reward</th>
                <th>Delta</th>
                <th>Win Rate</th>
                <th>Rally</th>
                <th>Return</th>
              </tr>
            </thead>
            <tbody id="metricsTable"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
<script>
let lastCombined = "";
let lastEval = "";
let lastStatusTick = 0;

async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const metrics = data.metrics || {};
    const report = (data.report || {}).summary || {};
    document.getElementById('bestModel').textContent = metrics.model_id || 'n/a';
    document.getElementById('bestReward').textContent = metrics.avg_reward || '--';
    document.getElementById('bestWin').textContent = metrics.win_rate || '--';
    document.getElementById('runId').textContent = report.run_timestamp || '--';
    document.getElementById('stopReason').textContent = report.stop_reason || '--';
    const combined = report.last_combined_video;
    const evalVid = report.last_eval_video;
    if (combined && combined !== lastCombined) {
      document.getElementById('vidCombined').src = '/file?path=' + encodeURIComponent(combined);
      lastCombined = combined;
    }
    if (evalVid && evalVid !== lastEval) {
      document.getElementById('vidEval').src = '/file?path=' + encodeURIComponent(evalVid);
      lastEval = evalVid;
    }
    lastStatusTick = Date.now();
    document.getElementById('statusLine').textContent = `Status: OK (${new Date().toLocaleTimeString()})`;
  } catch (err) {
    document.getElementById('statusLine').textContent = `Status: ERROR (${err})`;
  }
}
async function refreshHeatmap() {
  let data = null;
  try {
    const res = await fetch('/api/heatmap');
    data = await res.json();
  } catch (err) {
    document.getElementById('heatmapNote').textContent = 'Heatmap fetch failed.';
    drawEmpty(document.getElementById('heatmap'), 'Heatmap error');
    return;
  }
  if (data.error) {
    document.getElementById('heatmapNote').textContent = `Heatmap error: ${data.error}`;
    drawEmpty(document.getElementById('heatmap'), 'Heatmap error');
    return;
  }
  const heat = data.heatmap || [];
  const canvas = document.getElementById('heatmap');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!heat.length) {
    document.getElementById('heatmapNote').textContent = 'No heatmap yet (train a model first).';
    drawEmpty(canvas, 'No heatmap data');
    return;
  }
  document.getElementById('heatmapNote').textContent = 'Live density from latest model.';
  const rows = heat.length;
  const cols = heat[0].length;
  const max = Math.max(...heat.flat());
  for (let y=0; y<rows; y++) {
    for (let x=0; x<cols; x++) {
      const v = heat[y][x] / (max || 1);
      ctx.fillStyle = `rgba(20,241,149,${v})`;
      ctx.fillRect(x * canvas.width / cols, y * canvas.height / rows, canvas.width / cols, canvas.height / rows);
    }
  }
}
setInterval(refreshStatus, 2000);
setInterval(refreshHeatmap, 5000);
setInterval(refreshCharts, 3000);
refreshStatus();
refreshHeatmap();
refreshCharts();

document.querySelectorAll('.menu button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.menu button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const panel = btn.getAttribute('data-panel');
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`panel-${panel}`).classList.add('active');
  });
});

async function refreshCharts() {
  let data = null;
  try {
    const res = await fetch('/api/metrics');
    data = await res.json();
  } catch (err) {
    drawEmpty(document.getElementById('chartReward'), 'Metrics fetch failed');
    drawEmpty(document.getElementById('chartWin'), 'Metrics fetch failed');
    drawEmpty(document.getElementById('chartDelta'), 'Metrics fetch failed');
    drawEmpty(document.getElementById('chartRally'), 'Metrics fetch failed');
    return;
  }
  const series = data.series || [];
  document.getElementById('dataLine').textContent = `Data: ${series.length} rows`;
  if (!series.length) {
    drawEmpty(document.getElementById('chartReward'), 'No metrics yet');
    drawEmpty(document.getElementById('chartWin'), 'No metrics yet');
    drawEmpty(document.getElementById('chartDelta'), 'No metrics yet');
    drawEmpty(document.getElementById('chartRally'), 'No metrics yet');
  }
  const table = document.getElementById('metricsTable');
  table.innerHTML = '';
  const recent = series.slice(-15).reverse();
  for (const row of recent) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.cycle}</td><td>${row.model_id}</td><td>${row.avg_reward.toFixed(2)}</td><td>${row.delta_reward.toFixed(2)}</td><td>${row.win_rate.toFixed(2)}</td><td>${row.avg_rally_length.toFixed(1)}</td><td>${row.avg_return_rate.toFixed(2)}</td>`;
    table.appendChild(tr);
  }
  const byCycle = new Map();
  for (const row of series) {
    if (!byCycle.has(row.cycle)) byCycle.set(row.cycle, []);
    byCycle.get(row.cycle).push(row);
  }
  const cycles = Array.from(byCycle.keys()).sort((a,b)=>a-b);
  const rewards = cycles.map(c => {
    const rows = byCycle.get(c);
    const best = rows.reduce((acc, r) => Math.max(acc, r.avg_reward), -999);
    return best;
  });
  const wins = cycles.map(c => {
    const rows = byCycle.get(c);
    const best = rows.reduce((acc, r) => Math.max(acc, r.win_rate), 0);
    return best;
  });
  const deltas = cycles.map(c => {
    const rows = byCycle.get(c);
    const best = rows.reduce((acc, r) => Math.max(acc, r.delta_reward), -999);
    return best;
  });
  const rallies = cycles.map(c => {
    const rows = byCycle.get(c);
    const best = rows.reduce((acc, r) => Math.max(acc, r.avg_rally_length), 0);
    return best;
  });
  drawLineChart(document.getElementById('chartReward'), cycles, rewards, '#14f195');
  drawLineChart(document.getElementById('chartWin'), cycles, wins, '#f5a623');
  drawLineChart(document.getElementById('chartDelta'), cycles, deltas, '#7cc6ff');
  drawLineChart(document.getElementById('chartRally'), cycles, rallies, '#ff5f7a');
}

function drawLineChart(canvas, xs, ys, color) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!xs.length) return;
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 20;
  const w = canvas.width - pad*2;
  const h = canvas.height - pad*2;
  ctx.strokeStyle = '#243447';
  ctx.strokeRect(pad, pad, w, h);
  ctx.beginPath();
  xs.forEach((x, i) => {
    const nx = i / (xs.length - 1 || 1);
    const ny = (maxY === minY) ? 0.5 : (ys[i] - minY) / (maxY - minY);
    const px = pad + nx * w;
    const py = pad + (1 - ny) * h;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  // Draw points so single-cycle data is visible.
  xs.forEach((x, i) => {
    const nx = i / (xs.length - 1 || 1);
    const ny = (maxY === minY) ? 0.5 : (ys[i] - minY) / (maxY - minY);
    const px = pad + nx * w;
    const py = pad + (1 - ny) * h;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawEmpty(canvas, label) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = '#243447';
  ctx.strokeRect(20, 20, canvas.width - 40, canvas.height - 40);
  ctx.fillStyle = '#8aa3c5';
  ctx.font = '12px Segoe UI';
  ctx.fillText(label, 30, canvas.height / 2);
}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Pong training dashboard server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on.")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Dashboard running at http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
