import argparse
import csv
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse, parse_qs

import numpy as np
from stable_baselines3 import PPO

from train_pong_ppo import SB3PongEnv
from pong import simple_tracking_policy


ROOT = Path.cwd()


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
    if not model_path.exists():
        return []
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
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
    env.close()
    return heat.tolist()


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
        if parsed.path == "/api/heatmap":
            model_path = ROOT / "models" / "ppo_pong_custom_latest.zip"
            heat = _heatmap_from_model(model_path)
            self._send(200, json.dumps({"heatmap": heat}).encode("utf-8"), "application/json")
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
    .wrap { padding: 20px; max-width: 1200px; margin: 0 auto; }
    h1 { letter-spacing: 2px; text-transform: uppercase; font-size: 20px; color: var(--accent); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .card { background: var(--panel); border: 1px solid #1f2a36; border-radius: 12px; padding: 16px; }
    .stat { font-size: 28px; color: var(--accent2); }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    canvas { width: 100%; height: auto; background: #0f141b; border-radius: 8px; }
    video { width: 100%; border-radius: 8px; border: 1px solid #203040; }
    .label { font-size: 12px; text-transform: uppercase; color: #8aa3c5; letter-spacing: 1px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Live Pong Training Dashboard</h1>
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
      </div>
    </div>
    <div class="card" style="margin-top:16px;">
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
<script>
async function refreshStatus() {
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
  if (combined) document.getElementById('vidCombined').src = '/file?path=' + encodeURIComponent(combined);
  if (evalVid) document.getElementById('vidEval').src = '/file?path=' + encodeURIComponent(evalVid);
}
async function refreshHeatmap() {
  const res = await fetch('/api/heatmap');
  const data = await res.json();
  const heat = data.heatmap || [];
  const canvas = document.getElementById('heatmap');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!heat.length) return;
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
refreshStatus();
refreshHeatmap();
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
