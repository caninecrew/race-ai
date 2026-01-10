import argparse
import csv
import json
import math
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs

import numpy as np
from stable_baselines3 import PPO

from train_pong_ppo import SB3PongEnv
from pong import simple_tracking_policy


ROOT = Path(__file__).resolve().parent
_HEATMAP_CACHE: Optional[dict] = None
_HEATMAP_TS = 0.0
_ANNOTATIONS_FILE = ROOT / "logs" / "annotations.json"


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


def _read_annotations() -> Dict[str, Any]:
    if not _ANNOTATIONS_FILE.exists():
        return {"notes": []}
    try:
        return json.loads(_ANNOTATIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"notes": []}


def _write_annotations(payload: Dict[str, Any]) -> None:
    _ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ANNOTATIONS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _latest_report_info(log_dir: Path) -> Dict[str, Any]:
    report_path = _latest_report(log_dir)
    if not report_path or not report_path.exists():
        return {}
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    return value


def _list_reports(log_dir: Path) -> List[Dict[str, Any]]:
    reports = sorted(log_dir.glob("run_report_*.json"))
    items: List[Dict[str, Any]] = []
    for path in reports:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = data.get("summary", {})
        config = data.get("config", {})
        items.append(
            {
                "path": str(path.relative_to(ROOT)),
                "run_timestamp": summary.get("run_timestamp"),
                "profile": config.get("profile"),
                "summary": summary,
                "config": config,
            }
        )
    return items


def _heatmap_from_model(model_path: Path, steps: int = 1500, bins: int = 40) -> dict:
    global _HEATMAP_CACHE, _HEATMAP_TS
    now = time.time()
    if _HEATMAP_CACHE is not None and (now - _HEATMAP_TS) < 10:
        return _HEATMAP_CACHE
    if not model_path.exists():
        return {}
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        obs, _ = env.reset()
        heat_ball = np.zeros((bins, bins), dtype=np.int32)
        heat_left = np.zeros((bins, bins), dtype=np.int32)
        heat_right = np.zeros((bins, bins), dtype=np.int32)
        heat_hits = np.zeros((bins, bins), dtype=np.int32)
        heat_scores = np.zeros((bins, bins), dtype=np.int32)
        last_left = 0
        last_right = 0
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            bx, by = obs[0], obs[1]
            ly, ry = obs[4], obs[5]
            x = min(bins - 1, max(0, int(bx * bins)))
            y = min(bins - 1, max(0, int(by * bins)))
            heat_ball[y, x] += 1
            ly_idx = min(bins - 1, max(0, int(ly * bins)))
            ry_idx = min(bins - 1, max(0, int(ry * bins)))
            heat_left[ly_idx, 1] += 1
            heat_right[ry_idx, bins - 2] += 1
            if reward > 0 and reward < 0.5:
                heat_hits[y, x] += 1
            left_score = info.get("left_score", last_left)
            right_score = info.get("right_score", last_right)
            if left_score != last_left or right_score != last_right:
                heat_scores[y, x] += 1
                last_left, last_right = left_score, right_score
            if terminated or truncated:
                obs, _ = env.reset()
        _HEATMAP_CACHE = {
            "ball": heat_ball.tolist(),
            "paddles": (heat_left + heat_right).tolist(),
            "hits": heat_hits.tolist(),
            "scores": heat_scores.tolist(),
        }
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
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path else {}
            metrics_csv = ROOT / "logs" / "metrics.csv"
            last_update = None
            if metrics_csv.exists():
                last_update = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metrics_csv.stat().st_mtime))
            payload = {
                "metrics": metrics,
                "report": report,
                "paths": {
                    "metrics_csv": str(metrics_csv),
                    "latest_report": str(report_path) if report_path else "",
                },
                "last_update": last_update,
            }
            self._send(200, json.dumps(_sanitize_json(payload), allow_nan=False).encode("utf-8"), "application/json")
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
                                    "avg_reward_ci": float(row.get("avg_reward_ci", 0.0)),
                                    "win_rate": float(row.get("win_rate", 0.0)),
                                    "avg_rally_length": float(row.get("avg_rally_length", 0.0)),
                                    "avg_return_rate": float(row.get("avg_return_rate", 0.0)),
                                    "avg_return_rate_ci": float(row.get("avg_return_rate_ci", 0.0)),
                                    "delta_reward": float(row.get("delta_reward", 0.0)) if row.get("delta_reward") else 0.0,
                                    "timestamp": row.get("timestamp", ""),
                                    "run_id": row.get("run_timestamp", ""),
                                }
                            )
                        except Exception:
                            continue
            else:
                err = "metrics.csv not found"
            self._send(
                200,
                json.dumps(
                    {
                        "series": series,
                        "error": err,
                        "paths": {"metrics_csv": str(metrics_csv)},
                    }
                ).encode("utf-8"),
                "application/json",
            )
            return
        if parsed.path == "/api/reports":
            reports = _list_reports(ROOT / "logs")
            self._send(
                200,
                json.dumps(_sanitize_json({"reports": reports}), allow_nan=False).encode("utf-8"),
                "application/json",
            )
            return
        if parsed.path == "/api/annotations":
            payload = _read_annotations()
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
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

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotations":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self._send(400, b"invalid json", "text/plain")
            return
        existing = _read_annotations()
        notes = existing.get("notes", [])
        if not isinstance(notes, list):
            notes = []
        if payload:
            payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            notes.append(payload)
        _write_annotations({"notes": notes})
        self._send(200, json.dumps({"ok": True, "notes": notes}).encode("utf-8"), "application/json")


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
    .layout { display: grid; grid-template-columns: 240px 1fr; min-height: 100vh; }
    .sidebar { background: #0f141b; padding: 16px; border-right: 1px solid #1f2a36; }
    .sidebar h2 { font-size: 12px; letter-spacing: 2px; color: #8aa3c5; text-transform: uppercase; }
    .menu button { width: 100%; margin: 6px 0; padding: 10px; background: #121820; color: var(--text); border: 1px solid #1f2a36; border-radius: 8px; cursor: pointer; }
    .menu button.active { border-color: var(--accent); color: var(--accent); }
    .wrap { padding: 20px; max-width: 1400px; margin: 0 auto; }
    h1 { letter-spacing: 2px; text-transform: uppercase; font-size: 20px; color: var(--accent); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
    .card { background: var(--panel); border: 1px solid #1f2a36; border-radius: 12px; padding: 16px; }
    .stat { font-size: 28px; color: var(--accent2); }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .kpi-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
    canvas { width: 100%; height: auto; background: #0f141b; border-radius: 8px; }
    video { width: 100%; border-radius: 8px; border: 1px solid #203040; }
    .label { font-size: 12px; text-transform: uppercase; color: #8aa3c5; letter-spacing: 1px; }
    .panel { display: none; }
    .panel.active { display: block; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 6px 8px; border-bottom: 1px solid #1f2a36; text-align: left; }
    .legend { display: flex; align-items: center; gap: 8px; font-size: 11px; color: #8aa3c5; }
    .legend-bar { width: 120px; height: 10px; border-radius: 6px; background: linear-gradient(90deg, rgba(20,241,149,0.0), rgba(20,241,149,1.0)); border: 1px solid #1f2a36; }
    .tiny { font-size: 11px; color: #8aa3c5; }
    .pill { padding: 4px 8px; border-radius: 999px; border: 1px solid #1f2a36; background: #0f141b; font-size: 11px; color: var(--text); cursor: pointer; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .alert { color: #ff7d7d; }
    .ok { color: #14f195; }
    .spark { width: 120px; height: 36px; }
    .timeline { display: grid; gap: 8px; }
    .timeline-item { display: flex; gap: 8px; align-items: center; cursor: pointer; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
    .controls select, .controls input { background: #0f141b; border: 1px solid #1f2a36; color: var(--text); padding: 6px 8px; border-radius: 6px; }
  </style>
</head>
<body>
    <div class="layout">
      <div class="sidebar">
        <h2>Dashboard</h2>
        <div class="menu">
          <button data-panel="overview" class="active">Overview</button>
          <button data-panel="charts">Charts</button>
          <button data-panel="insights">Insights</button>
          <button data-panel="cohorts">Cohorts</button>
          <button data-panel="videos">Videos</button>
          <button data-panel="table">Recent Metrics</button>
        </div>
        <div class="label" style="margin-top:10px;">Run Filter</div>
        <div class="tiny">Choose a past run to filter charts and tables.</div>
        <select id="profileSelect" style="width:100%; margin-top:6px;"></select>
        <div class="label" id="statusLine" style="margin-top:12px;">Status: --</div>
        <div class="tiny">Shows if the dashboard is receiving live data.</div>
        <div class="label" id="dataLine">Data: --</div>
        <div class="tiny">How many rows are currently loaded.</div>
        <div class="label" id="freshnessLine">Last update: --</div>
      </div>
    <div class="wrap">
      <h1>Live Pong Training Dashboard</h1>
      <div class="tiny" style="margin-bottom:12px;">
        This page tracks how the AI is learning to play Pong. Each "cycle" is one round of training and testing.
        If a number is bigger, that usually means better play (unless noted).
      </div>
      <div id="panel-overview" class="panel active">
        <div class="kpi-strip">
          <div class="card">
            <div class="label">Best Reward</div>
            <div id="kpiBestReward" class="stat">--</div>
            <div class="tiny" id="kpiBestRewardCi">--</div>
            <div class="tiny">Highest average score from the latest cycle (bigger is better).</div>
          </div>
          <div class="card">
            <div class="label">Avg Reward</div>
            <div id="kpiAvgReward" class="stat">--</div>
            <div class="tiny" id="kpiAvgRewardDelta">--</div>
            <div class="tiny">Average score across all models in the latest cycle.</div>
            <div class="tiny">Reward is the game score: higher means the agent wins more points.</div>
          </div>
          <div class="card">
            <div class="label">Win Rate</div>
            <div id="kpiWinRate" class="stat">--</div>
            <div class="tiny" id="kpiWinRateDelta">--</div>
            <div class="tiny">Fraction of games the AI wins (1.00 = 100%).</div>
            <div class="tiny">Example: 0.25 means 25% of games won.</div>
          </div>
          <div class="card">
            <div class="label">Return Rate</div>
            <div id="kpiReturnRate" class="stat">--</div>
            <div class="tiny" id="kpiReturnRateCi">--</div>
            <div class="tiny">How often the AI hits the ball back.</div>
            <div class="tiny">1.00 = returns every shot, 0.00 = misses every shot.</div>
          </div>
          <div class="card">
            <div class="label">Rally Length</div>
            <div id="kpiRally" class="stat">--</div>
            <div class="tiny" id="kpiRallyDelta">--</div>
            <div class="tiny">Average number of hits before a point ends.</div>
            <div class="tiny">Longer rallies usually mean better defense.</div>
          </div>
        </div>
        <div class="grid" style="margin-top:16px;">
          <div class="card">
            <div class="label">Best Model</div>
            <div id="bestModel" class="stat">--</div>
            <div class="tiny">Model with the highest reward so far.</div>
            <div class="label" style="margin-top:8px;">Avg Reward</div>
            <div id="bestReward">--</div>
            <div class="label">Win Rate</div>
            <div id="bestWin">--</div>
            <div class="tiny">This is the current champion model.</div>
          </div>
          <div class="card">
            <div class="label">Latest Run</div>
            <div id="runId">--</div>
            <div class="tiny">Most recent training session.</div>
            <div class="label" style="margin-top:8px;">Stop Reason</div>
            <div id="stopReason">--</div>
            <div class="label" style="margin-top:8px;">Data Sources</div>
            <div id="dataSources" class="tiny">--</div>
            <div class="tiny">Useful if data looks stale or missing.</div>
          </div>
          <div class="card">
            <div class="label">Heatmaps</div>
            <div class="tiny">Where the ball and paddles spend time. Brighter means more activity.</div>
            <div class="menu" style="margin-bottom:8px;">
              <button data-heat="ball" class="active">Ball Density</button>
              <button data-heat="paddles">Paddle Density</button>
              <button data-heat="hits">Hit Hotspots</button>
              <button data-heat="scores">Score Zones</button>
            </div>
            <div class="controls">
              <label class="tiny"><input type="checkbox" id="heatLog"/> Log scale</label>
              <label class="tiny"><input type="checkbox" id="heatOverlay" checked/> Court overlay</label>
            </div>
            <canvas id="heatmap" width="400" height="400"></canvas>
            <div class="legend" style="margin-top:8px;">
              <div class="legend-bar"></div>
              <span id="heatLegend">0 -> --</span>
            </div>
            <div id="heatmapNote" class="label" style="margin-top:6px;">--</div>
          </div>
          <div class="card">
            <div class="label">Quality Gates</div>
            <div class="tiny">Set minimum targets to see if the latest model passes.</div>
            <div class="tiny">Use this to decide if the model is "good enough" to keep.</div>
            <div class="controls">
              <input id="gateReward" type="number" step="0.1" placeholder="Reward > X"/>
              <input id="gateWin" type="number" step="0.01" placeholder="Win rate > Y"/>
            </div>
            <div id="gateStatus" class="tiny">--</div>
          </div>
        </div>
      </div>
      <div id="panel-charts" class="panel">
        <div class="card">
          <div class="label">Training Charts</div>
          <div class="tiny">Trends over time. Use the selector to compare best vs average runs.</div>
          <div class="tiny">If the lines rise over cycles, the AI is improving.</div>
          <div class="controls">
            <select id="compareMode">
              <option value="best">Best</option>
              <option value="avg" selected>Avg</option>
              <option value="median">Median</option>
            </select>
            <label class="tiny"><input type="checkbox" id="rollingToggle"/> Rolling avg</label>
            <input id="rollingWindow" type="number" min="2" max="10" value="3" style="width:64px;"/>
          </div>
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
        <div class="grid" style="margin-top:16px;">
          <div class="card">
            <div class="label">Cycle Distribution</div>
            <div class="tiny">How spread out the scores are within one cycle.</div>
            <div class="tiny">Wide spread = some models are much better than others.</div>
            <div class="controls">
              <select id="cycleSelect"></select>
            </div>
            <div class="split">
              <div>
                <div class="label">Reward Histogram</div>
                <canvas id="histReward" width="400" height="200"></canvas>
              </div>
              <div>
                <div class="label">Rally Length Histogram</div>
                <canvas id="histRally" width="400" height="200"></canvas>
              </div>
            </div>
          </div>
          <div class="card">
            <div class="label">Correlations</div>
            <div class="tiny">See if longer rallies or better returns link to higher reward.</div>
            <div class="tiny">Upward trend = that metric helps the reward.</div>
            <div class="split">
              <div>
                <div class="label">Reward vs Rally Length</div>
                <canvas id="scatterRally" width="400" height="200"></canvas>
              </div>
              <div>
                <div class="label">Reward vs Return Rate</div>
                <canvas id="scatterReturn" width="400" height="200"></canvas>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div id="panel-insights" class="panel">
        <div class="grid">
          <div class="card">
            <div class="label">Model Leaderboard</div>
            <div class="tiny">Top models by average reward.</div>
            <div class="tiny">Higher rank means stronger overall performance.</div>
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model</th>
                  <th>Avg Reward</th>
                  <th>Trend</th>
                </tr>
              </thead>
              <tbody id="leaderboard"></tbody>
            </table>
          </div>
          <div class="card">
            <div class="label">Run Timeline</div>
            <div class="tiny">Click a cycle to see a quick summary.</div>
            <div class="tiny">Use this to spot when performance changed.</div>
            <div id="timeline" class="timeline"></div>
            <div id="timelineDetail" class="tiny" style="margin-top:8px;">--</div>
          </div>
          <div class="card">
            <div class="label">Video Insights</div>
            <div class="tiny">Highlights from the longest rallies.</div>
            <div class="tiny">Long rallies often show strong defense.</div>
            <div id="videoInsights" class="tiny">--</div>
          </div>
          <div class="card">
            <div class="label">Annotations</div>
            <div class="tiny">Leave short observations as you watch training.</div>
            <div class="tiny">Example: "Cycle 3 learns to track high balls."</div>
            <div class="controls">
              <input id="noteCycle" type="number" placeholder="Cycle #"/>
              <input id="noteText" type="text" placeholder="What stood out?"/>
              <button id="noteSave" class="pill">Save</button>
            </div>
            <div id="notesList" class="tiny">--</div>
          </div>
        </div>
      </div>
      <div id="panel-cohorts" class="panel">
        <div class="card">
          <div class="label">Cohort Comparison</div>
          <div class="tiny">Pick two runs to compare average performance.</div>
          <div class="tiny">If numbers are missing, that run has no tagged metrics yet.</div>
          <div id="cohortStatus" class="tiny">Runs available: --</div>
          <div class="controls">
            <select id="cohortA">
              <option value="all">All Runs</option>
            </select>
            <select id="cohortB">
              <option value="all">All Runs</option>
            </select>
          </div>
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Run A</th>
                <th>Run B</th>
              </tr>
            </thead>
            <tbody id="cohortTable"></tbody>
          </table>
        </div>
      </div>
      <div id="panel-videos" class="panel">
        <div class="card">
          <div class="label">Comparative Split</div>
          <div class="tiny">Side-by-side videos from the latest evaluation.</div>
          <div class="tiny">Left is the latest combined clip; right is a longer review.</div>
          <div class="split">
            <div>
              <div class="label">Latest Combined</div>
              <div class="tiny">Short clip of the newest run.</div>
              <video id="vidCombined" controls muted></video>
            </div>
            <div>
              <div class="label">Extended Eval</div>
              <div class="tiny">Longer clip for deeper review.</div>
              <video id="vidEval" controls muted></video>
            </div>
          </div>
        </div>
      </div>
      <div id="panel-table" class="panel">
        <div class="card">
          <div class="label">Recent Metrics</div>
          <div class="tiny">Raw numbers per cycle. Scroll to see older entries.</div>
          <div class="tiny">If a row looks odd, it may be from a different run.</div>
          <div class="controls">
            <button id="downloadCsv" class="pill">Download CSV</button>
            <select id="snapshotSelect"></select>
            <button id="snapshotBtn" class="pill">Snapshot PNG</button>
          </div>
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
let heatmapMode = "ball";
let heatmapCache = {};
let metricsSeries = [];
let reportCache = [];
let annotationsCache = [];
let selectedRun = "all";
let compareMode = "avg";
let rollingEnabled = false;
let activePanel = "overview";
let lastReportKey = "";
const apiBase = (() => {
  const params = new URLSearchParams(window.location.search);
  const override = params.get('api') || '';
  if (override) return override.replace(/\\/$/, '');
  const proto = window.location.protocol;
  if (proto === 'http:' || proto === 'https:') return '';
  return 'http://127.0.0.1:8000';
})();

function apiFetch(path, options) {
  return fetch(`${apiBase}${path}`, options);
}

async function refreshStatus() {
  try {
    const res = await apiFetch('/api/status');
    const data = await res.json();
    const metrics = data.metrics || {};
    const report = (data.report || {}).summary || {};
    document.getElementById('bestModel').textContent = metrics.model_id || 'n/a';
    document.getElementById('bestReward').textContent = metrics.avg_reward || '--';
    document.getElementById('bestWin').textContent = metrics.win_rate || '--';
    document.getElementById('runId').textContent = report.run_timestamp || '--';
    document.getElementById('stopReason').textContent = report.stop_reason || '--';
    const sources = data.paths || {};
    document.getElementById('dataSources').textContent = `metrics: ${sources.metrics_csv || '--'} | report: ${sources.latest_report || '--'}`;
    document.getElementById('freshnessLine').textContent = `Last update: ${data.last_update || '--'}`;
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
    const res = await apiFetch('/api/heatmap');
    if (!res.ok) {
      throw new Error(`heatmap HTTP ${res.status}`);
    }
    data = await res.json();
  } catch (err) {
    document.getElementById('heatmapNote').textContent = `Heatmap fetch failed: ${err}`;
    drawEmpty(document.getElementById('heatmap'), 'Heatmap error');
    return;
  }
  if (data.error) {
    document.getElementById('heatmapNote').textContent = `Heatmap error: ${data.error}`;
    drawEmpty(document.getElementById('heatmap'), 'Heatmap error');
    return;
  }
  heatmapCache = data.heatmap || {};
  const heat = heatmapCache[heatmapMode] || [];
  const canvas = document.getElementById('heatmap');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!heat.length) {
    document.getElementById('heatmapNote').textContent = 'No heatmap yet (train a model first).';
    drawEmpty(canvas, 'No heatmap data');
    return;
  }
  document.getElementById('heatmapNote').textContent = `Live ${heatmapMode} density from latest model.`;
  const rows = heat.length;
  const cols = heat[0].length;
  const max = Math.max(...heat.flat());
  const logScale = document.getElementById('heatLog').checked;
  for (let y=0; y<rows; y++) {
    for (let x=0; x<cols; x++) {
      const raw = heat[y][x];
      const scaled = logScale ? Math.log1p(raw) / Math.log1p(max || 1) : raw / (max || 1);
      ctx.fillStyle = `rgba(20,241,149,${scaled})`;
      ctx.fillRect(x * canvas.width / cols, y * canvas.height / rows, canvas.width / cols, canvas.height / rows);
    }
  }
  if (document.getElementById('heatOverlay').checked) {
    drawHeatOverlay(canvas);
  }
  document.getElementById('heatLegend').textContent = `0 -> ${max}`;
}
setInterval(refreshStatus, 2000);
setInterval(refreshHeatmap, 5000);
setInterval(refreshCharts, 3000);
setInterval(refreshReports, 6000);
setInterval(refreshAnnotations, 7000);
refreshStatus();
refreshHeatmap();
refreshCharts();
refreshReports();
refreshAnnotations();

document.querySelectorAll('.sidebar .menu button[data-panel]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.sidebar .menu button[data-panel]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const panel = btn.getAttribute('data-panel');
    activePanel = panel || 'overview';
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    const target = document.getElementById(`panel-${panel}`);
    if (target) target.classList.add('active');
    if (panel === 'cohorts') {
      refreshReports();
    }
  });
});

document.querySelectorAll('.card .menu button[data-heat]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.card .menu button[data-heat]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    heatmapMode = btn.getAttribute('data-heat');
    refreshHeatmap();
  });
});

async function refreshCharts() {
  let data = null;
  try {
    const res = await apiFetch('/api/metrics');
    if (!res.ok) {
      throw new Error(`metrics HTTP ${res.status}`);
    }
    data = await res.json();
  } catch (err) {
    drawEmpty(document.getElementById('chartReward'), 'Metrics fetch failed');
    drawEmpty(document.getElementById('chartWin'), 'Metrics fetch failed');
    drawEmpty(document.getElementById('chartDelta'), 'Metrics fetch failed');
    drawEmpty(document.getElementById('chartRally'), 'Metrics fetch failed');
    document.getElementById('dataLine').textContent = `Data: error (${err})`;
    return;
  }
  if (data.error) {
    drawEmpty(document.getElementById('chartReward'), 'Metrics error');
    drawEmpty(document.getElementById('chartWin'), 'Metrics error');
    drawEmpty(document.getElementById('chartDelta'), 'Metrics error');
    drawEmpty(document.getElementById('chartRally'), 'Metrics error');
    document.getElementById('dataLine').textContent = `Data: error (${data.error})`;
    return;
  }
  try {
    metricsSeries = data.series || [];
    const series = filterByRun(metricsSeries, selectedRun);
    document.getElementById('dataLine').textContent = `Data: ${series.length} rows`;
    if (!series.length) {
      drawEmpty(document.getElementById('chartReward'), 'No metrics yet');
    drawEmpty(document.getElementById('chartWin'), 'No metrics yet');
    drawEmpty(document.getElementById('chartDelta'), 'No metrics yet');
    drawEmpty(document.getElementById('chartRally'), 'No metrics yet');
  }
  updateKpis(series);
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
  updateCycleSelectors(byCycle);
  const cycles = Array.from(byCycle.keys()).sort((a,b)=>a-b);
  if (cycles.length === 1) {
    const rows = byCycle.get(cycles[0]);
    const rewardVals = rows.map(r => r.avg_reward);
    const winVals = rows.map(r => r.win_rate);
    const deltaVals = rows.map(r => r.delta_reward);
    const rallyVals = rows.map(r => r.avg_rally_length);
    const minMaxAvg = (vals) => {
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      const avg = vals.reduce((a,b)=>a+b,0) / vals.length;
      return {min, max, avg};
    };
    const rewardStats = minMaxAvg(rewardVals);
    const winStats = minMaxAvg(winVals);
    const deltaStats = minMaxAvg(deltaVals);
    const rallyStats = minMaxAvg(rallyVals);
    drawRangeChart(document.getElementById('chartReward'), rewardStats, '#14f195', 'Avg Reward (single cycle)');
    drawRangeChart(document.getElementById('chartWin'), winStats, '#f5a623', 'Win Rate (single cycle)', 0, 1);
    drawRangeChart(document.getElementById('chartDelta'), deltaStats, '#7cc6ff', 'Delta Reward (single cycle)');
    drawRangeChart(document.getElementById('chartRally'), rallyStats, '#ff5f7a', 'Avg Rally Length (single cycle)');
    return;
  }
  const pickByMode = (rows, key) => {
    if (compareMode === 'best') {
      return rows.reduce((acc, r) => Math.max(acc, r[key]), -999);
    }
    if (compareMode === 'median') {
      const vals = rows.map(r => r[key]).sort((a,b)=>a-b);
      const mid = Math.floor(vals.length / 2);
      return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
    }
    return rows.reduce((acc, r) => acc + r[key], 0) / rows.length;
  };
  const applyRolling = (vals) => {
    if (!rollingEnabled) return vals;
    const window = Math.max(2, parseInt(document.getElementById('rollingWindow').value || '3', 10));
    return vals.map((_, idx) => {
      const start = Math.max(0, idx - window + 1);
      const slice = vals.slice(start, idx + 1);
      return slice.reduce((a,b)=>a+b,0) / slice.length;
    });
  };
  const rewards = applyRolling(cycles.map(c => pickByMode(byCycle.get(c), 'avg_reward')));
  const wins = applyRolling(cycles.map(c => pickByMode(byCycle.get(c), 'win_rate')));
  const deltas = applyRolling(cycles.map(c => pickByMode(byCycle.get(c), 'delta_reward')));
  const rallies = applyRolling(cycles.map(c => pickByMode(byCycle.get(c), 'avg_rally_length')));
  drawLineChart(document.getElementById('chartReward'), cycles, [
    { label: compareMode, color: '#14f195', values: rewards },
  ], 'Avg Reward');
  drawLineChart(document.getElementById('chartWin'), cycles, [
    { label: compareMode, color: '#f5a623', values: wins },
  ], 'Win Rate');
  drawLineChart(document.getElementById('chartDelta'), cycles, [
    { label: compareMode, color: '#7cc6ff', values: deltas },
  ], 'Delta Reward');
  drawLineChart(document.getElementById('chartRally'), cycles, [
    { label: compareMode, color: '#ff5f7a', values: rallies },
  ], 'Avg Rally Length');
    drawDistributions(byCycle);
    drawCorrelations(series);
    updateLeaderboard(series);
    updateQualityGates(series);
    updateVideoInsights(series);
  } catch (err) {
    drawEmpty(document.getElementById('chartReward'), 'Training charts failed');
    drawEmpty(document.getElementById('chartWin'), 'Training charts failed');
    drawEmpty(document.getElementById('chartDelta'), 'Training charts failed');
    drawEmpty(document.getElementById('chartRally'), 'Training charts failed');
    document.getElementById('dataLine').textContent = `Data: chart error (${err})`;
  }
}

function drawLineChart(canvas, xs, series, title) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!xs.length) return;
  const allValues = series.flatMap(s => s.values);
  const minY = Math.min(...allValues);
  const maxY = Math.max(...allValues);
  const pad = 28;
  const w = canvas.width - pad*2;
  const h = canvas.height - pad*2;
  ctx.strokeStyle = '#243447';
  ctx.strokeRect(pad, pad, w, h);
  ctx.fillStyle = '#8aa3c5';
  ctx.font = '11px Segoe UI';
  ctx.fillText(title, pad, 14);
  const minLabel = minY.toFixed(2);
  const maxLabel = maxY.toFixed(2);
  ctx.fillText(maxLabel, 4, pad + 6);
  ctx.fillText(minLabel, 4, pad + h);
  series.forEach(s => {
    ctx.beginPath();
    s.values.forEach((y, i) => {
      const nx = i / (xs.length - 1 || 1);
      const ny = (maxY === minY) ? 0.5 : (y - minY) / (maxY - minY);
      const px = pad + nx * w;
      const py = pad + (1 - ny) * h;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    s.values.forEach((y, i) => {
      const nx = i / (xs.length - 1 || 1);
      const ny = (maxY === minY) ? 0.5 : (y - minY) / (maxY - minY);
      const px = pad + nx * w;
      const py = pad + (1 - ny) * h;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  if (series.length > 1) {
    const legendX = pad + 6;
    let legendY = pad + h + 16;
    series.forEach(s => {
      ctx.fillStyle = s.color;
      ctx.fillRect(legendX, legendY - 8, 10, 10);
      ctx.fillStyle = '#8aa3c5';
      ctx.fillText(s.label, legendX + 14, legendY);
      legendY += 14;
    });
  }
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

function drawRangeChart(canvas, stats, color, title, minOverride, maxOverride) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const pad = 28;
  const w = canvas.width - pad*2;
  const h = canvas.height - pad*2;
  const minY = (minOverride !== undefined) ? minOverride : stats.min;
  const maxY = (maxOverride !== undefined) ? maxOverride : stats.max;
  ctx.strokeStyle = '#243447';
  ctx.strokeRect(pad, pad, w, h);
  ctx.fillStyle = '#8aa3c5';
  ctx.font = '11px Segoe UI';
  ctx.fillText(title, pad, 14);
  ctx.fillText(maxY.toFixed(2), 4, pad + 6);
  ctx.fillText(minY.toFixed(2), 4, pad + h);
  const midY = pad + h / 2;
  const rangeMinX = pad + ((stats.min - minY) / ((maxY - minY) || 1)) * w;
  const rangeMaxX = pad + ((stats.max - minY) / ((maxY - minY) || 1)) * w;
  const avgX = pad + ((stats.avg - minY) / ((maxY - minY) || 1)) * w;
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(rangeMinX, midY);
  ctx.lineTo(rangeMaxX, midY);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(avgX, midY, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#8aa3c5';
  ctx.fillText(`min ${stats.min.toFixed(2)}`, pad, midY + 20);
  ctx.fillText(`avg ${stats.avg.toFixed(2)}`, pad + 120, midY + 20);
  ctx.fillText(`max ${stats.max.toFixed(2)}`, pad + 230, midY + 20);
}

function drawHeatOverlay(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth = 2;
  ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20);
  ctx.beginPath();
  ctx.moveTo(canvas.width / 2, 10);
  ctx.lineTo(canvas.width / 2, canvas.height - 10);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(245,166,35,0.4)';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(22, canvas.height * 0.2);
  ctx.lineTo(22, canvas.height * 0.8);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(canvas.width - 22, canvas.height * 0.2);
  ctx.lineTo(canvas.width - 22, canvas.height * 0.8);
  ctx.stroke();
  ctx.restore();
}

function updateKpis(series) {
  if (!series.length) return;
  const byCycle = new Map();
  for (const row of series) {
    if (!byCycle.has(row.cycle)) byCycle.set(row.cycle, []);
    byCycle.get(row.cycle).push(row);
  }
  const cycles = Array.from(byCycle.keys()).sort((a,b)=>a-b);
  const latest = byCycle.get(cycles[cycles.length - 1]) || [];
  const prev = cycles.length > 1 ? (byCycle.get(cycles[cycles.length - 2]) || []) : [];
  const bestReward = Math.max(...latest.map(r => r.avg_reward));
  const bestRewardCi = latest.find(r => r.avg_reward === bestReward)?.avg_reward_ci || 0;
  const avgReward = latest.reduce((a,b)=>a+b.avg_reward,0) / latest.length;
  const avgWin = latest.reduce((a,b)=>a+b.win_rate,0) / latest.length;
  const avgReturn = latest.reduce((a,b)=>a+b.avg_return_rate,0) / latest.length;
  const avgReturnCi = latest.reduce((a,b)=>a+b.avg_return_rate_ci,0) / latest.length;
  const avgRally = latest.reduce((a,b)=>a+b.avg_rally_length,0) / latest.length;
  const prevAvgReward = prev.length ? prev.reduce((a,b)=>a+b.avg_reward,0) / prev.length : avgReward;
  const prevAvgWin = prev.length ? prev.reduce((a,b)=>a+b.win_rate,0) / prev.length : avgWin;
  const prevAvgRally = prev.length ? prev.reduce((a,b)=>a+b.avg_rally_length,0) / prev.length : avgRally;
  const deltaReward = avgReward - prevAvgReward;
  const deltaWin = avgWin - prevAvgWin;
  const deltaRally = avgRally - prevAvgRally;
  document.getElementById('kpiBestReward').textContent = bestReward.toFixed(2);
  document.getElementById('kpiBestRewardCi').textContent = `CI +/-${bestRewardCi.toFixed(2)}`;
  document.getElementById('kpiAvgReward').textContent = avgReward.toFixed(2);
  document.getElementById('kpiAvgRewardDelta').textContent = `Delta ${deltaReward.toFixed(2)} vs last cycle`;
  document.getElementById('kpiWinRate').textContent = avgWin.toFixed(2);
  document.getElementById('kpiWinRateDelta').textContent = `Delta ${deltaWin.toFixed(2)} vs last cycle`;
  document.getElementById('kpiReturnRate').textContent = avgReturn.toFixed(2);
  document.getElementById('kpiReturnRateCi').textContent = `CI +/-${avgReturnCi.toFixed(2)}`;
  document.getElementById('kpiRally').textContent = avgRally.toFixed(1);
  document.getElementById('kpiRallyDelta').textContent = `Delta ${deltaRally.toFixed(1)} vs last cycle`;
}

function updateCycleSelectors(byCycle) {
  const cycleSelect = document.getElementById('cycleSelect');
  const cycles = Array.from(byCycle.keys()).sort((a,b)=>a-b);
  if (!cycles.length) return;
  const previous = cycleSelect.value;
  cycleSelect.innerHTML = '';
  for (const c of cycles) {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = `Cycle ${c}`;
    cycleSelect.appendChild(opt);
  }
  cycleSelect.value = cycles.includes(parseInt(previous, 10)) ? previous : cycles[cycles.length - 1];
}

function drawDistributions(byCycle) {
  const cycleSelect = document.getElementById('cycleSelect');
  if (!cycleSelect.value) return;
  const cycle = parseInt(cycleSelect.value, 10);
  const rows = byCycle.get(cycle) || [];
  drawHistogram(document.getElementById('histReward'), rows.map(r => r.avg_reward), '#14f195', 'Reward');
  drawHistogram(document.getElementById('histRally'), rows.map(r => r.avg_rally_length), '#ff5f7a', 'Rally');
}

function drawHistogram(canvas, values, color, title) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!values.length) {
    drawEmpty(canvas, 'No data');
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const bins = 8;
  const bucket = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  values.forEach(v => {
    const idx = Math.min(bins - 1, Math.floor((v - min) / bucket));
    counts[idx] += 1;
  });
  const pad = 24;
  const w = canvas.width - pad*2;
  const h = canvas.height - pad*2;
  const maxCount = Math.max(...counts);
  ctx.fillStyle = '#8aa3c5';
  ctx.font = '11px Segoe UI';
  ctx.fillText(title, pad, 14);
  counts.forEach((count, i) => {
    const barW = w / bins - 6;
    const barH = (count / (maxCount || 1)) * h;
    const x = pad + i * (w / bins) + 3;
    const y = pad + (h - barH);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, barW, barH);
  });
}

function drawCorrelations(series) {
  drawScatter(document.getElementById('scatterRally'), series.map(r => [r.avg_rally_length, r.avg_reward]), '#ff5f7a', 'Rally', 'Reward');
  drawScatter(document.getElementById('scatterReturn'), series.map(r => [r.avg_return_rate, r.avg_reward]), '#2aa4ff', 'Return', 'Reward');
}

function drawScatter(canvas, points, color, xLabel, yLabel) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!points.length) {
    drawEmpty(canvas, 'No data');
    return;
  }
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 28;
  const w = canvas.width - pad*2;
  const h = canvas.height - pad*2;
  ctx.strokeStyle = '#243447';
  ctx.strokeRect(pad, pad, w, h);
  ctx.fillStyle = '#8aa3c5';
  ctx.font = '11px Segoe UI';
  ctx.fillText(`${xLabel} vs ${yLabel}`, pad, 14);
  points.forEach(([x, y]) => {
    const nx = (x - minX) / ((maxX - minX) || 1);
    const ny = (y - minY) / ((maxY - minY) || 1);
    const px = pad + nx * w;
    const py = pad + (1 - ny) * h;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function updateLeaderboard(series) {
  const byModel = new Map();
  series.forEach(row => {
    if (!byModel.has(row.model_id)) byModel.set(row.model_id, []);
    byModel.get(row.model_id).push(row);
  });
  const rows = Array.from(byModel.entries()).map(([model, vals]) => {
    const avg = vals.reduce((a,b)=>a+b.avg_reward,0) / vals.length;
    return { model, avg, vals };
  }).sort((a,b)=>b.avg - a.avg).slice(0,5);
  const tbody = document.getElementById('leaderboard');
  tbody.innerHTML = '';
  rows.forEach((row, idx) => {
    const tr = document.createElement('tr');
    const sparkId = `spark-${idx}`;
    tr.innerHTML = `<td>${idx + 1}</td><td>${row.model}</td><td>${row.avg.toFixed(2)}</td><td><canvas id="${sparkId}" class="spark" width="120" height="36"></canvas></td>`;
    tbody.appendChild(tr);
    const sparkData = row.vals.sort((a,b)=>a.cycle-b.cycle).map(r => r.avg_reward);
    drawSparkline(document.getElementById(sparkId), sparkData);
  });
}

function drawSparkline(canvas, values) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!values.length) return;
  const min = Math.min(...values);
  const max = Math.max(...values);
  ctx.strokeStyle = '#14f195';
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1 || 1)) * canvas.width;
    const y = (1 - (v - min) / ((max - min) || 1)) * canvas.height;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function refreshReports() {
  try {
    const res = await apiFetch('/api/reports');
    const data = await res.json();
    reportCache = data.reports || [];
    const key = reportCache.map(r => r.run_timestamp || '').join('|');
    if (key !== lastReportKey) {
      updateRunSelectors();
      updateTimeline();
      updateCohorts();
      lastReportKey = key;
    }
    const status = document.getElementById('cohortStatus');
    if (status) status.textContent = `Runs available: ${reportCache.length}`;
  } catch (err) {
    reportCache = [];
    const status = document.getElementById('cohortStatus');
    if (status) status.textContent = 'Runs available: error loading reports';
  }
}

function updateRunSelectors() {
  const select = document.getElementById('profileSelect');
  const previous = selectedRun;
  select.innerHTML = '';
  const optAll = document.createElement('option');
  optAll.value = 'all';
  optAll.textContent = 'All Runs';
  select.appendChild(optAll);
  reportCache.forEach(rep => {
    const label = `${rep.run_timestamp || 'run'} (${rep.profile || 'default'})`;
    const opt = document.createElement('option');
    opt.value = rep.run_timestamp || '';
    opt.textContent = label;
    select.appendChild(opt);
  });
  if (previous && Array.from(select.options).some(o => o.value === previous)) {
    select.value = previous;
  } else {
    select.value = 'all';
    selectedRun = 'all';
  }
}

function filterByRun(series, runId) {
  if (!runId || runId === 'all') return series;
  const byRunId = series.filter(row => row.run_id && row.run_id === runId);
  if (byRunId.length) return byRunId;
  const report = reportCache.find(r => r.run_timestamp === runId);
  if (!report) return series;
  const cycles = report.summary?.cycles || [];
  const tsSet = new Set(cycles.map(c => c.timestamp));
  return series.filter(row => tsSet.has(row.timestamp));
}

function updateTimeline() {
  const target = document.getElementById('timeline');
  target.innerHTML = '';
  const report = reportCache[reportCache.length - 1];
  if (!report || !report.summary || !report.summary.cycles) {
    target.textContent = 'No run reports yet.';
    return;
  }
  report.summary.cycles.forEach(cycle => {
    const item = document.createElement('div');
    item.className = 'timeline-item';
    item.innerHTML = `<span class="dot"></span><span>Cycle ${cycle.cycle} @ ${cycle.timestamp}</span>`;
    item.addEventListener('click', () => {
      document.getElementById('timelineDetail').textContent = `Cycle ${cycle.cycle} best ${cycle.best_id} reward ${cycle.best_score?.toFixed(2)}`;
    });
    target.appendChild(item);
  });
}

function updateCohorts() {
  const selectA = document.getElementById('cohortA');
  const selectB = document.getElementById('cohortB');
  const prevA = selectA.value;
  const prevB = selectB.value;
  [selectA, selectB].forEach(sel => {
    sel.innerHTML = '';
    const optAll = document.createElement('option');
    optAll.value = 'all';
    optAll.textContent = 'All Runs';
    sel.appendChild(optAll);
    reportCache.forEach(rep => {
      const opt = document.createElement('option');
      opt.value = rep.run_timestamp || '';
      opt.textContent = `${rep.run_timestamp || 'run'} (${rep.profile || 'default'})`;
      sel.appendChild(opt);
    });
  });
  if (reportCache.length) {
    const defaultA = reportCache[0].run_timestamp || 'all';
    const defaultB = reportCache[Math.min(1, reportCache.length - 1)].run_timestamp || 'all';
    const valuesA = Array.from(selectA.options).map(o => o.value);
    const valuesB = Array.from(selectB.options).map(o => o.value);
    selectA.value = valuesA.includes(prevA) ? prevA : defaultA;
    selectB.value = valuesB.includes(prevB) ? prevB : defaultB;
  } else {
    selectA.value = 'all';
    selectB.value = 'all';
  }
  renderCohortTable();
}

function renderCohortTable() {
  const runA = document.getElementById('cohortA').value;
  const runB = document.getElementById('cohortB').value;
  const seriesA = filterByRun(metricsSeries, runA);
  const seriesB = filterByRun(metricsSeries, runB);
  const status = document.getElementById('cohortStatus');
  if (status) {
    const missingA = runA !== 'all' && !seriesA.length;
    const missingB = runB !== 'all' && !seriesB.length;
    if (missingA || missingB) {
      status.textContent = 'Runs available: metrics for selected runs not yet tagged';
    } else {
      status.textContent = `Runs available: ${reportCache.length}`;
    }
  }
  const rewardA = seriesA.length ? avgOf(seriesA, 'avg_reward') : avgRewardFromReport(runA);
  const rewardB = seriesB.length ? avgOf(seriesB, 'avg_reward') : avgRewardFromReport(runB);
  const rows = [
    ['Avg Reward', rewardA, rewardB],
    ['Win Rate', avgOf(seriesA, 'win_rate'), avgOf(seriesB, 'win_rate')],
    ['Return Rate', avgOf(seriesA, 'avg_return_rate'), avgOf(seriesB, 'avg_return_rate')],
    ['Rally Length', avgOf(seriesA, 'avg_rally_length'), avgOf(seriesB, 'avg_rally_length')],
  ];
  const tbody = document.getElementById('cohortTable');
  tbody.innerHTML = '';
  rows.forEach(([label, a, b]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${label}</td><td>${fmt(a)}</td><td>${fmt(b)}</td>`;
    tbody.appendChild(tr);
  });
}

function avgOf(series, key) {
  if (!series.length) return null;
  return series.reduce((a,b)=>a+b[key],0) / series.length;
}

function avgRewardFromReport(runId) {
  if (!runId || runId === 'all') return null;
  const report = reportCache.find(r => r.run_timestamp === runId);
  const cycles = report?.summary?.cycles || [];
  const scores = [];
  cycles.forEach(cycle => {
    (cycle.scores || []).forEach(pair => {
      if (pair && pair.length >= 2) scores.push(pair[1]);
    });
  });
  if (!scores.length) return null;
  return scores.reduce((a,b)=>a+b,0) / scores.length;
}

function fmt(val) {
  if (val === null || val === undefined) return '--';
  return val.toFixed(2);
}

async function refreshAnnotations() {
  try {
    const res = await apiFetch('/api/annotations');
    const data = await res.json();
    annotationsCache = data.notes || [];
    renderNotes();
  } catch (err) {
    annotationsCache = [];
  }
}

function renderNotes() {
  const target = document.getElementById('notesList');
  if (!annotationsCache.length) {
    target.textContent = 'No annotations yet.';
    return;
  }
  target.innerHTML = annotationsCache.slice(-6).reverse().map(n => `Cycle ${n.cycle || '--'}: ${n.note || ''} (${n.saved_at || ''})`).join('<br/>');
}

function updateQualityGates(series) {
  if (!series.length) return;
  const latest = series.reduce((acc, r) => r.cycle > acc.cycle ? r : acc, series[0]);
  const rewardGate = parseFloat(document.getElementById('gateReward').value || '0');
  const winGate = parseFloat(document.getElementById('gateWin').value || '0');
  const rewardOk = latest.avg_reward >= rewardGate;
  const winOk = latest.win_rate >= winGate;
  const status = `Reward ${latest.avg_reward.toFixed(2)} (${rewardOk ? 'pass' : 'fail'}) | Win ${latest.win_rate.toFixed(2)} (${winOk ? 'pass' : 'fail'})`;
  const node = document.getElementById('gateStatus');
  node.textContent = status;
  node.className = rewardOk && winOk ? 'tiny ok' : 'tiny alert';
}

function updateVideoInsights(series) {
  const target = document.getElementById('videoInsights');
  if (!series.length) {
    target.textContent = 'No rally insights yet.';
    return;
  }
  const top = [...series].sort((a,b)=>b.avg_rally_length - a.avg_rally_length).slice(0,5);
  target.innerHTML = top.map(row => (
    `Cycle ${row.cycle} ${row.model_id}: rally ${row.avg_rally_length.toFixed(1)} @ ${row.timestamp} ` +
    `<button class="pill" data-jump="1">Jump video</button>`
  )).join('<br/>');
  target.querySelectorAll('button[data-jump]').forEach(btn => {
    btn.addEventListener('click', () => {
      const vid = document.getElementById('vidCombined');
      if (vid && vid.src) {
        vid.currentTime = 0;
        vid.play();
      }
    });
  });
}

document.getElementById('compareMode').addEventListener('change', (e) => {
  compareMode = e.target.value;
  refreshCharts();
});
document.getElementById('rollingToggle').addEventListener('change', (e) => {
  rollingEnabled = e.target.checked;
  refreshCharts();
});
document.getElementById('rollingWindow').addEventListener('change', refreshCharts);
document.getElementById('cycleSelect').addEventListener('change', refreshCharts);
document.getElementById('heatLog').addEventListener('change', refreshHeatmap);
document.getElementById('heatOverlay').addEventListener('change', refreshHeatmap);
document.getElementById('profileSelect').addEventListener('change', (e) => {
  selectedRun = e.target.value || 'all';
  refreshCharts();
});
document.getElementById('cohortA').addEventListener('change', renderCohortTable);
document.getElementById('cohortB').addEventListener('change', renderCohortTable);
document.getElementById('gateReward').addEventListener('input', () => refreshCharts());
document.getElementById('gateWin').addEventListener('input', () => refreshCharts());
document.getElementById('noteSave').addEventListener('click', async () => {
  const cycle = parseInt(document.getElementById('noteCycle').value || '0', 10);
  const note = document.getElementById('noteText').value || '';
  if (!note) return;
  await apiFetch('/api/annotations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cycle, note, run_id: selectedRun }) });
  document.getElementById('noteText').value = '';
  refreshAnnotations();
});
document.getElementById('downloadCsv').addEventListener('click', () => {
  window.location.href = '/file?path=' + encodeURIComponent('logs/metrics.csv');
});
document.getElementById('snapshotBtn').addEventListener('click', () => {
  const id = document.getElementById('snapshotSelect').value;
  if (!id) return;
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const link = document.createElement('a');
  link.download = `${id}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
});

function initSnapshotOptions() {
  const ids = ['chartReward', 'chartWin', 'chartDelta', 'chartRally', 'histReward', 'histRally', 'scatterRally', 'scatterReturn', 'heatmap'];
  const select = document.getElementById('snapshotSelect');
  ids.forEach(id => {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = id;
    select.appendChild(opt);
  });
}

initSnapshotOptions();
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
