# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache attention heatmap — metrics dump + static viewer.

The eviction manager calls ``write_heatmap`` once per scoring cycle.
We persist the per-cycle metrics as JSON (a rolling buffer of recent
cycles) and write a self-contained viewer HTML once at startup. The
viewer fetches the JSON, renders the heatmap with ECharts, and
provides a timeline scrubber so a researcher can step through the
trace forward and backward.

Output files:

  - ``/tmp/vllm_kv_metrics.json`` (or ``$VLLM_KV_METRICS_PATH``):
    JSON array of recent cycles. Overwritten on every call with the
    full rolling buffer. Schema documented in ``_dump_snapshot``.
  - ``/tmp/vllm_kv_viewer.html`` (or ``$VLLM_KV_VIEWER_PATH``):
    Static HTML viewer. Written once on the first call and never
    rewritten. Fetches the JSON file via a relative URL and re-fetches
    every few seconds to stay live.

To view: open the HTML in a browser. The viewer lives next to the
JSON, so the relative fetch resolves correctly when both are served
from the same directory (``file://`` works for local use; serve via
``python -m http.server`` for remote use).
"""

import json
import os
import time
from collections import deque
from pathlib import Path

_METRICS_PATH = Path(os.environ.get(
    "VLLM_KV_METRICS_PATH", "/tmp/vllm_kv_metrics.json"
))
_VIEWER_PATH = Path(os.environ.get(
    "VLLM_KV_VIEWER_PATH", "/tmp/vllm_kv_viewer.html"
))
_BUFFER_SIZE = int(os.environ.get("VLLM_KV_METRICS_BUFFER", "500"))

# Archive mode: when ``VLLM_KV_METRICS_ARCHIVE_DIR`` is set, we ALSO
# write a per-cycle Parquet (or JSON, if pyarrow is unavailable) into
# that directory in addition to the rolling-buffer file. Used for
# post-hoc analysis: load the directory as a single dataset via
# ``polars.read_parquet(dir + "/*.parquet")`` or duckdb. The rolling
# buffer is still the source of truth for the live viewer; archive
# mode is purely additive and opt-in.
_ARCHIVE_DIR_ENV = os.environ.get("VLLM_KV_METRICS_ARCHIVE_DIR", "").strip()
_RUN_TAG = os.environ.get("VLLM_KV_METRICS_RUN_TAG", "run").strip() or "run"

_buffer: deque = deque(maxlen=_BUFFER_SIZE)
_seq = 0
_viewer_written = False
_archive_initialized = False
_archive_dir: Path | None = None
_archive_use_parquet: bool = False
_archive_warn_once: bool = False


def write_heatmap(
    requests_data: list[dict],
    model_name: str = "",
    cols: int = 40,
    dead_threshold: float | None = None,
) -> None:
    """Append a per-cycle snapshot to the metrics file and ensure the
    viewer HTML exists.

    Args:
        requests_data: List of dicts, each with:
            - req_id: str
            - total_blocks: int
            - num_computed_tokens: int
            - scores: dict[int, float]
            - num_protected_head: int
            - num_protected_tail: int
            - evicted_positions: set[int]
            - block_token_ids: dict[int, list[int]]
        model_name: HuggingFace model name for client-side tokenizer
            (consumed by the viewer to decode token IDs to text on
            hover; not used server-side).
        cols: Number of columns in the heatmap grid (viewer uses).
        dead_threshold: Current Otsu/static threshold; recorded in the
            snapshot's ``globals`` block so the timeline can chart its
            evolution.
    """
    if not requests_data:
        return

    global _seq, _viewer_written
    _seq += 1
    snapshot = _build_snapshot(
        requests_data,
        model_name=model_name,
        cols=cols,
        cycle=_seq,
        dead_threshold=dead_threshold,
    )
    _buffer.append(snapshot)

    try:
        _METRICS_PATH.write_text(json.dumps(list(_buffer)))
    except OSError:
        pass

    if not _viewer_written:
        try:
            _VIEWER_PATH.write_text(
                _VIEWER_HTML.replace(
                    "__METRICS_FILENAME__",
                    _METRICS_PATH.name,
                )
            )
            _viewer_written = True
        except OSError:
            pass

    if _ARCHIVE_DIR_ENV:
        _write_archive(snapshot)


def _write_archive(snapshot: dict) -> None:
    """Persist this snapshot to the archive directory in Parquet
    (preferred) or per-cycle JSON (fallback).

    Schema for Parquet: flat block-level records, one row per
    (run_id, cycle, req_id, block_pos). Columns include score, state
    flags, dead_threshold, num_computed_tokens, model_name, ts, and
    block_token_ids (list[int]). This is the format polars/duckdb
    consume natively for the §3.2 and §7.2/§7.3 post-hoc analyses
    (bimodal histograms, ratio tables, threshold trajectories,
    structural-floor validation across runs).

    Each cycle writes one file; analysis loads the whole directory
    via globbing. Multiple runs can share the directory if the
    ``VLLM_KV_METRICS_RUN_TAG`` env var is set per run.
    """
    global _archive_initialized, _archive_dir, _archive_use_parquet
    global _archive_warn_once

    if not _archive_initialized:
        _archive_dir = Path(_ARCHIVE_DIR_ENV)
        try:
            _archive_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        try:
            import pyarrow  # noqa: F401
            import pyarrow.parquet  # noqa: F401
            _archive_use_parquet = True
        except ImportError:
            _archive_use_parquet = False
            if not _archive_warn_once:
                # Lazy import to avoid pulling logger at module load.
                from vllm.logger import init_logger
                logger = init_logger(__name__)
                logger.warning(
                    "VLLM_KV_METRICS_ARCHIVE_DIR is set but pyarrow is "
                    "not installed; falling back to per-cycle JSON. "
                    "Install pyarrow for compressed columnar archives "
                    "that polars/duckdb can read directly.",
                )
                _archive_warn_once = True
        _archive_initialized = True

    if _archive_dir is None:
        return

    cycle = snapshot["cycle"]
    suffix = "parquet" if _archive_use_parquet else "json"
    path = _archive_dir / f"{_RUN_TAG}_cycle_{cycle:08d}.{suffix}"

    if _archive_use_parquet:
        _write_archive_parquet(snapshot, path)
    else:
        try:
            path.write_text(json.dumps(snapshot))
        except OSError:
            pass


def _write_archive_parquet(snapshot: dict, path: Path) -> None:
    """Flatten a snapshot to per-block records and write as Parquet
    with zstd compression."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict] = []
    cycle = snapshot["cycle"]
    ts = snapshot["ts"]
    model_name = snapshot["model_name"]
    g = snapshot.get("globals") or {}
    dead_threshold = g.get("dead_threshold")
    active_request_count = g.get("active_request_count", 0)

    for req in snapshot.get("requests", []):
        evicted = set(req.get("evicted_positions") or [])
        head_n = req.get("num_protected_head", 0)
        tail_n = req.get("num_protected_tail", 0)
        total = req.get("total_blocks", 0)
        scores = req.get("scores") or {}
        token_ids_map = req.get("block_token_ids") or {}
        for pos in range(total):
            score = float(scores.get(str(pos), 0.0))
            tail_start = total - tail_n
            if pos < head_n:
                state = "head_protected"
            elif pos >= tail_start:
                state = "tail_protected"
            elif pos in evicted:
                state = "evicted"
            else:
                state = "resident"
            rows.append({
                "run_id": _RUN_TAG,
                "cycle": cycle,
                "ts": ts,
                "model_name": model_name,
                "dead_threshold": dead_threshold,
                "active_request_count": active_request_count,
                "req_id": req.get("req_id", ""),
                "total_blocks": total,
                "num_computed_tokens": req.get("num_computed_tokens", 0),
                "block_pos": pos,
                "score": score,
                "state": state,
                "is_evicted": (state == "evicted"),
                "token_ids": list(token_ids_map.get(str(pos), [])),
            })
    if not rows:
        return
    try:
        table = pa.Table.from_pylist(rows)
        pq.write_table(
            table, path, compression="zstd", use_dictionary=True,
        )
    except OSError:
        pass
    except Exception:
        # Defensive: any pyarrow conversion error shouldn't break the
        # serving loop. Log but continue.
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        logger.exception(
            "Failed to write KV metrics archive to %s; archive disabled "
            "for the rest of this run.", path,
        )
        global _archive_use_parquet
        _archive_use_parquet = False


def _build_snapshot(
    requests_data: list[dict],
    *,
    model_name: str,
    cols: int,
    cycle: int,
    dead_threshold: float | None,
) -> dict:
    """Convert per-request runtime structures into a plain dict ready
    for JSON serialization. Sets become lists; integer-keyed dicts
    become string-keyed dicts (JSON requirement)."""
    requests_out = []
    for req in requests_data:
        scores = req.get("scores") or {}
        scores_out = {
            str(int(k)): round(float(v), 8)
            for k, v in scores.items()
        }
        evicted = req.get("evicted_positions") or set()
        block_token_ids = req.get("block_token_ids") or {}
        block_token_ids_out = {
            str(int(k)): list(v)
            for k, v in block_token_ids.items()
        }
        requests_out.append({
            "req_id": req["req_id"],
            "total_blocks": int(req["total_blocks"]),
            "num_computed_tokens": int(req.get("num_computed_tokens", 0)),
            "num_protected_head": int(req.get("num_protected_head", 0)),
            "num_protected_tail": int(req.get("num_protected_tail", 0)),
            "scores": scores_out,
            "evicted_positions": sorted(int(p) for p in evicted),
            "block_token_ids": block_token_ids_out,
        })
    return {
        "ts": time.time(),
        "cycle": cycle,
        "model_name": model_name,
        "cols": cols,
        "globals": {
            "dead_threshold": dead_threshold,
            "active_request_count": len(requests_out),
        },
        "requests": requests_out,
    }


# ---------------------------------------------------------------------
# Static viewer HTML — written once at startup. The placeholder
# ``__METRICS_FILENAME__`` is replaced with the basename of the metrics
# file so the viewer fetches it via a relative URL.
# ---------------------------------------------------------------------

_VIEWER_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>vLLM KV Cache Heatmap</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 16px;
    background: #fafafa;
    color: #222;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
  }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  .subtitle { color: #666; font-size: 12px; }
  #status { color: #888; font-size: 11px; }
  #tokenizer-status { color: #888; font-size: 11px; margin-left: 8px; }

  #timeline {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  #slider {
    flex: 1;
    height: 6px;
  }
  #cycle-info {
    font-family: monospace;
    font-size: 12px;
    color: #444;
    min-width: 240px;
    text-align: right;
  }
  button {
    padding: 4px 10px;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    background: #fff;
    cursor: pointer;
    font-size: 12px;
    user-select: none;
  }
  button:hover { background: #f0f0f0; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  label { font-size: 12px; color: #555; user-select: none; }

  /* Two-pane layout: charts on left, inspector panel on right. */
  #main {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 16px;
  }
  #charts { min-width: 0; }
  #inspector {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 14px 16px;
    position: sticky;
    top: 16px;
    height: fit-content;
    max-height: calc(100vh - 32px);
    overflow-y: auto;
    font-size: 12px;
  }
  #inspector h2 {
    margin: 0 0 8px 0;
    font-size: 13px;
    font-weight: 600;
    color: #222;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  #inspector .placeholder {
    color: #aaa;
    font-style: italic;
  }
  #inspector .field {
    margin-bottom: 8px;
  }
  #inspector .field-label {
    display: block;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #888;
    margin-bottom: 2px;
  }
  #inspector .field-value {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 11px;
    color: #222;
    word-break: break-all;
  }
  #inspector .decoded-text {
    background: #f6f6f6;
    border: 1px solid #e8e8e8;
    border-radius: 4px;
    padding: 8px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 11px;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 360px;
    overflow-y: auto;
    color: #111;
  }
  #inspector .token-ids {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 10px;
    color: #555;
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 4px;
    padding: 6px 8px;
    max-height: 100px;
    overflow-y: auto;
  }
  #inspector .state-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  #inspector .state-resident { background: #e3f2e3; color: #2a662a; }
  #inspector .state-head { background: #fff3e0; color: #a85f00; }
  #inspector .state-tail { background: #e8f0fb; color: #2a5790; }
  #inspector .state-evicted { background: #fbe5e3; color: #a8302a; }

  .chart-container {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    margin-bottom: 12px;
    padding: 10px 14px;
  }
  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
    font-size: 12px;
    color: #666;
  }
  .chart-title { font-weight: 600; color: #222; }
  .chart {
    width: 100%;
    height: 220px;
  }

  @media (max-width: 1100px) {
    #main { grid-template-columns: 1fr; }
    #inspector { position: static; max-height: none; }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>vLLM KV Cache Heatmap</h1>
    <div class="subtitle">
      Q@K Attention Scoring &mdash; <span id="model-name">(loading...)</span>
      <span id="tokenizer-status">tokenizer: idle</span>
    </div>
  </div>
  <div style="display:flex; gap:10px; align-items:center;">
    <button id="btn-load-file" title="Load metrics file (offline mode)">
      📂 Load file…
    </button>
    <input type="file" id="file-picker" accept=".json,application/json"
           style="display:none">
    <div id="status">no data</div>
  </div>
</header>

<div id="timeline">
  <button id="btn-first" title="First cycle">⏮</button>
  <button id="btn-prev" title="Previous cycle">◀</button>
  <button id="btn-play" title="Play">▶</button>
  <button id="btn-next" title="Next cycle">▶</button>
  <button id="btn-last" title="Latest cycle">⏭</button>
  <input type="range" id="slider" min="0" max="0" value="0">
  <label><input type="checkbox" id="follow" checked> follow latest</label>
  <div id="cycle-info">cycle: – | ts: –</div>
</div>

<div id="main">
  <div id="charts"></div>
  <aside id="inspector">
    <h2>
      Block Inspector
      <button id="btn-clear-inspect" style="font-size:10px; padding:2px 6px;">×</button>
    </h2>
    <div id="inspector-body" class="placeholder">
      Click any block to inspect its tokens, score, and state.
    </div>
  </aside>
</div>

<script type="module">
const METRICS_URL = "__METRICS_FILENAME__";
const REFRESH_MS = 3000;
const PLAY_INTERVAL_MS = 400;
const TOKENIZER_OVERRIDE_PARAM = "tokenizer";

let cycles = [];          // array of snapshot objects
let activeIndex = 0;      // current position in cycles[]
let charts = new Map();   // req_id -> ECharts instance
let playing = false;
let playTimer = null;
let tokenizer = null;
let tokenizerLoading = null;
let pendingInspect = null; // {req_id, pos} — re-rendered as snapshots change
let liveMode = true;       // false when a file has been loaded manually
let refreshTimer = null;   // setInterval handle for live refresh

const $ = id => document.getElementById(id);

// --- Tokenizer (lazy-loaded from CDN; only runs when first needed) ---
// Strip common quantization suffixes so e.g. ``Qwen/Qwen3-8B-FP8`` ->
// ``Qwen/Qwen3-8B`` (the base repo where tokenizer.json actually lives).
function stripQuantSuffix(modelName) {
  if (!modelName) return "";
  return modelName.replace(
    /-(?:FP8|fp8|FP4|fp4|Int8|int8|INT8|Int4|int4|INT4|GPTQ|gptq|AWQ|awq|GGUF|gguf)$/,
    ""
  );
}

async function tryLoadTokenizer(mod, candidate) {
  return await mod.AutoTokenizer.from_pretrained(candidate, {
    legacy: false,
  });
}

async function ensureTokenizer(modelName) {
  if (tokenizer) return tokenizer;
  if (tokenizerLoading) return tokenizerLoading;
  const params = new URLSearchParams(window.location.search);
  const override = params.get(TOKENIZER_OVERRIDE_PARAM);
  const primary = override || modelName;
  const fallback = !override && modelName ? stripQuantSuffix(modelName) : "";
  const candidates = [primary];
  if (fallback && fallback !== primary) candidates.push(fallback);
  if (candidates.length === 0 || !candidates[0]) return null;

  $("tokenizer-status").textContent =
    "tokenizer: loading " + candidates[0];
  tokenizerLoading = (async () => {
    let mod;
    try {
      // Use the official @huggingface/transformers (Xenova's fork was
      // upstreamed). esm.sh is reliable for ES modules; the jsdelivr
      // path occasionally serves a CommonJS bundle that fails to load
      // as a module.
      mod = await import(
        "https://esm.sh/@huggingface/transformers@3"
      );
    } catch (e) {
      $("tokenizer-status").textContent =
        "tokenizer library load failed: " + (e.message || e);
      console.warn("Tokenizer library load failed:", e);
      tokenizer = null;
      tokenizerLoading = null;
      return null;
    }
    let lastErr = null;
    for (const candidate of candidates) {
      try {
        $("tokenizer-status").textContent =
          "tokenizer: loading " + candidate;
        const tok = await tryLoadTokenizer(mod, candidate);
        tokenizer = tok;
        $("tokenizer-status").textContent = "tokenizer: " + candidate;
        tokenizerLoading = null;
        return tok;
      } catch (e) {
        lastErr = e;
        console.warn("Tokenizer load failed for " + candidate + ":", e);
      }
    }
    $("tokenizer-status").textContent =
      "tokenizer unavailable (tried: " + candidates.join(", ") + ")";
    console.warn(
      "All tokenizer candidates failed. Override by appending " +
      "?tokenizer=<HF model id> to the URL. Metrics model_name was '"
      + modelName + "'."
    );
    tokenizer = null;
    tokenizerLoading = null;
    return null;
  })();
  return tokenizerLoading;
}

// --- Data loading and refresh ---
async function refresh() {
  if (!liveMode) return;  // user has loaded a file manually; don't clobber it
  try {
    const r = await fetch(METRICS_URL + "?t=" + Date.now(), {cache: "no-store"});
    if (!r.ok) {
      $("status").textContent = "fetch failed: HTTP " + r.status +
        " — try '📂 Load file…' to browse manually";
      return;
    }
    const newCycles = await r.json();
    if (!Array.isArray(newCycles)) {
      $("status").textContent = "metrics file malformed";
      return;
    }
    const wasAtLatest = activeIndex >= cycles.length - 1;
    cycles = newCycles;
    $("slider").max = Math.max(0, cycles.length - 1);
    if ($("follow").checked || wasAtLatest) {
      activeIndex = cycles.length - 1;
      $("slider").value = activeIndex;
    }
    $("status").textContent = "live: " + cycles.length + " cycles";
    if (cycles.length > 0) {
      const modelName = cycles[0].model_name || "(unknown)";
      $("model-name").textContent = modelName;
      // Kick off tokenizer load on first data arrival.
      ensureTokenizer(modelName);
      render();
    }
  } catch (e) {
    // Most common cause on file:// is the browser's same-origin
    // policy blocking same-directory fetch. Hint at the offline path.
    const hint =
      window.location.protocol === "file:"
        ? " — file:// blocks fetch; use '📂 Load file…' or run "
          + "`python3 -m http.server` from this directory"
        : " — try '📂 Load file…' to browse manually";
    $("status").textContent = "fetch error: " + (e.message || e) + hint;
  }
}

function fmtTs(ts) {
  if (!ts) return "–";
  return new Date(ts * 1000).toLocaleTimeString();
}

function render() {
  if (cycles.length === 0) return;
  if (activeIndex < 0) activeIndex = 0;
  if (activeIndex > cycles.length - 1) activeIndex = cycles.length - 1;
  const snap = cycles[activeIndex];
  const dead = snap.globals && snap.globals.dead_threshold;
  $("cycle-info").textContent =
    "cycle " + snap.cycle +
    " | " + fmtTs(snap.ts) +
    " | dead_thr " + (dead != null ? dead.toFixed(6) : "–") +
    " | " + (snap.requests || []).length + " req";

  const chartsRoot = $("charts");
  const seenReqs = new Set();

  for (const req of snap.requests || []) {
    seenReqs.add(req.req_id);
    let container = document.getElementById("chart-" + req.req_id);
    if (!container) {
      container = document.createElement("div");
      container.className = "chart-container";
      container.id = "chart-" + req.req_id;
      container.innerHTML = `
        <div class="chart-header">
          <span class="chart-title">${req.req_id.slice(-12)}</span>
          <span class="chart-meta"></span>
        </div>
        <div class="chart"></div>`;
      chartsRoot.appendChild(container);
    }
    renderRequest(container, req, snap);
  }

  for (const child of chartsRoot.children) {
    const reqId = child.id && child.id.replace(/^chart-/, "");
    child.style.display = seenReqs.has(reqId) ? "" : "none";
  }

  // Re-render the inspector for the same (req_id, pos) as the snapshot
  // changes — lets the user scrub through time and watch one block's
  // score/state evolve.
  if (pendingInspect) renderInspector();
}

function renderRequest(container, req, snap) {
  const total = req.total_blocks;
  const cols = snap.cols || 40;
  const rows = Math.max(1, Math.ceil(total / cols));
  const headN = req.num_protected_head;
  const tailN = req.num_protected_tail;
  const evictedSet = new Set(req.evicted_positions || []);
  const scores = req.scores || {};

  // Color always reflects the actual score (you can still see what an
  // evicted block scored). State is shown via an overlay scatter series
  // that draws rectangular borders on top: solid yellow for head-
  // protected, solid green for tail-protected, dashed gray for evicted.
  // This preserves the score signal across all blocks while making
  // state visually distinct.
  const STATE_COLORS = { 1: "#ffd700", 2: "#00ff88", 3: "#555555" };

  const data = [];
  const markers = [];
  const validScores = [];
  let minS = Infinity, maxS = -Infinity;
  for (let pos = 0; pos < total; pos++) {
    const score = parseFloat(scores[pos] || 0);
    if (score > 0) {
      if (score < minS) minS = score;
      if (score > maxS) maxS = score;
      validScores.push(score);
    }
    let state = 0;
    if (pos < headN) state = 1;
    else if (pos >= total - tailN) state = 2;
    else if (evictedSet.has(pos)) state = 3;
    const row = Math.floor(pos / cols);
    const col = pos % cols;
    // Layout: [x, y, score, state, pos]
    data.push([col, row, score, state, pos]);
    if (state !== 0) {
      markers.push({
        value: [col, row],
        itemStyle: {
          color: "transparent",
          borderColor: STATE_COLORS[state],
          borderWidth: 2,
          borderType: state === 3 ? "dashed" : "solid",
        },
        // Carry pos through so the click handler can read it.
        _pos: pos,
        _state: state,
      });
    }
  }
  const ratio = (minS > 0 && maxS > 0) ? (maxS / minS) : 0;

  // Adaptive visualMap range. Fixed [0, 0.1] worked for the late-trace
  // bimodal regime but fails when scores are predominantly above 0.1
  // (early decode, or after an eviction sweep that shifted the
  // distribution). Use the 95th percentile as the max so sink-class
  // outliers (1.0+) don't compress the gradient. Floor at 0.05 so
  // we always have a usable spread.
  validScores.sort((a, b) => a - b);
  const p95 = validScores.length > 0
    ? validScores[Math.min(
        validScores.length - 1,
        Math.floor(validScores.length * 0.95),
      )]
    : 0.1;
  const visMin = 0;
  const visMax = Math.max(0.05, p95);

  const headerMeta = container.querySelector(".chart-meta");
  if (headerMeta) {
    headerMeta.textContent =
      total + " blocks | " +
      req.num_computed_tokens + " tokens | " +
      "ratio " + (ratio > 0 ? ratio.toFixed(0) + "x" : "–");
  }

  const chartEl = container.querySelector(".chart");
  let inst = charts.get(req.req_id);
  if (!inst || inst.getDom() !== chartEl) {
    if (inst) inst.dispose();
    inst = echarts.init(chartEl);
    inst.on("click", "series", (params) => {
      if (!params || !params.value) return;
      // Heatmap series: value is [col, row, score, state, pos];
      // marker series is silent so it doesn't fire clicks.
      if (params.seriesName === "blocks" && params.value.length >= 5) {
        const pos = params.value[4];
        pendingInspect = { req_id: req.req_id, pos };
        renderInspector();
      }
    });
    charts.set(req.req_id, inst);
  }

  inst.setOption({
    grid: { left: 24, right: 24, top: 8, bottom: 30 },
    tooltip: {
      formatter: p => {
        if (p.seriesName !== "blocks" || !p.value || p.value.length < 5) {
          return "";
        }
        const [, , score, state, pos] = p.value;
        const stateLabel =
          ["resident", "head-protected", "tail-protected", "evicted"][state];
        return "pos " + pos + "<br>" +
          "score: " + score.toFixed(6) + "<br>" +
          "state: " + stateLabel +
          "<br><span style='font-size:10px;color:#888;'>click to inspect</span>";
      }
    },
    xAxis: { type: "category", data: Array.from({length: cols}, (_, i) => i), show: false },
    yAxis: { type: "category", data: Array.from({length: rows}, (_, i) => i), show: false, inverse: true },
    visualMap: {
      min: visMin,
      max: visMax,
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: "2%",
      itemWidth: 12,
      itemHeight: 120,
      textStyle: { color: "#666", fontSize: 10 },
      formatter: v => v.toFixed(3),
      inRange: {
        color: [
          "#0d0887", "#3b049a", "#2563eb",
          "#fdae61", "#ed7953", "#d44842",
        ],
      },
    },
    series: [
      {
        name: "blocks",
        type: "heatmap",
        data: data,
        itemStyle: {
          borderColor: "#f0f0f0",
          borderWidth: 1,
          borderRadius: 1,
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 8,
            shadowColor: "rgba(0,0,0,0.4)",
          },
        },
        progressive: 0,
        animation: false,
      },
      {
        name: "state-markers",
        type: "scatter",
        data: markers,
        symbol: "rect",
        // 14px nominally matches the heatmap cell at typical layouts;
        // ECharts scales scatter symbols independent of the heatmap
        // cell size, so on very wide charts the markers may slightly
        // under-cover the cell. Acceptable: state is still visually
        // unambiguous.
        symbolSize: 14,
        silent: true,         // don't fire clicks/tooltips
        animation: false,
        z: 10,
      },
    ]
  }, { notMerge: true });
}

// --- Inspector panel ---
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function renderInspector() {
  const body = $("inspector-body");
  if (!pendingInspect) {
    body.className = "placeholder";
    body.textContent = "Click any block to inspect its tokens, score, and state.";
    return;
  }
  const { req_id, pos } = pendingInspect;
  const snap = cycles[activeIndex];
  const req = (snap.requests || []).find(r => r.req_id === req_id);
  if (!req) {
    body.className = "placeholder";
    body.textContent = "request not in current cycle (" + req_id + ")";
    return;
  }
  const score = req.scores ? parseFloat(req.scores[pos] || 0) : 0;
  const headN = req.num_protected_head;
  const tailN = req.num_protected_tail;
  const total = req.total_blocks;
  const evictedSet = new Set(req.evicted_positions || []);
  let stateClass = "state-resident", stateText = "resident";
  if (pos < headN) { stateClass = "state-head"; stateText = "head-protected"; }
  else if (pos >= total - tailN) { stateClass = "state-tail"; stateText = "tail-protected"; }
  else if (evictedSet.has(pos)) { stateClass = "state-evicted"; stateText = "evicted"; }

  const tokenIds = (req.block_token_ids && req.block_token_ids[pos]) || [];

  body.className = "";
  body.innerHTML = `
    <div class="field">
      <span class="field-label">request</span>
      <span class="field-value">${escapeHtml(req_id)}</span>
    </div>
    <div class="field">
      <span class="field-label">block position</span>
      <span class="field-value">${pos} / ${total - 1}</span>
    </div>
    <div class="field">
      <span class="field-label">state</span>
      <span class="state-badge ${stateClass}">${stateText}</span>
    </div>
    <div class="field">
      <span class="field-label">score (Q@K)</span>
      <span class="field-value">${score.toFixed(6)}</span>
    </div>
    <div class="field">
      <span class="field-label">decoded tokens</span>
      <div class="decoded-text" id="inspector-decoded">
        ${tokenIds.length === 0 ? "<span style='color:#aaa;'>(no token ids)</span>" : "decoding..."}
      </div>
    </div>
    <div class="field">
      <span class="field-label">raw token ids (${tokenIds.length})</span>
      <div class="token-ids">${escapeHtml(JSON.stringify(tokenIds))}</div>
    </div>
  `;
  if (tokenIds.length > 0) {
    const decodedEl = $("inspector-decoded");
    const tok = await ensureTokenizer(snap.model_name || "");
    if (!tok) {
      decodedEl.innerHTML =
        "<span style='color:#aaa;'>tokenizer unavailable; raw ids below</span>";
      return;
    }
    try {
      const text = tok.decode(tokenIds, { skip_special_tokens: false });
      decodedEl.textContent = text;
    } catch (e) {
      decodedEl.innerHTML =
        "<span style='color:#a85;'>decode error: " + escapeHtml(e.message || e) + "</span>";
    }
  }
}

function setIndex(i) {
  if (cycles.length === 0) return;
  activeIndex = Math.max(0, Math.min(cycles.length - 1, i));
  $("slider").value = activeIndex;
  if (activeIndex < cycles.length - 1) {
    $("follow").checked = false;
  }
  render();
}

function togglePlay() {
  playing = !playing;
  $("btn-play").textContent = playing ? "⏸" : "▶";
  if (playing) {
    playTimer = setInterval(() => {
      if (activeIndex >= cycles.length - 1) {
        togglePlay();
        return;
      }
      setIndex(activeIndex + 1);
    }, PLAY_INTERVAL_MS);
  } else if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
}

$("slider").addEventListener("input", e => setIndex(parseInt(e.target.value)));
$("btn-first").addEventListener("click", () => setIndex(0));
$("btn-prev").addEventListener("click", () => setIndex(activeIndex - 1));
$("btn-next").addEventListener("click", () => setIndex(activeIndex + 1));
$("btn-last").addEventListener("click", () => setIndex(cycles.length - 1));
$("btn-play").addEventListener("click", togglePlay);
$("btn-clear-inspect").addEventListener("click", () => {
  pendingInspect = null;
  renderInspector();
});

// --- File picker (offline mode) ---
// Lets the viewer work without a fetch path — useful when:
//   * the page is opened directly via file:// (Chrome blocks
//     same-directory file fetches by default);
//   * an archived metrics JSON is being inspected after the run
//     finished (no live source);
//   * sharing a metrics dump with a collaborator.
$("btn-load-file").addEventListener("click", () => $("file-picker").click());
$("file-picker").addEventListener("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed)) {
      $("status").textContent =
        "load failed: expected JSON array of cycles, got " + typeof parsed;
      return;
    }
    // Switch to offline mode: cancel auto-refresh; the file is now
    // the source of truth.
    liveMode = false;
    if (refreshTimer !== null) {
      clearInterval(refreshTimer);
      refreshTimer = null;
    }
    cycles = parsed;
    activeIndex = cycles.length - 1;
    $("slider").max = Math.max(0, cycles.length - 1);
    $("slider").value = activeIndex;
    $("status").textContent =
      "file: " + file.name + " (" + cycles.length + " cycles)";
    if (cycles.length > 0) {
      const modelName = cycles[0].model_name || "(unknown)";
      $("model-name").textContent = modelName;
      ensureTokenizer(modelName);
      render();
    }
  } catch (err) {
    $("status").textContent = "load failed: " + (err.message || err);
  } finally {
    // Allow re-loading the same file (browsers de-dupe identical
    // selections without this).
    e.target.value = "";
  }
});

window.addEventListener("resize", () => {
  for (const inst of charts.values()) inst.resize();
});

// Try the live URL first; if it fails (file:// + Chrome CORS), the
// user can fall back to the file picker. The status indicator shows
// the failure reason in that case.
refresh();
refreshTimer = setInterval(() => { if (liveMode) refresh(); }, REFRESH_MS);
</script>
</body>
</html>
"""
