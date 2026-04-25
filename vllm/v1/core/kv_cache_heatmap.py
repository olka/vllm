# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache attention heatmap — G1HeapVis-style ECharts visualization.

Writes a self-contained HTML file with an ECharts heatmap grid where
each cell = one KV cache block, color = Q@K attention intensity.
Hover tooltips decode token IDs to text via @huggingface/transformers.
Auto-refreshes to show live updates.
"""

import json
import math
import os
import time
from pathlib import Path

_HEATMAP_PATH = Path(os.environ.get(
    "VLLM_KV_HEATMAP_PATH", "/tmp/vllm_kv_heatmap.html"
))

_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="5">
    <title>vLLM KV Cache Heatmap</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
</head>
<body class="bg-gray-50 text-gray-800 min-h-screen">
    <header class="border-b border-gray-200 bg-white px-6 py-4 flex items-center justify-between shadow-sm">
        <div>
            <h1 class="text-xl font-bold text-gray-900 tracking-tight">
                vLLM KV Cache Heatmap
            </h1>
            <p class="text-xs text-gray-400">
                Q@K Attention Scoring — MODEL_NAME — auto-refreshes every 5s
            </p>
        </div>
        <div class="text-right">
            <span id="tokStatus" class="text-xs text-gray-400">Loading tokenizer...</span>
            <br><span class="text-xs text-gray-400">TIMESTAMP</span>
        </div>
    </header>

    <div class="flex">
        <div class="flex-1 p-4">
            CHARTS
        </div>

        <aside class="w-56 p-4 pt-8 border-l border-gray-200 flex-shrink-0">
            <div class="mb-6">
                <h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                    Block Types
                </h3>
                <div class="space-y-2 text-sm text-gray-700">
                    <div class="flex items-center gap-2">
                        <span class="w-3.5 h-3.5 rounded-sm flex-shrink-0"
                              style="background:#ffd700;border:1px solid #b8960e"></span>
                        <span>Protected head</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="w-3.5 h-3.5 rounded-sm flex-shrink-0"
                              style="background:#00ff88;border:1px solid #00b05e"></span>
                        <span>Protected tail</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="w-3.5 h-3.5 rounded-sm flex-shrink-0"
                              style="background:#555;border:1px solid #333"></span>
                        <span>Evicted</span>
                    </div>
                </div>
            </div>
            <div class="mb-6">
                <h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                    Attention Scale
                </h3>
                <div class="text-sm text-gray-700 space-y-1">
                    <div class="flex items-center gap-2">
                        <span class="w-3.5 h-3.5 rounded-sm flex-shrink-0"
                              style="background:#0d0887"></span>
                        <span>Cold (low attention)</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="w-3.5 h-3.5 rounded-sm flex-shrink-0"
                              style="background:#ed7953"></span>
                        <span>Warm (≥ 0.08)</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="w-3.5 h-3.5 rounded-sm flex-shrink-0"
                              style="background:#d44842"></span>
                        <span>Hot (≥ 0.10)</span>
                    </div>
                </div>
            </div>
            METRICS
        </aside>
    </div>

    <script type="module">
        import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

        var tokStatus = document.getElementById('tokStatus');
        var tokenizer = null;
        var decodeCache = {};

        try {
            tokenizer = await AutoTokenizer.from_pretrained('MODEL_NAME');
            tokStatus.textContent = 'Tokenizer loaded';
            tokStatus.style.color = '#22c55e';
        } catch(e) {
            tokStatus.textContent = 'Tokenizer unavailable';
            tokStatus.style.color = '#ef4444';
        }

        // Expose decode function globally for chart tooltips.
        window.decodeTokenIds = function(ids) {
            if (!ids || ids.length === 0) return '';
            var key = ids.join(',');
            if (decodeCache[key]) return decodeCache[key];
            if (!tokenizer) return 'IDs: [' + ids.slice(0, 8).join(', ') + (ids.length > 8 ? '...' : '') + ']';
            try {
                var text = tokenizer.decode(ids, { skip_special_tokens: false });
                if (text.length > 80) text = text.slice(0, 77) + '...';
                decodeCache[key] = text;
                return text;
            } catch(e) {
                return 'IDs: [' + ids.slice(0, 8).join(', ') + '...]';
            }
        };
    </script>
</body>
</html>
"""

_CHART_TEMPLATE = """\
<div class="mb-2">
    <span class="text-xs font-semibold text-gray-500 uppercase tracking-wider">
        REQ_LABEL
    </span>
    <span class="text-xs text-gray-400 ml-2">
        TOTAL_BLOCKS blocks | COMPUTED tokens |
        ratio: RATIOx
    </span>
</div>
<div id="CHART_ID" class="w-full bg-white rounded-lg border border-gray-200 shadow-sm"
     style="height: CHART_HEIGHTpx"></div>
<script type="text/javascript">
(function() {
    var chart = echarts.init(document.getElementById('CHART_ID'));
    var data = BLOCK_DATA;
    var tokenIds = TOKEN_IDS;
    var cols = GRID_COLS;
    var rows = Math.ceil(data.length / cols);

    var xCats = [];
    for (var i = 0; i < cols; i++) xCats.push(i);
    var yCats = [];
    for (var i = 0; i < rows; i++) yCats.push(i);

    // Build heatmap data: [x, y, score, blockPos, state, rawScore]
    // Linear value-based: visualMap range [0, 0.1] with palette
    // anchored so 0.08 lands at orange (warm) and 0.10 at red (hot).
    var heatData = [];
    for (var i = 0; i < data.length; i++) {
        var x = i % cols;
        var y = Math.floor(i / cols);
        var raw = data[i][1];
        heatData.push([x, y, raw, data[i][0], data[i][2], raw]);
    }

    var stateNames = {0: 'Active', 1: 'Protected head', 2: 'Protected tail', 3: 'Evicted'};
    var stateColors = {1: '#ffd700', 2: '#00ff88', 3: '#555555'};

    var option = {
        tooltip: {
            formatter: function(params) {
                if (!params.value || params.seriesName === 'Markers') return '';
                var d = params.value;
                var state = stateNames[d[4]] || 'Active';
                var blockPos = d[3];
                var raw = d[5];
                var ids = tokenIds[blockPos] || [];
                var decoded = window.decodeTokenIds ? window.decodeTokenIds(ids) : '';
                var escapedText = decoded.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
                return '<b>Block ' + blockPos + '</b> (' + state + ')<br/>'
                    + 'Score: ' + raw.toFixed(6) + '<br/>'
                    + '<span style="color:#aaa;font-size:11px;max-width:300px;display:inline-block;word-break:break-all">'
                    + escapedText + '</span>';
            },
            extraCssText: 'max-width:350px;white-space:normal;'
        },
        grid: {
            height: '85%', width: '88%', top: '4%', left: '6%'
        },
        xAxis: {
            type: 'category', data: xCats, show: false,
            splitArea: { show: false }
        },
        yAxis: {
            type: 'category', data: yCats, show: false,
            splitArea: { show: false },
            inverse: true
        },
        visualMap: {
            min: MIN_SCORE,
            max: MAX_SCORE,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: '2%',
            itemWidth: 12,
            itemHeight: 120,
            textStyle: { color: '#666', fontSize: 10 },
            formatter: function(v) {
                return v.toFixed(3);
            },
            inRange: {
                color: ['#0d0887', '#3b049a', '#2563eb',
                        '#fdae61', '#ed7953', '#d44842']
            }
        },
        series: [{
            name: 'Attention',
            type: 'heatmap',
            data: heatData,
            progressive: 0,
            animation: false,
            itemStyle: {
                borderWidth: 1,
                borderColor: '#f0f0f0'
            },
            emphasis: {
                itemStyle: {
                    shadowBlur: 8,
                    shadowColor: 'rgba(0,0,0,0.4)'
                }
            }
        },
        {
            name: 'Markers',
            type: 'scatter',
            data: heatData.filter(function(d) { return d[4] > 0; }).map(function(d) {
                return {
                    value: [d[0], d[1]],
                    itemStyle: {
                        color: 'transparent',
                        borderColor: stateColors[d[4]] || '#fff',
                        borderWidth: 2,
                        borderType: d[4] === 3 ? 'dashed' : 'solid'
                    }
                };
            }),
            symbolSize: 12,
            symbol: 'rect',
            silent: true,
            animation: false,
            z: 10
        }]
    };
    chart.setOption(option);
    window.addEventListener('resize', function() { chart.resize(); });
})();
</script>
"""

_METRICS_TEMPLATE = """\
<div class="mb-4">
    <h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
        Metrics — REQ_LABEL
    </h3>
    <div class="space-y-1.5 text-sm font-mono">
        <div class="flex justify-between">
            <span class="text-gray-400">Blocks:</span>
            <span class="text-gray-900">TOTAL_BLOCKS</span>
        </div>
        <div class="flex justify-between">
            <span class="text-gray-400">Tokens:</span>
            <span class="text-gray-900">COMPUTED</span>
        </div>
        <div class="flex justify-between">
            <span class="text-gray-400">Min score:</span>
            <span class="text-gray-900">MIN_SCORE</span>
        </div>
        <div class="flex justify-between">
            <span class="text-gray-400">Max score:</span>
            <span class="text-gray-900">MAX_SCORE</span>
        </div>
        <div class="flex justify-between">
            <span class="text-gray-400">Ratio:</span>
            <span class="text-gray-900">RATIOx</span>
        </div>
        <div class="flex justify-between">
            <span class="text-gray-400">Evicted:</span>
            <span class="text-gray-900">NUM_EVICTED</span>
        </div>
    </div>
</div>
"""


def write_heatmap(
    requests_data: list[dict],
    model_name: str = "",
    cols: int = 40,
) -> None:
    """Write the heatmap HTML file.

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
        model_name: HuggingFace model name for client-side tokenizer.
        cols: Number of columns in the grid.
    """
    charts_html = []
    metrics_html = []

    for idx, req in enumerate(requests_data):
        scores = req["scores"]
        total = req["total_blocks"]
        head_n = req["num_protected_head"]
        tail_n = req["num_protected_tail"]
        evicted = req.get("evicted_positions", set())
        block_tids = req.get("block_token_ids", {})
        req_label = req["req_id"][-12:]

        if not scores:
            continue

        all_vals = list(scores.values())
        min_s = min(all_vals)
        max_s = max(all_vals)
        ratio = max_s / max(min_s, 1e-10)
        # Linear value-based color mapping with explicit thresholds:
        #   bounds [0, 0.1], palette [purple, purple, blue, orange, red].
        # 0.08 -> ~80% -> orange (warm threshold).
        # 0.10 -> 100% -> red (hot).
        # Anything above 0.10 clamps to red. Cold differentiation below
        # 0.08 is intentionally sacrificed for clear absolute thresholds —
        # log scale and rank-based both made it too hard to reason about
        # which color means what score.
        log_min = 0.0
        log_max = 0.1

        # Block data: [block_pos, score, state]. JS reads d[1] for color.
        block_data = []
        for pos in range(total):
            score = scores.get(pos, 0.0)
            if pos < head_n:
                state = 1
            elif pos >= total - tail_n:
                state = 2
            elif pos in evicted:
                state = 3
            else:
                state = 0
            block_data.append([pos, round(score, 8), state])

        # Token IDs map: {block_pos: [token_ids]}
        token_ids_map = {}
        for pos in range(total):
            tids = block_tids.get(pos, [])
            if tids:
                token_ids_map[pos] = tids

        rows = max(1, -(-total // cols))
        chart_height = max(200, rows * 16 + 80)

        chart = _CHART_TEMPLATE
        chart = chart.replace("CHART_ID", f"chart_{idx}")
        chart = chart.replace("REQ_LABEL", req_label)
        chart = chart.replace("TOTAL_BLOCKS", str(total))
        chart = chart.replace("COMPUTED",
                              str(req.get("num_computed_tokens", 0)))
        chart = chart.replace("RATIO", f"{ratio:.0f}")
        chart = chart.replace("GRID_COLS", str(cols))
        chart = chart.replace("CHART_HEIGHT", str(chart_height))
        chart = chart.replace("MIN_SCORE", f"{log_min:.4f}")
        chart = chart.replace("MAX_SCORE", f"{log_max:.4f}")
        chart = chart.replace("BLOCK_DATA", json.dumps(block_data))
        chart = chart.replace("TOKEN_IDS", json.dumps(token_ids_map))
        charts_html.append(chart)

        metrics = _METRICS_TEMPLATE
        metrics = metrics.replace("REQ_LABEL", req_label)
        metrics = metrics.replace("TOTAL_BLOCKS", str(total))
        metrics = metrics.replace("COMPUTED",
                                  str(req.get("num_computed_tokens", 0)))
        metrics = metrics.replace("MIN_SCORE", f"{min_s:.6f}")
        metrics = metrics.replace("MAX_SCORE", f"{max_s:.6f}")
        metrics = metrics.replace("RATIO", f"{ratio:.0f}")
        metrics = metrics.replace("NUM_EVICTED", str(len(evicted)))
        metrics_html.append(metrics)

    if not charts_html:
        return

    page = _HTML_TEMPLATE
    page = page.replace("MODEL_NAME", model_name)
    page = page.replace("TIMESTAMP", time.strftime("%Y-%m-%d %H:%M:%S"))
    page = page.replace("CHARTS", "\n".join(charts_html))
    page = page.replace("METRICS", "\n".join(metrics_html))

    try:
        _HEATMAP_PATH.write_text(page)
    except OSError:
        pass
