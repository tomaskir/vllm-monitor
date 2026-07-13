"""Metrics polling and parsing for vLLM server."""

from __future__ import annotations

import math
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import httpx

# Maximum history samples kept for sparkline
HISTORY_SIZE = 60

# Latency histograms we surface, keyed by a short name. Each exposes a vLLM
# `<name>_sum` / `<name>_count` pair from which we derive a mean. Any that the
# server doesn't expose simply stay absent (the UI shows "—").
LATENCY_HISTOGRAMS = {
    "e2e": "vllm:e2e_request_latency_seconds",
    "ttft": "vllm:time_to_first_token_seconds",
    "tpot": "vllm:request_time_per_output_token_seconds",
    "queue": "vllm:request_queue_time_seconds",
}

# Per-request token histograms whose cumulative mean lands on a VllmMetrics field.
_TOKEN_HISTOGRAMS = {
    "vllm:request_prompt_tokens": "avg_prompt_tokens",
    "vllm:request_generation_tokens": "avg_generation_tokens",
}
# Histogram base name → short latency key (inverse of LATENCY_HISTOGRAMS).
_LATENCY_BY_METRIC = {name: key for key, name in LATENCY_HISTOGRAMS.items()}
# Every histogram base we accumulate _sum/_count for in the single parse pass.
_HIST_BASES = frozenset(_LATENCY_BY_METRIC) | frozenset(_TOKEN_HISTOGRAMS)


@dataclass
class ModelInfo:
    model_id: str = "unknown"
    max_model_len: int | None = None
    tensor_parallel_size: int | None = None
    # From vllm:cache_config_info labels (when present)
    num_gpu_blocks: int | None = None
    block_size: int | None = None
    gpu_memory_utilization: float | None = None
    cache_dtype: str | None = None


@dataclass
class VllmMetrics:
    # Server state
    timestamp: float = 0.0
    server_reachable: bool = False

    # Request metrics
    num_requests_running: float = 0.0
    num_requests_waiting: float = 0.0
    num_requests_swapped: float = 0.0
    num_preemptions_total: float = 0.0
    request_success_total: float = 0.0
    # Completed-request counts keyed by finish reason (stop/length/abort/error/…)
    finished_reasons: dict[str, float] = field(default_factory=dict)

    # Token throughput (tokens/sec, computed as delta)
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    prompt_tokens_per_sec: float = 0.0
    generation_tokens_per_sec: float = 0.0

    # Running average generation-token throughput since monitoring started
    avg_gen_tokens_per_sec: float = 0.0

    # Average per-request token counts (from request_*_tokens histograms)
    avg_prompt_tokens: float = 0.0
    avg_generation_tokens: float = 0.0
    # Cumulative mean E2E latency per request (all-time sum/count from the
    # e2e_request_latency histogram). Distinct from latency_mean_s["e2e"],
    # which _compute_rates overwrites with the recent windowed mean for the
    # live E2E Latency card.
    avg_e2e_latency_s: float = 0.0

    # Cache
    gpu_cache_usage_perc: float = 0.0
    cpu_cache_usage_perc: float = 0.0
    gpu_prefix_cache_hit_rate: float = 0.0
    # Prefix cache counters (vLLM v1 — hit rate is derived from these)
    prefix_cache_queries_total: float = 0.0
    prefix_cache_hits_total: float = 0.0

    # Speculative decoding (MTP/draft). active=False when the server isn't
    # running spec decode (the metrics are absent) — the UI then shows "—".
    spec_decode_active: bool = False
    spec_draft_tokens_total: float = 0.0
    spec_accepted_tokens_total: float = 0.0
    spec_drafts_total: float = 0.0
    spec_acceptance_rate: float = 0.0  # percent (accepted / drafted tokens)
    spec_accept_length: float = 0.0  # accepted tokens per draft step

    # Latency histograms, keyed by short name (see LATENCY_HISTOGRAMS). Means
    # are recent (delta between polls) when possible, else cumulative.
    latency_sum: dict[str, float] = field(default_factory=dict)
    latency_count: dict[str, float] = field(default_factory=dict)
    latency_mean_s: dict[str, float] = field(default_factory=dict)

    # Model info
    model_info: ModelInfo = field(default_factory=ModelInfo)


def _zero_history() -> deque[float]:
    """A fixed-size history deque pre-filled with zeros (so sparklines start flat)."""
    return deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)


@dataclass
class MetricsHistory:
    requests_running: deque[float] = field(default_factory=_zero_history)
    generation_tps: deque[float] = field(default_factory=_zero_history)
    gpu_cache: deque[float] = field(default_factory=_zero_history)


# Matches "metric_name{labels} value" or "metric_name value"; value may be a
# float, scientific notation, or NaN/±Inf.
_METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)"  # metric name + optional {labels}
    r"\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|NaN|[+-]?Inf)\s*$"  # value
)


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus text format into a flat metric name → value dict."""
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if m:
            name = m.group(1)
            try:
                value = float(m.group(2))
            except ValueError:
                continue
            # vLLM emits NaN/Inf for some gauges (e.g. an idle hit rate).
            # Drop them so they read as "no data" (0.0) instead of crashing
            # downstream math (e.g. int(NaN) in sparkline()).
            if math.isfinite(value):
                result[name] = value
    return result


def _get_gauge(raw: dict[str, float], *keys: str) -> float:
    for k in keys:
        if k in raw:
            return raw[k]
        # Also try with labels
        for rk in raw:
            if rk.startswith(k + "{") or rk == k:
                return raw[rk]
    return 0.0


_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


def _parse_labels(key: str) -> dict[str, str]:
    """Parse the ``{name="v",...}`` label set of a single metric key into a dict."""
    return dict(_LABEL_RE.findall(key))


@cache
def _label_pattern(label: str) -> re.Pattern[str]:
    # Anchored on a label boundary ({ or ,) so e.g. ``block_size`` does not
    # match inside ``hash_block_size``. Cached so it compiles once per label.
    return re.compile(r'[{,]' + re.escape(label) + r'="([^"]*)"')


def _extract_label(raw: dict[str, float], label: str) -> str | None:
    """Return the first value of `label` found across all metric keys."""
    pat = _label_pattern(label)
    for key in raw:
        m = pat.search(key)
        if m and m.group(1):
            return m.group(1)
    return None


class MetricsPoller:
    def __init__(self, base_url: str, api_key: str | None = None, interval: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.interval = interval
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=5.0)
        self._prev_metrics: VllmMetrics | None = None
        self.history = MetricsHistory()
        # Cumulative token/time accumulators for average throughput.
        # Only ticks with actual traffic contribute, so idle periods
        # and polling-frequency changes don't distort the average.
        self._cumul_tokens: float = 0.0
        self._cumul_seconds: float = 0.0
        self._last_avg_tps: float = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def poll(self) -> VllmMetrics:
        m = VllmMetrics(timestamp=time.time())
        try:
            prom_text = await self._fetch_prometheus()
            m.server_reachable = True
            # Model name and cache config come from the /metrics labels — always
            # current and auth-free — so a model change (e.g. a vLLM restart) is
            # reflected on the next poll, with no /v1/models request spam.
            self._parse_into(m, prom_text)
            self._compute_rates(m)
            self._update_avg_tps(m)
        except Exception:
            m.server_reachable = False

        m.avg_gen_tokens_per_sec = self._last_avg_tps
        self._update_history(m)
        self._prev_metrics = m
        return m

    async def _fetch_prometheus(self) -> str:
        resp = await self._client.get(f"{self.base_url}/metrics")
        resp.raise_for_status()
        return resp.text

    def _parse_into(self, m: VllmMetrics, text: str) -> None:
        raw = _parse_prometheus(text)

        m.num_requests_running = _get_gauge(raw, "vllm:num_requests_running")
        m.num_requests_waiting = _get_gauge(raw, "vllm:num_requests_waiting")
        m.num_requests_swapped = _get_gauge(raw, "vllm:num_requests_swapped")

        # Single pass over every series: sum counters across label sets and
        # accumulate histogram _sum/_count. Derived breakdowns (e.g.
        # *_by_source_total) are excluded by matching exact base names so they
        # aren't double-counted; histogram _bucket lines fall through untouched.
        prompt_total = 0.0
        gen_total = 0.0
        success_total = 0.0
        prefix_queries = 0.0
        prefix_hits = 0.0
        spec_draft = 0.0
        spec_accepted = 0.0
        spec_drafts = 0.0
        spec_present = False
        preemptions = 0.0
        finished: dict[str, float] = {}
        hist_sum: dict[str, float] = {}
        hist_count: dict[str, float] = {}
        cache_info_key: str | None = None
        for k, v in raw.items():
            name = k.split("{", 1)[0]
            if name == "vllm:prompt_tokens_total":
                prompt_total += v
            elif name == "vllm:generation_tokens_total":
                gen_total += v
            elif name == "vllm:request_success_total":
                success_total += v
                rm = re.search(r'finished_reason="([^"]*)"', k)
                reason = rm.group(1) if rm else "unknown"
                finished[reason] = finished.get(reason, 0.0) + v
            elif name == "vllm:num_preemptions_total":
                preemptions += v
            elif name == "vllm:prefix_cache_queries_total":
                prefix_queries += v
            elif name == "vllm:prefix_cache_hits_total":
                prefix_hits += v
            elif name == "vllm:spec_decode_num_draft_tokens_total":
                spec_draft += v
                spec_present = True
            elif name == "vllm:spec_decode_num_accepted_tokens_total":
                spec_accepted += v
            elif name == "vllm:spec_decode_num_drafts_total":
                spec_drafts += v
            elif name == "vllm:cache_config_info":
                cache_info_key = k
            elif name.endswith("_sum"):
                base = name[:-4]
                if base in _HIST_BASES:
                    hist_sum[base] = hist_sum.get(base, 0.0) + v
            elif name.endswith("_count"):
                base = name[:-6]
                if base in _HIST_BASES:
                    hist_count[base] = hist_count.get(base, 0.0) + v
        m.prompt_tokens_total = prompt_total
        m.generation_tokens_total = gen_total
        m.request_success_total = success_total
        m.num_preemptions_total = preemptions
        m.finished_reasons = finished
        m.prefix_cache_queries_total = prefix_queries
        m.prefix_cache_hits_total = prefix_hits

        # Speculative decoding. acceptance = accepted / drafted tokens (%);
        # accept length = accepted tokens per draft step (the MTP speedup).
        m.spec_decode_active = spec_present
        m.spec_draft_tokens_total = spec_draft
        m.spec_accepted_tokens_total = spec_accepted
        m.spec_drafts_total = spec_drafts
        if spec_present and spec_draft > 0:
            m.spec_acceptance_rate = spec_accepted / spec_draft * 100
        if spec_present and spec_drafts > 0:
            m.spec_accept_length = spec_accepted / spec_drafts

        # KV cache usage: vLLM v1 renamed gpu_cache_usage_perc → kv_cache_usage_perc.
        m.gpu_cache_usage_perc = (
            _get_gauge(raw, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc") * 100
        )
        m.cpu_cache_usage_perc = _get_gauge(raw, "vllm:cpu_cache_usage_perc") * 100

        # Prefix hit rate: prefer the legacy gauge; otherwise derive from counters.
        legacy_prefix = _get_gauge(raw, "vllm:gpu_prefix_cache_hit_rate")
        if legacy_prefix > 0:
            m.gpu_prefix_cache_hit_rate = legacy_prefix * 100
        elif prefix_queries > 0:
            m.gpu_prefix_cache_hit_rate = prefix_hits / prefix_queries * 100

        # Latency histograms — keep raw sum/count; recent means derived in
        # _compute_rates. Absent histograms simply never get a key.
        for name, key in _LATENCY_BY_METRIC.items():
            hcount = hist_count.get(name, 0.0)
            if hcount > 0:
                hsum = hist_sum.get(name, 0.0)
                mean = hsum / hcount
                m.latency_sum[key] = hsum
                m.latency_count[key] = hcount
                m.latency_mean_s[key] = mean
                if key == "e2e":
                    m.avg_e2e_latency_s = mean

        # Average per-request token counts (cumulative mean from the
        # per-request histograms). Bases are exact, so request_generation_tokens
        # is not confused with request_max_num_generation_tokens.
        for name, attr in _TOKEN_HISTOGRAMS.items():
            hcount = hist_count.get(name, 0.0)
            if hcount > 0:
                setattr(m, attr, hist_sum.get(name, 0.0) / hcount)

        # Model name: read it from the model_name label every poll (always
        # current, no auth), so a model change after a restart is picked up.
        info = m.model_info
        if info.model_id in ("", "unknown"):
            label_model = _extract_label(raw, "model_name")
            if label_model:
                info.model_id = label_model

        # Enrich from the single vllm:cache_config_info series (absent → None).
        if cache_info_key:
            labels = _parse_labels(cache_info_key)
            blocks = labels.get("num_gpu_blocks")
            if blocks and blocks.isdigit():
                info.num_gpu_blocks = int(blocks)
            block_size = labels.get("block_size")
            if block_size and block_size.isdigit():
                info.block_size = int(block_size)
            util = labels.get("gpu_memory_utilization")
            if util:
                try:
                    info.gpu_memory_utilization = float(util)
                except ValueError:
                    pass
            dtype = labels.get("cache_dtype")
            if dtype and dtype != "None":
                info.cache_dtype = dtype

    def _compute_rates(self, current: VllmMetrics) -> None:
        prev = self._prev_metrics
        if prev is None or not prev.server_reachable:
            return
        dt = current.timestamp - prev.timestamp
        if dt <= 0:
            return
        current.prompt_tokens_per_sec = max(
            0.0, (current.prompt_tokens_total - prev.prompt_tokens_total) / dt
        )
        current.generation_tokens_per_sec = max(
            0.0, (current.generation_tokens_total - prev.generation_tokens_total) / dt
        )

        # Recent mean latency per histogram: change in sum / change in count
        # since the last poll. Falls back to the cumulative mean from _parse_into.
        for key in LATENCY_HISTOGRAMS:
            d_count = current.latency_count.get(key, 0.0) - prev.latency_count.get(key, 0.0)
            d_sum = current.latency_sum.get(key, 0.0) - prev.latency_sum.get(key, 0.0)
            if d_count > 0 and d_sum >= 0:
                current.latency_mean_s[key] = d_sum / d_count

    def _update_history(self, m: VllmMetrics) -> None:
        self.history.requests_running.append(m.num_requests_running)
        self.history.generation_tps.append(m.generation_tokens_per_sec)
        self.history.gpu_cache.append(m.gpu_cache_usage_perc)

    def _update_avg_tps(self, current: VllmMetrics) -> None:
        """Update running average generation throughput from per-tick deltas.

        Only ticks where new generation tokens were processed contribute to
        the cumulative time, so idle periods and polling-frequency changes
        don't distort the average. A system clock jump that produces a
        non-positive dt is silently skipped (the average stalls until a
        future tick with a valid dt).
        """
        prev = self._prev_metrics
        if prev is None or not prev.server_reachable:
            return
        dt = current.timestamp - prev.timestamp
        if dt <= 0:
            return
        
        dt_gen = current.generation_tokens_total - prev.generation_tokens_total
        if dt_gen > 0:
            self._cumul_tokens += dt_gen
            self._cumul_seconds += dt
            if self._cumul_seconds > 0:
                self._last_avg_tps = self._cumul_tokens / self._cumul_seconds


def bar_chart(values: deque[float], width: int = 20, height: int = 4) -> list[str]:
    """Render a multi-row vertical bar chart from a deque of floats.

    Returns `height` strings (top row first), each `min(width, len)` wide.
    Bars are scaled to the window's peak using eighth-block resolution.
    """
    blocks = " ▁▂▃▄▅▆▇█"  # index 0..8 eighths of a cell
    # Treat non-finite samples (NaN/Inf) as 0 so int() below can't raise.
    samples = [v if math.isfinite(v) else 0.0 for v in list(values)[-width:]]
    if not samples:
        samples = [0.0]
    max_val = max(samples) or 1.0
    grid = [[" "] * len(samples) for _ in range(height)]
    for col, v in enumerate(samples):
        eighths = int(round(max(0.0, v) / max_val * height * 8))
        eighths = min(eighths, height * 8)
        full, rem = divmod(eighths, 8)
        for r in range(full):
            grid[height - 1 - r][col] = "█"
        if rem and full < height:
            grid[height - 1 - full][col] = blocks[rem]
    return ["".join(row) for row in grid]


_BLOCKS = " ▁▂▃▄▅▆▇█"  # index == number of eighths filled (see bar_chart)


def _style_bar_row(row: str, fill: str) -> str:
    """Wrap a bar row in Textual markup so it renders as a solid fill.

    Every filled cell becomes a background-colored space rather than a
    foreground block glyph. Textual exports a run of same-background cells as a
    single ``<rect>`` (with ``shape-rendering="crispEdges"``), so the fill
    tiles seamlessly in *any* SVG renderer. Foreground block glyphs can't:
    the export uses a 20px font on a 24.4px line, so a ``█`` glyph leaves a
    ~4px gap below it (a broken-grid seam), and a partial eighth-block leaves
    the same gap between its crest and the bar body — plus it renders as tofu
    in browsers whose fallback monospace lacks the block characters. So we snap
    each cell to full-or-empty (half-cell-or-more rounds up) and emit no
    foreground glyphs at all; the chart loses sub-cell smoothing but renders
    identically everywhere.
    """
    out: list[str] = []
    run = 0  # consecutive filled cells pending a flush

    def flush() -> None:
        nonlocal run
        if run:
            out.append(f"[on {fill}]{' ' * run}[/]")
            run = 0

    for ch in row:
        # Round a cell up to a full block when it's at least half filled.
        if ch == "█" or _BLOCKS.find(ch) >= 4:
            run += 1
        else:
            flush()
            out.append(" ")
    flush()
    return "".join(out)


def render_spark(
    values: deque[float],
    content_width: int,
    fmt: Callable[[float], str],
    height: int,
    fill: str = "$success",
) -> list[str]:
    """Render a labeled bar chart; each line is ``"<axis>│<bars>"``.

    `fmt` formats the axis numbers (peak at the top row, ``0`` at the bottom).
    `fill` is the Textual color (name, hex, or ``$variable``) the bars are
    painted with; the bar cells carry inline markup so they export as solid
    rects (see `_style_bar_row`).

    The y-axis peak is derived from the same trailing window ``bar_chart``
    draws (the last ``chart_w`` samples), *not* the whole deque — otherwise a
    spike that has scrolled off the visible chart keeps inflating the "max"
    label until it ages out of ``HISTORY_SIZE``. The gutter, however, is sized
    off the full-history peak so the chart width doesn't jitter as values move.
    """
    finite_all = [v for v in values if math.isfinite(v)]
    provisional = fmt(max(finite_all) if finite_all else 0.0)
    gutter = max(len(provisional), 1)
    # Fit the chart to the available width, leaving room for the axis.
    avail = max(content_width, gutter + 9)
    chart_w = max(8, avail - gutter - 1)
    window = [v for v in list(values)[-chart_w:] if math.isfinite(v)]
    peak = max(window) if window else 0.0
    top = fmt(peak)
    rows = bar_chart(values, chart_w, height)
    lines = []
    for i, row in enumerate(rows):
        if i == 0:
            axis = top.rjust(gutter)
        elif i == len(rows) - 1:
            axis = "0".rjust(gutter)
        else:
            axis = " " * gutter
        lines.append(f"{axis}│{_style_bar_row(row, fill)}")
    return lines
