"""Metrics polling and parsing for vLLM server."""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class ModelInfo:
    model_id: str = "unknown"
    max_model_len: Optional[int] = None
    tensor_parallel_size: Optional[int] = None
    # From vllm:cache_config_info labels (when present)
    num_gpu_blocks: Optional[int] = None
    block_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    cache_dtype: Optional[str] = None


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

    # Average per-request token counts (from request_*_tokens histograms)
    avg_prompt_tokens: float = 0.0
    avg_generation_tokens: float = 0.0

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


@dataclass
class MetricsHistory:
    requests_running: deque[float] = field(default_factory=lambda: deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE))
    generation_tps: deque[float] = field(default_factory=lambda: deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE))
    gpu_cache: deque[float] = field(default_factory=lambda: deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE))


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus text format into a flat metric name → value dict."""
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Match metric_name{labels} value or metric_name value
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|NaN|[+-]?Inf)\s*$', line)
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


def _hist_sum_count(raw: dict[str, float], name: str) -> tuple[float, float]:
    """Sum a histogram's _sum and _count series across all label sets."""
    total_sum = 0.0
    total_count = 0.0
    for k, v in raw.items():
        if f"{name}_sum" in k:
            total_sum += v
        elif f"{name}_count" in k:
            total_count += v
    return total_sum, total_count


def _extract_label(raw: dict[str, float], label: str) -> Optional[str]:
    """Return the first value of `label` found across all metric keys.

    Anchored on a label boundary ({ or ,) so e.g. ``block_size`` does not
    match inside ``hash_block_size``.
    """
    pat = re.compile(r'[{,]' + re.escape(label) + r'="([^"]*)"')
    for key in raw:
        m = pat.search(key)
        if m and m.group(1):
            return m.group(1)
    return None


class MetricsPoller:
    def __init__(self, base_url: str, api_key: Optional[str] = None, interval: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.interval = interval
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=5.0)
        self._prev_metrics: Optional[VllmMetrics] = None
        self._prev_time: float = 0.0
        self.history = MetricsHistory()

    async def close(self) -> None:
        await self._client.aclose()

    async def poll(self) -> VllmMetrics:
        m = VllmMetrics(timestamp=time.time())
        try:
            prom_text = await self._fetch_prometheus()
            model_info = await self._fetch_model_info()
            m.server_reachable = True
            m.model_info = model_info
            self._parse_into(m, prom_text)
            self._compute_rates(m)
        except Exception:
            m.server_reachable = False

        self._update_history(m)
        self._prev_metrics = m
        self._prev_time = m.timestamp
        return m

    async def _fetch_prometheus(self) -> str:
        resp = await self._client.get(f"{self.base_url}/metrics")
        resp.raise_for_status()
        return resp.text

    async def _fetch_model_info(self) -> ModelInfo:
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                first = models[0]
                info = ModelInfo(model_id=first.get("id", "unknown"))
                perms = first.get("permission", [{}])
                if perms:
                    pass  # vLLM doesn't always expose context_length here
                return info
        except Exception:
            pass
        return ModelInfo()

    def _parse_into(self, m: VllmMetrics, text: str) -> None:
        raw = _parse_prometheus(text)

        m.num_requests_running = _get_gauge(raw, "vllm:num_requests_running")
        m.num_requests_waiting = _get_gauge(raw, "vllm:num_requests_waiting")
        m.num_requests_swapped = _get_gauge(raw, "vllm:num_requests_swapped")

        # Sum across all model labels for token / counter totals.
        # Exclude derived breakdowns (e.g. *_by_source_total) that would double-count.
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
        for key, name in LATENCY_HISTOGRAMS.items():
            hsum, hcount = _hist_sum_count(raw, name)
            if hcount > 0:
                m.latency_sum[key] = hsum
                m.latency_count[key] = hcount
                m.latency_mean_s[key] = hsum / hcount

        # Average per-request token counts (cumulative mean from the
        # per-request histograms). Names are exact, so request_generation_tokens
        # is not confused with request_max_num_generation_tokens.
        psum, pcount = _hist_sum_count(raw, "vllm:request_prompt_tokens")
        if pcount > 0:
            m.avg_prompt_tokens = psum / pcount
        gsum, gcount = _hist_sum_count(raw, "vllm:request_generation_tokens")
        if gcount > 0:
            m.avg_generation_tokens = gsum / gcount

        # Model name fallback: when /v1/models is unauthorized, every metric
        # carries a model_name="..." label we can read instead.
        info = m.model_info
        if info.model_id in ("", "unknown"):
            label_model = _extract_label(raw, "model_name")
            if label_model:
                info.model_id = label_model

        # Enrich from vllm:cache_config_info labels (absent → stays None).
        blocks = _extract_label(raw, "num_gpu_blocks")
        if blocks and blocks.isdigit():
            info.num_gpu_blocks = int(blocks)
        block_size = _extract_label(raw, "block_size")
        if block_size and block_size.isdigit():
            info.block_size = int(block_size)
        util = _extract_label(raw, "gpu_memory_utilization")
        if util:
            try:
                info.gpu_memory_utilization = float(util)
            except ValueError:
                pass
        dtype = _extract_label(raw, "cache_dtype")
        if dtype and dtype != "None":
            info.cache_dtype = dtype

    def _compute_rates(self, current: VllmMetrics) -> None:
        if self._prev_metrics is None or not self._prev_metrics.server_reachable:
            return
        dt = current.timestamp - self._prev_metrics.timestamp
        if dt <= 0:
            return
        current.prompt_tokens_per_sec = max(0.0, (current.prompt_tokens_total - self._prev_metrics.prompt_tokens_total) / dt)
        current.generation_tokens_per_sec = max(0.0, (current.generation_tokens_total - self._prev_metrics.generation_tokens_total) / dt)

        # Recent mean latency per histogram: change in sum / change in count
        # since the last poll. Falls back to the cumulative mean from _parse_into.
        for key in LATENCY_HISTOGRAMS:
            d_count = current.latency_count.get(key, 0.0) - self._prev_metrics.latency_count.get(key, 0.0)
            d_sum = current.latency_sum.get(key, 0.0) - self._prev_metrics.latency_sum.get(key, 0.0)
            if d_count > 0 and d_sum >= 0:
                current.latency_mean_s[key] = d_sum / d_count

    def _update_history(self, m: VllmMetrics) -> None:
        self.history.requests_running.append(m.num_requests_running)
        self.history.generation_tps.append(m.generation_tokens_per_sec)
        self.history.gpu_cache.append(m.gpu_cache_usage_perc)


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
