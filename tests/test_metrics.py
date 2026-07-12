"""Tests for metrics parsing."""

from __future__ import annotations

from collections import deque

import pytest

from vllm_monitor.metrics import (
    MetricsPoller,
    VllmMetrics,
    _parse_prometheus,
    bar_chart,
    render_spark,
)

# Mirrors the schema emitted by the vLLM v1 engine (engine/model labels,
# kv_cache_usage_perc, prefix cache counters, no /v1/models needed for the name).
SAMPLE_PROMETHEUS = """
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="deepseek-v4-flash"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="deepseek-v4-flash"} 7.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="deepseek-v4-flash"} 0.42
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="deepseek-v4-flash"} 1000.0
# HELP vllm:prefix_cache_hits_total Prefix cache hits, in cached tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="deepseek-v4-flash"} 750.0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="deepseek-v4-flash"} 12000.0
# HELP vllm:prompt_tokens_by_source_total Number of prompt tokens by source.
# TYPE vllm:prompt_tokens_by_source_total counter
vllm:prompt_tokens_by_source_total{model_name="deepseek-v4-flash",source="local_compute"} 12000.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="deepseek-v4-flash"} 45000.0
# HELP vllm:e2e_request_latency_seconds Histogram of e2e request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="deepseek-v4-flash"} 320.5
vllm:e2e_request_latency_seconds_count{engine="0",model_name="deepseek-v4-flash"} 200.0
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{engine="0",model_name="deepseek-v4-flash"} 50.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="deepseek-v4-flash"} 200.0
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_sum{engine="0",model_name="deepseek-v4-flash"} 20.0
vllm:request_queue_time_seconds_count{engine="0",model_name="deepseek-v4-flash"} 200.0
"""


def test_parse_prometheus_basic():
    raw = _parse_prometheus(SAMPLE_PROMETHEUS)
    assert any("num_requests_running" in k for k in raw)
    assert any("kv_cache_usage_perc" in k for k in raw)


def test_parse_into_metrics():
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, SAMPLE_PROMETHEUS)

    assert m.num_requests_running == pytest.approx(3.0)
    assert m.num_requests_waiting == pytest.approx(7.0)
    assert m.gpu_cache_usage_perc == pytest.approx(42.0)
    # Hit rate derived from counters: 750 / 1000 = 75%.
    assert m.gpu_prefix_cache_hit_rate == pytest.approx(75.0)
    # _by_source_total must not be double-counted into the prompt total.
    assert m.prompt_tokens_total == pytest.approx(12000.0)
    assert m.generation_tokens_total == pytest.approx(45000.0)
    assert m.latency_mean_s["e2e"] == pytest.approx(320.5 / 200.0)
    # Other latency histograms parsed; an absent one (tpot) stays unset.
    assert m.latency_mean_s["ttft"] == pytest.approx(50.0 / 200.0)
    assert m.latency_mean_s["queue"] == pytest.approx(20.0 / 200.0)
    assert "tpot" not in m.latency_mean_s
    # Model name recovered from metric labels (no /v1/models call).
    assert m.model_info.model_id == "deepseek-v4-flash"


def test_rate_computation():
    poller = MetricsPoller(base_url="http://localhost:8000")

    prev = VllmMetrics(
        timestamp=0.0,
        server_reachable=True,
        generation_tokens_total=1000.0,
        prompt_tokens_total=500.0,
    )
    poller._prev_metrics = prev

    current = VllmMetrics(
        timestamp=2.0,
        server_reachable=True,
        generation_tokens_total=1200.0,
        prompt_tokens_total=600.0,
    )
    poller._compute_rates(current)

    assert current.generation_tokens_per_sec == pytest.approx(100.0)
    assert current.prompt_tokens_per_sec == pytest.approx(50.0)


def test_cache_config_enrichment():
    # block_size must not be confused with hash_block_size / mamba_block_size.
    text = (
        'vllm:cache_config_info{_block_size_resolved="True",block_size="256",'
        'cache_dtype="fp8",gpu_memory_utilization="0.92",hash_block_size="None",'
        'num_gpu_blocks="94671",mamba_block_size="None"} 1.0\n'
    )
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, text)
    assert m.model_info.block_size == 256
    assert m.model_info.num_gpu_blocks == 94671
    assert m.model_info.gpu_memory_utilization == pytest.approx(0.92)
    assert m.model_info.cache_dtype == "fp8"


def test_cache_config_absent_leaves_none():
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, 'vllm:num_requests_running{model_name="m"} 1.0')
    assert m.model_info.num_gpu_blocks is None
    assert m.model_info.cache_dtype is None


def test_avg_request_tokens():
    # request_generation_tokens must not be confused with the decoy
    # request_max_num_generation_tokens histogram.
    text = (
        'vllm:request_prompt_tokens_sum{model_name="m"} 10000.0\n'
        'vllm:request_prompt_tokens_count{model_name="m"} 100.0\n'
        'vllm:request_generation_tokens_sum{model_name="m"} 5000.0\n'
        'vllm:request_generation_tokens_count{model_name="m"} 100.0\n'
        'vllm:request_max_num_generation_tokens_sum{model_name="m"} 99999.0\n'
        'vllm:request_max_num_generation_tokens_count{model_name="m"} 100.0\n'
    )
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, text)
    assert m.avg_prompt_tokens == pytest.approx(100.0)
    assert m.avg_generation_tokens == pytest.approx(50.0)  # not 999.99


def test_preemptions_and_finish_reasons():
    text = (
        'vllm:num_preemptions_total{engine="0",model_name="m"} 5.0\n'
        'vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 10.0\n'
        'vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 4.0\n'
        'vllm:request_success_total{engine="0",finished_reason="error",model_name="m"} 1.0\n'
    )
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, text)
    assert m.num_preemptions_total == pytest.approx(5.0)
    assert m.finished_reasons == {
        "stop": pytest.approx(10.0),
        "length": pytest.approx(4.0),
        "error": pytest.approx(1.0),
    }
    assert m.request_success_total == pytest.approx(15.0)


def test_spec_decode_acceptance():
    text = (
        'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 500.0\n'
        'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 1000.0\n'
        'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 700.0\n'
    )
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, text)
    assert m.spec_decode_active is True
    assert m.spec_acceptance_rate == pytest.approx(70.0)  # 700 / 1000 tokens
    assert m.spec_accept_length == pytest.approx(1.4)  # 700 / 500 drafts


def test_spec_decode_absent_is_inactive():
    # MTP/spec-decode off → metrics absent → inactive, no crash, no rate.
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, 'vllm:num_requests_running{model_name="m"} 1.0')
    assert m.spec_decode_active is False
    assert m.spec_acceptance_rate == 0.0


def test_model_name_from_metric_label():
    # No /v1/models call — the name comes from the metric label and updates
    # whenever the served model changes.
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, 'vllm:num_requests_running{model_name="new-model"} 1.0')
    assert m.model_info.model_id == "new-model"


def test_poll_makes_no_v1_models_request():
    # The poller must only ever hit /metrics (no /v1/models spam).
    poller = MetricsPoller(base_url="http://localhost:8000")
    assert not hasattr(poller, "_fetch_model_info")
    assert not hasattr(poller, "_resolve_model_info")


def test_bar_chart_dimensions():
    rows = bar_chart(deque([1, 2, 3], maxlen=60), width=10, height=4)
    assert len(rows) == 4  # height rows
    assert all(len(r) == 3 for r in rows)  # one column per sample (capped at width)


def test_bar_chart_empty():
    rows = bar_chart(deque(maxlen=60), width=10, height=4)
    assert len(rows) == 4
    assert all(set(r) <= {" "} for r in rows)  # nothing to plot → blank


def test_bar_chart_scaling():
    # 0 → empty column; peak → full column (every row filled, incl. the top).
    rows = bar_chart(deque([0.0, 4.0], maxlen=60), width=2, height=3)
    assert all(r[0] == " " for r in rows)  # zero column blank top to bottom
    assert all(r[1] == "█" for r in rows)  # peak column full top to bottom


def test_parse_drops_non_finite():
    # vLLM can emit NaN/Inf for idle gauges; these must not enter the dict.
    text = (
        'vllm:kv_cache_usage_perc{model_name="m"} NaN\n'
        'vllm:num_requests_running{model_name="m"} +Inf\n'
        'vllm:num_requests_waiting{model_name="m"} 4.0\n'
    )
    raw = _parse_prometheus(text)
    assert not any("kv_cache_usage_perc" in k for k in raw)
    assert not any("num_requests_running" in k for k in raw)
    assert raw['vllm:num_requests_waiting{model_name="m"}'] == pytest.approx(4.0)


def test_bar_chart_handles_non_finite():
    # NaN/Inf samples must not raise (int(NaN) would).
    rows = bar_chart(deque([0.0, float("nan"), float("inf"), 2.0], maxlen=60), width=4, height=3)
    assert len(rows) == 3
    assert all(len(r) == 4 for r in rows)


def test_render_spark_dimensions():
    lines = render_spark(
        deque([1.0, 2.0, 3.0], maxlen=60), content_width=30, fmt=lambda v: f"{v:.1f}", height=4
    )
    assert len(lines) == 4  # one line per chart row (height)
    assert all("│" in ln for ln in lines)  # axis gutter + separator on every line


def test_render_spark_axis_labels():
    lines = render_spark(
        deque([0.0, 5.0, 10.0], maxlen=60), content_width=40, fmt=lambda v: f"{v:.0f}", height=4
    )
    assert lines[0].split("│")[0].strip() == "10"  # peak labels the top row
    assert lines[-1].split("│")[0].strip() == "0"  # zero labels the bottom row
    gutters = {len(ln.split("│")[0]) for ln in lines}
    assert len(gutters) == 1  # every row shares the same gutter width


def test_render_spark_axis_tracks_visible_window():
    # Regression: a spike that has scrolled out of the visible window must not
    # keep inflating the axis "max" — the peak reflects only the drawn slice.
    values = deque([100.0] + [2.0] * 20, maxlen=60)
    lines = render_spark(values, content_width=20, fmt=lambda v: f"{v:.0f}", height=4)
    axis = lines[0].split("│")[0]
    assert axis.strip() == "2"  # visible-window peak, not the off-screen 100
    assert len(axis) == 3  # gutter still sized to full-history peak ("100") → no jitter


def test_render_spark_all_zero_axis():
    # All-zero history (the startup state) must not blow up or mislabel.
    lines = render_spark(
        deque([0.0] * 60, maxlen=60), content_width=30, fmt=lambda v: f"{v:.0f}", height=4
    )
    assert lines[0].split("│")[0].strip() == "0"
    assert lines[-1].split("│")[0].strip() == "0"


def _run_avg_tps(poller: MetricsPoller, current: VllmMetrics) -> None:
    """Run the real _update_avg_tps method and assign the result."""
    poller._update_avg_tps(current)
    current.avg_gen_tokens_per_sec = poller._last_avg_tps


def test_avg_tps_first_poll_no_prev():
    """First poll: _prev_metrics is None → avg stays 0.0."""
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics(
        timestamp=10.0, server_reachable=True,
        prompt_tokens_total=1000.0, generation_tokens_total=5000.0,
    )
    _run_avg_tps(poller, m)
    assert m.avg_gen_tokens_per_sec == 0.0
    assert poller._cumul_tokens == 0.0
    assert poller._cumul_seconds == 0.0


def test_avg_tps_single_active_tick():
    """One active tick → avg = gen_delta / dt."""
    poller = MetricsPoller(base_url="http://localhost:8000")
    poller._prev_metrics = VllmMetrics(
        timestamp=0.0, server_reachable=True,
        prompt_tokens_total=500.0, generation_tokens_total=1000.0,
    )
    m = VllmMetrics(
        timestamp=2.0, server_reachable=True,
        prompt_tokens_total=600.0, generation_tokens_total=1200.0,
    )
    _run_avg_tps(poller, m)
    # gen delta = 1200-1000 = 200, dt = 2.0 → 100 tok/s
    assert m.avg_gen_tokens_per_sec == pytest.approx(100.0)
    assert poller._cumul_tokens == 200.0
    assert poller._cumul_seconds == 2.0


def test_avg_tps_idle_tick_no_change():
    """Idle tick (no new gen tokens) → avg unchanged, accumulators unchanged."""
    poller = MetricsPoller(base_url="http://localhost:8000")
    poller._prev_metrics = VllmMetrics(
        timestamp=2.0, server_reachable=True,
        prompt_tokens_total=600.0, generation_tokens_total=1200.0,
    )
    # First an active tick to establish a baseline avg
    poller._cumul_tokens = 200.0
    poller._cumul_seconds = 2.0
    poller._last_avg_tps = 100.0

    # Now an idle tick: same gen tokens (prompt may change but that's ignored)
    m = VllmMetrics(
        timestamp=4.0, server_reachable=True,
        prompt_tokens_total=700.0, generation_tokens_total=1200.0,
    )
    _run_avg_tps(poller, m)
    assert m.avg_gen_tokens_per_sec == pytest.approx(100.0)
    assert poller._cumul_tokens == 200.0
    assert poller._cumul_seconds == 2.0


def test_avg_tps_active_then_idle_then_active():
    """Active → idle → active: idle doesn't dilute, second burst is added."""
    poller = MetricsPoller(base_url="http://localhost:8000")
    poller._prev_metrics = VllmMetrics(
        timestamp=2.0, server_reachable=True,
        prompt_tokens_total=600.0, generation_tokens_total=1200.0,
    )
    # Active tick 1
    m1 = VllmMetrics(
        timestamp=4.0, server_reachable=True,
        prompt_tokens_total=700.0, generation_tokens_total=1500.0,
    )
    _run_avg_tps(poller, m1)
    # gen delta = 1500-1200 = 300, dt = 2.0 → 150 tok/s
    assert m1.avg_gen_tokens_per_sec == pytest.approx(150.0)

    # Advance prev to m1 for the next tick
    poller._prev_metrics = m1

    # Idle tick: no new gen tokens (prompt increased but that's ignored)
    m_idle = VllmMetrics(
        timestamp=6.0, server_reachable=True,
        prompt_tokens_total=800.0, generation_tokens_total=1500.0,
    )
    _run_avg_tps(poller, m_idle)
    assert m_idle.avg_gen_tokens_per_sec == pytest.approx(150.0)  # unchanged

    # Advance prev
    poller._prev_metrics = m_idle

    # Active tick 2: more gen tokens arrive
    m2 = VllmMetrics(
        timestamp=8.0, server_reachable=True,
        prompt_tokens_total=900.0, generation_tokens_total=1800.0,
    )
    _run_avg_tps(poller, m2)
    # cumulative gen delta = 300 + (1800-1500) = 300+300 = 600
    # cumulative seconds = 2.0 + 2.0 = 4.0 (idle tick not counted)
    # avg = 600/4.0 = 150.0
    assert m2.avg_gen_tokens_per_sec == pytest.approx(150.0)
    assert poller._cumul_tokens == 600.0
    assert poller._cumul_seconds == 4.0


def test_avg_tps_varying_interval():
    """Ticks with different dt are weighted by their actual duration."""
    poller = MetricsPoller(base_url="http://localhost:8000")
    poller._prev_metrics = VllmMetrics(
        timestamp=0.0, server_reachable=True,
        prompt_tokens_total=0.0, generation_tokens_total=0.0,
    )
    # Tick 1: dt=1.0, 100 gen tokens
    m1 = VllmMetrics(
        timestamp=1.0, server_reachable=True,
        prompt_tokens_total=50.0, generation_tokens_total=100.0,
    )
    _run_avg_tps(poller, m1)
    assert m1.avg_gen_tokens_per_sec == pytest.approx(100.0)

    poller._prev_metrics = m1

    # Tick 2: dt=3.0, 300 gen tokens (different interval)
    m2 = VllmMetrics(
        timestamp=4.0, server_reachable=True,
        prompt_tokens_total=200.0, generation_tokens_total=400.0,
    )
    _run_avg_tps(poller, m2)
    # cumulative gen delta = 100 + 300 = 400
    # cumulative seconds = 1.0 + 3.0 = 4.0
    # avg = 400/4.0 = 100.0
    assert m2.avg_gen_tokens_per_sec == pytest.approx(100.0)
    assert poller._cumul_seconds == 4.0


def test_avg_tps_server_down_does_not_corrupt():
    """Unreachable tick → accumulators unchanged, avg carries forward."""
    poller = MetricsPoller(base_url="http://localhost:8000")
    poller._prev_metrics = VllmMetrics(
        timestamp=2.0, server_reachable=True,
        prompt_tokens_total=600.0, generation_tokens_total=1200.0,
    )
    poller._cumul_tokens = 200.0
    poller._cumul_seconds = 2.0
    poller._last_avg_tps = 100.0

    # Server unreachable: server_reachable=False
    m = VllmMetrics(
        timestamp=4.0, server_reachable=False,
        prompt_tokens_total=600.0, generation_tokens_total=1200.0,
    )
    _run_avg_tps(poller, m)
    # prev is reachable but current is not → condition fails → avg unchanged
    assert m.avg_gen_tokens_per_sec == pytest.approx(100.0)
    assert poller._cumul_tokens == 200.0
    assert poller._cumul_seconds == 2.0


def test_avg_tps_prev_unreachable_skips():
    """Previous poll was unreachable → skip avg computation (no baseline)."""
    poller = MetricsPoller(base_url="http://localhost:8000")
    poller._prev_metrics = VllmMetrics(
        timestamp=2.0, server_reachable=False,
        prompt_tokens_total=600.0, generation_tokens_total=1200.0,
    )
    poller._cumul_tokens = 200.0
    poller._cumul_seconds = 2.0
    poller._last_avg_tps = 100.0

    m = VllmMetrics(
        timestamp=4.0, server_reachable=True,
        prompt_tokens_total=800.0, generation_tokens_total=1600.0,
    )
    _run_avg_tps(poller, m)
    # prev.server_reachable is False → condition fails
    assert m.avg_gen_tokens_per_sec == pytest.approx(100.0)
    assert poller._cumul_tokens == 200.0
    assert poller._cumul_seconds == 2.0
