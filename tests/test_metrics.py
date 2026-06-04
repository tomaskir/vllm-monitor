"""Tests for metrics parsing."""

from __future__ import annotations

from collections import deque

import pytest

from vllm_monitor.metrics import MetricsPoller, VllmMetrics, _parse_prometheus, bar_chart

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

    prev = VllmMetrics(timestamp=0.0, server_reachable=True, generation_tokens_total=1000.0, prompt_tokens_total=500.0)
    poller._prev_metrics = prev
    poller._prev_time = 0.0

    current = VllmMetrics(timestamp=2.0, server_reachable=True, generation_tokens_total=1200.0, prompt_tokens_total=600.0)
    poller._compute_rates(current)

    assert current.generation_tokens_per_sec == pytest.approx(100.0)
    assert current.prompt_tokens_per_sec == pytest.approx(50.0)


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
        'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 1000.0\n'
        'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 700.0\n'
    )
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, text)
    assert m.spec_decode_active is True
    assert m.spec_acceptance_rate == pytest.approx(70.0)


def test_spec_decode_absent_is_inactive():
    # MTP/spec-decode off → metrics absent → inactive, no crash, no rate.
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, 'vllm:num_requests_running{model_name="m"} 1.0')
    assert m.spec_decode_active is False
    assert m.spec_acceptance_rate == 0.0


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
