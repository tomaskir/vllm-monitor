# CLAUDE.md

Guidance for working in this repository.

## What this is

`vllm-monitor` is a real-time terminal UI dashboard for monitoring a [vLLM](https://github.com/vllm-project/vllm) server. It polls the server's Prometheus `/metrics` endpoint and `/v1/models`, then renders live request, throughput, cache, and GPU-memory stats — plus rolling sparklines — in a [Textual](https://textual.textualize.io/) TUI. The goal is a zero-Grafana, single-command health view of a running vLLM instance.

It is a Python 3.10+ package published to PyPI; the console entry point `vllm-monitor` maps to `vllm_monitor.cli:app`.

## Layout

```
src/vllm_monitor/
  cli.py        # Typer CLI: parses --url/--interval/--api-key, builds poller + app
  metrics.py    # MetricsPoller, dataclasses, Prometheus parsing, rate/sparkline logic
  app.py        # Textual App + widgets (MetricCard, SparklineCard, ModelInfoPanel)
tests/
  test_metrics.py   # parsing, rate computation, sparkline tests
```

The three modules form a clean pipeline: `cli` wires things up → `metrics` fetches/parses → `app` renders. `metrics.py` has no Textual dependency and is the unit-tested core.

## Architecture notes

- **`MetricsPoller.poll()`** (async) fetches `/metrics` (Prometheus text) and `/v1/models` each tick, parses into a fresh `VllmMetrics`, computes per-second token rates from the *delta* against the previous poll (`_prev_metrics`), and appends to a fixed-size (`HISTORY_SIZE=60`) `MetricsHistory` of deques for sparklines. Any fetch exception sets `server_reachable = False` rather than crashing.
- **Prometheus parsing** is regex-based (`_parse_prometheus`) — no prometheus client dependency. Metric names keep their `{labels}`. `_get_gauge` looks up by name (with or without labels, trying each fallback name in order); counters like `vllm:prompt_tokens_total` are *summed across label sets by exact base name* (matching the bare name, not a substring, so derived breakdowns like `*_by_source_total` aren't double-counted). Percentage gauges from vLLM are 0–1 and get multiplied by 100 on parse.
- **vLLM schema drift**: this targets the **vLLM v1 engine** (verified against a live `deepseek-v4-flash` server). Key differences from older vLLM that the parser handles: KV cache is `vllm:kv_cache_usage_perc` (was `gpu_cache_usage_perc`); there is no prefix-hit-rate gauge — it's derived from `prefix_cache_hits_total / prefix_cache_queries_total`; `gpu_memory_*_bytes` is **not emitted at all** (the GPU Memory card shows `—`); `/v1/models` may return 401, so the model name falls back to the `model_name` metric label via `_extract_label`. When pointing at a different server, `curl $URL/metrics | grep '^# TYPE'` first and reconcile names in `_parse_into`.
- **`VllmMonitorApp`** (`app.py`) drives the UI via `set_interval(self._interval, self._tick)`. Each tick calls `poll()` then `_update_ui()`, which writes Rich markup into the cards. Alert coloring (`_color_pct`) is green/yellow/red against `GPU_*_WARN`/`CRIT` constants.
- Widgets are defined inline with `DEFAULT_CSS`; the app's `CSS` only handles layout rows. Cards are looked up by `id` (e.g. `#card-running`) in `_update_ui`.

## Commands

```bash
pip install -e ".[dev]"        # dev install
pytest                          # run tests (asyncio_mode=auto, set in pyproject)
ruff check src/ tests/          # lint (E, F, I, UP; line-length 100)
mypy src/                       # type-check (deps declared but no config block)
vllm-monitor --url http://localhost:8000 --interval 2   # run against a server
```

There is currently no CI; run `ruff check` and `pytest` locally before pushing. The package targets Python 3.10/3.11/3.12. The local default branch is `master`.

## Conventions

- `from __future__ import annotations` at the top of every module; type hints throughout.
- Dataclasses for all data shapes (`VllmMetrics`, `ModelInfo`, `MetricsHistory`).
- The TUI cannot be exercised in tests; keep parsing/rate/render-helper logic in `metrics.py` so it stays unit-testable, and keep `app.py` a thin rendering layer.
- When adding a metric: add a field to `VllmMetrics`, parse it in `_parse_into`, add a card in `compose()`, and update it in `_update_ui()`. Add a sample line to `SAMPLE_PROMETHEUS` in the test and assert on it.
