# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.3] - 2026-07-12

### Added
- The **Average Request** card (formerly *Avg Req Tokens*) now also shows the
  running average generation throughput (`tok/s`) and mean end-to-end latency,
  alongside the mean prompt/generation token counts. The running average counts
  only ticks with real traffic, so idle periods and polling-interval changes
  don't distort it. (#1)

## [1.0.2] - 2026-06-05

### Fixed
- A manual refresh (`r`) landing mid-poll no longer starts an overlapping
  request, which could add a stray history sample and a spurious rate spike.
- The HTTP client is now closed on exit.

### Changed
- Faster `/metrics` parsing (single pass instead of repeated dict rescans).

## [1.0.1] - 2026-06-05

### Fixed
- History chart y-axis "max" could stay pinned to a stale value. The axis peak
  was computed over the full history deque while the bars only render the last
  `chart_w` samples, so a spike that scrolled off the visible chart kept
  inflating the label until it aged out. The axis now scales to the same visible
  window the bars draw. The sparkline render logic was extracted into a
  unit-tested `render_spark` helper in `metrics.py`.

## [1.0.0] - 2026-06-04

First stable release. Builds on 0.2.0 with a much richer metric set, a
redesigned UI, environment-variable configuration, and a published Docker image.

### Added
- Latency detail cards — **TTFT**, **TPOT**, and **queue time** — alongside
  end-to-end latency (recent means, units auto-scaled to ms / s / m / h).
- **Speculative decoding (MTP)** card: acceptance rate and accept length
  (accepted tokens per draft step); shows `—` when spec decode is off.
- **Preemptions** card — a KV-cache-pressure signal.
- **Completed** card: total requests and tokens processed, with a finish-reason
  breakdown (truncated-by-length, errors).
- **Average request shape** — mean prompt vs generation tokens per request.
- **Model panel cache config** — KV dtype, GPU blocks, and GPU-memory-utilization
  target (from `vllm:cache_config_info`).
- Taller multi-row **history charts** with a y-axis scale.
- **Environment-variable configuration**: `VLLM_URL` and `VLLM_MONITOR_INTERVAL`
  (joining the existing `VLLM_API_KEY`).
- **Docker image** published to GitHub Container Registry on each release, with a
  GitHub Actions workflow that builds on every push/PR and publishes on `v*` tags.
- A rendered SVG dashboard screenshot in the README (`scripts/screenshot.py`).

### Changed
- Redesigned **sectioned, compact layout**: tile titles moved into their borders,
  tiles grouped under LOAD / LATENCY / THROUGHPUT & CACHE / STATS / HISTORY, and
  the model info moved to a framed header line.
- **Unified color palette** — green/yellow/red for status and thresholds, white
  for neutral values, cyan for the model name, dim for secondary text.
- The model name is now read from the `model_name` metric label every poll (so it
  follows model changes after a vLLM restart); the `/v1/models` request — which
  spammed unauthorized requests without an API key — was removed.
- Command palette restyled as a centered, outlined popup.
- Queued requests are green when empty and yellow once a backlog forms; the
  default header icon was removed.

### Removed
- GPU Memory card — the vLLM v1 engine doesn't expose live GPU VRAM
  (`vllm:gpu_memory_*_bytes`).

### Fixed
- Two-line metric values (Completed, Avg Req Tokens) were clipped to one row.

## [0.2.0] - 2026-06-04

First release of the maintained project — making the dashboard work against
current vLLM servers.

### Added
- `ansi-dark` default theme.

### Fixed
- vLLM v1 metrics schema: read `kv_cache_usage_perc`, derive the prefix-cache hit
  rate from the prefix-cache counters, and fall back to the `model_name` metric
  label when `/v1/models` requires auth.
- App no longer crashes on launch (invalid CSS, duplicate widget id).
- Card layout — cards share each row, and rows no longer overflow on narrow
  terminals.
- Robustness — drop non-finite (NaN/Inf) metric values, escape server-provided
  markup, and guard the refresh loop against a single bad sample.

[1.0.2]: https://github.com/tomaskir/vllm-monitor/releases/tag/v1.0.2
[1.0.1]: https://github.com/tomaskir/vllm-monitor/releases/tag/v1.0.1
[1.0.0]: https://github.com/tomaskir/vllm-monitor/releases/tag/v1.0.0
[0.2.0]: https://github.com/tomaskir/vllm-monitor/releases/tag/v0.2.0
