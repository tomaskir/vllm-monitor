# vllm-monitor

Real-time terminal UI dashboard for monitoring [vLLM](https://github.com/vllm-project/vllm) server metrics. No Grafana required.

```
              vllm-monitor — vLLM Health Dashboard
 ╭───────────────────────────────────────────────────╮
 │ ● ONLINE   http://localhost:8000   ·   refresh 2s   │
 │ deepseek-v4-flash · kv fp8 · 94671 blks · util 92%  │
 ╰───────────────────────────────────────────────────╯
 LOAD
 ╭ Running ─╮ ╭ Queued ──╮ ╭ Preemptions ╮
 │    3     │ │    7     │ │      0      │
 ╰──────────╯ ╰──────────╯ ╰─────────────╯
 LATENCY
 ╭ E2E Latency ╮ ╭ TTFT ─╮ ╭ TPOT ─╮ ╭ Queue Time ╮
 │    1.6s     │ │ 280ms │ │ 22ms  │ │   140ms    │
 ╰─────────────╯ ╰───────╯ ╰───────╯ ╰────────────╯
 THROUGHPUT & CACHE
 ╭ Prompt Tokens/s ╮ ╭ Gen Tokens/s ╮ ╭ GPU KV Cache ╮ ╭ Prefix Cache Hit ╮
 │      450.2      │ │    312.7     │ │    42.0%     │ │      75.0%       │
 ╰─────────────────╯ ╰──────────────╯ ╰──────────────╯ ╰──────────────────╯
 STATS
 ╭ Spec Accept (MTP) ╮ ╭ Completed ───────────╮ ╭ Avg Req Tokens ╮
 │ 70.6%             │ │ 1,021 req · 14.7M tok │ │ 13.9K in       │
 │ 1.41 tok/step     │ │ len 387 · err 0       │ │ 500 out        │
 ╰───────────────────╯ ╰───────────────────────╯ ╰────────────────╯
 HISTORY
 ╭ Active Requests ╮ ╭ Gen Tokens/s ───╮ ╭ GPU Cache % ────╮
 │312│      ▁█▅▁   │ │   │     ▂▄▆█▆▄▂  │ │   │    ▁▂▃▄▅▆▇   │
 │   │    ▂█████▇  │ │   │   ▄██████▇   │ │   │  ▃▅███████   │
 │  0│ ▁▂▄▇███████ │ │  0│ ▁▃▅████████  │ │  0│ ▅█████████   │
 │ current=312.7   │ │ current=42.0%   │ │ current=75.0%   │
 ╰─────────────────╯ ╰─────────────────╯ ╰─────────────────╯
  q Quit  r Refresh now
```

## Features

- **Request load**: running, queued, and preemption counts (preemptions flag KV-cache pressure)
- **Latency**: end-to-end, time-to-first-token (TTFT), time-per-output-token (TPOT), and queue time — recent means with units auto-scaled to magnitude (ms / s / m / h)
- **Throughput**: prompt and generation tokens/sec
- **Cache**: GPU KV-cache utilization and prefix (radix) cache hit rate, with alert colors (yellow ≥ 80%, red ≥ 95% on KV cache)
- **Speculative decoding (MTP)**: acceptance rate and accept length (accepted tokens per draft step); shows `—` when spec decode is off
- **Completed requests**: totals (requests + tokens) with a finish-reason breakdown (truncated-by-length, errors)
- **Average request shape**: mean prompt (input) vs generation (output) tokens per request
- **History charts**: rolling 60-sample multi-row bar charts with a y-axis scale for active requests, generation throughput, and KV-cache usage
- **Model panel**: model name (from `/v1/models`, or the `model_name` metric label when that endpoint needs auth) plus cache config — KV dtype, GPU blocks, memory-utilization target
- **Clean TUI**: tiles grouped into labeled sections, framed header, a centered/outlined command palette (`Ctrl+P`), and the `ansi-dark` theme by default
- **Graceful degradation**: any metric the server doesn't expose simply shows `—`
- **Configurable poll interval** (default 2s)

## Installation

Install from source, or use the [Docker image](#docker).

```bash
# Recommended: isolated install with pipx
pipx install git+https://github.com/tomaskir/vllm-monitor

# or with pip
pip install git+https://github.com/tomaskir/vllm-monitor
```

## Docker

A prebuilt image is published to GitHub Container Registry on every release
(tags: `latest`, `X.Y.Z`, `X.Y`). vllm-monitor is a **client** — it connects
out to a vLLM server and renders a terminal UI, so run it with `-it` (a TTY)
and point it at your server. It serves no ports.

```bash
# Pull the latest release
docker pull ghcr.io/tomaskir/vllm-monitor:latest

# Monitor a remote vLLM server
docker run --rm -it ghcr.io/tomaskir/vllm-monitor --url http://10.0.0.5:8000

# Monitor vLLM on the same host (Linux): share the host network
docker run --rm -it --network host ghcr.io/tomaskir/vllm-monitor --url http://localhost:8000

# macOS / Windows: reach the host via host.docker.internal
docker run --rm -it ghcr.io/tomaskir/vllm-monitor --url http://host.docker.internal:8000

# Configure entirely via environment variables
docker run --rm -it \
  -e VLLM_URL=http://10.0.0.5:8000 \
  -e VLLM_API_KEY=mytoken \
  -e VLLM_MONITOR_INTERVAL=1 \
  ghcr.io/tomaskir/vllm-monitor
```

| Env var | Equivalent flag | Default |
|---------|-----------------|---------|
| `VLLM_URL` | `--url` | `http://localhost:8000` |
| `VLLM_API_KEY` | `--api-key` | _(none)_ |
| `VLLM_MONITOR_INTERVAL` | `--interval` | `2` |

Build it yourself:

```bash
docker build -t vllm-monitor .
docker run --rm -it vllm-monitor --url http://10.0.0.5:8000
```

> `-it` is required — without a TTY the TUI cannot render. `--rm` removes the container on exit.

## Usage

```bash
# Monitor local vLLM server (default: http://localhost:8000, 2s interval)
vllm-monitor

# Custom server URL
vllm-monitor --url http://my-vllm-server:8000

# Faster refresh
vllm-monitor --interval 1

# With API key
vllm-monitor --url http://my-server:8000 --api-key mytoken
# or via env var:
VLLM_API_KEY=mytoken vllm-monitor
```

### Key Bindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Refresh immediately |
| `Ctrl+P` | Command palette |

## Metrics Displayed

| Metric | Source | Description |
|--------|--------|-------------|
| Running | `vllm:num_requests_running` | Requests actively being processed |
| Queued | `vllm:num_requests_waiting` | Requests waiting for GPU capacity |
| Preemptions | `vllm:num_preemptions_total` | Requests evicted under KV-cache pressure (green at 0) |
| E2E Latency | `vllm:e2e_request_latency_seconds` | Mean end-to-end request latency |
| TTFT | `vllm:time_to_first_token_seconds` | Mean time to first token |
| TPOT | `vllm:request_time_per_output_token_seconds` | Mean time per output token |
| Queue Time | `vllm:request_queue_time_seconds` | Mean time a request waits before scheduling |
| Prompt Tokens/s | `vllm:prompt_tokens_total` (rate) | Prompt token processing throughput |
| Gen Tokens/s | `vllm:generation_tokens_total` (rate) | Token generation throughput |
| GPU KV Cache | `vllm:kv_cache_usage_perc` (falls back to `vllm:gpu_cache_usage_perc`) | KV cache block utilization |
| Prefix Cache Hit | `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` | Prefix (radix) cache hit rate |
| Spec Accept (MTP) | `vllm:spec_decode_num_{accepted_tokens,draft_tokens,drafts}_total` | Acceptance rate and accepted tokens per draft step |
| Completed | `vllm:request_success_total` (by `finished_reason`) | Completed requests + tokens, with truncated/errored split |
| Avg Req Tokens | `vllm:request_prompt_tokens` / `vllm:request_generation_tokens` | Mean prompt / generation tokens per request |
| Model / config | `/v1/models`, `vllm:cache_config_info` | Model name, KV dtype, GPU blocks, mem-util target |

Latency values are *recent* means (the change in the histogram's sum/count between polls), falling back to the cumulative mean. Any metric the server doesn't expose is shown as `—`.

## Requirements

- Python 3.10+
- A vLLM server exposing `/metrics` (Prometheus — enabled by default). Works with the vLLM **v1** engine metric names. `/v1/models` is optional: the model name falls back to the `model_name` metric label when that endpoint is unavailable or requires auth.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Acknowledgments

Originally created by [Dennis Reichenberg](https://github.com/dennisreichenberg). Maintained as a fork by [Tomas Kirnak](https://github.com/tomaskir).

## License

MIT — see [LICENSE](LICENSE).
