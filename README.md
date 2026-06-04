# vllm-monitor

Real-time terminal UI dashboard for monitoring [vLLM](https://github.com/vllm-project/vllm) server metrics. No Grafana required.

```
┌─────────────────────────────────── vllm-monitor ───────────────────────────────────┐
│ ● ONLINE  http://localhost:8000  interval=2s                                        │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────────┤
│ Model Info      │ Running Req      │ Queued Req       │ Avg E2E Latency             │
│ llama3          │       3          │       7          │       1600ms                │
├─────────────────┼─────────────────┼─────────────────┼─────────────────────────────┤
│ Prompt Tok/s    │ Gen Tok/s        │ GPU KV Cache     │ Prefix Hit                  │
│    450.2        │    312.7         │   42.0%          │   75.0%                     │
├─────────────────────────────┬─────────────────────────────┬───────────────────────────┤
│ Active Requests (history)   │ Gen Tokens/s (history)      │ GPU Cache % (history)     │
│  3│        ▁█▅▁             │ 312│        ▁█▅▁            │  42│       ▂▄▆█▆▄▂         │
│   │      ▂█████▇            │    │      ▂█████▇           │    │     ▄██████▇          │
│  0│ ▁▂▄▇██████████▅▃        │   0│ ▁▂▄▇██████████▅▃       │   0│ ▁▃▅███████████▅▃      │
│ current=3                   │ current=312.7 tok/s         │ current=42.0%             │
└─────────────────────────────────────────────────────────────────────────────────────┘
  q Quit  r Refresh now
```

## Features

- **Real-time metrics**: requests/sec, active + queued requests, token throughput (prompt & generated)
- **Cache stats**: GPU KV cache utilization, prefix cache hit rate
- **Request history charts**: rolling 60-sample multi-row bar charts (with a y-axis scale) for active requests, token throughput, and cache usage
- **Alert colors**: yellow at 80%, red at 95% for GPU KV cache
- **Model info panel**: loaded model name from `/v1/models`, or the `model_name` metric label when that endpoint requires auth
- **Configurable poll interval** (default 2s)

## Installation

```bash
pip install vllm-monitor
```

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

## Metrics Displayed

| Metric | Source | Description |
|--------|--------|-------------|
| Running Requests | `vllm:num_requests_running` | Requests actively being processed |
| Queued Requests | `vllm:num_requests_waiting` | Requests waiting for GPU capacity |
| Avg E2E Latency | `vllm:e2e_request_latency_seconds` | Mean end-to-end request latency |
| Prompt Tokens/s | `vllm:prompt_tokens_total` (rate) | Prompt token processing throughput |
| Gen Tokens/s | `vllm:generation_tokens_total` (rate) | Token generation throughput |
| GPU KV Cache | `vllm:kv_cache_usage_perc` (falls back to `vllm:gpu_cache_usage_perc`) | KV cache block utilization |
| Prefix Cache Hit | `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` | Prefix (radix) cache hit rate |

## Requirements

- Python 3.10+
- vLLM server with `/metrics` (Prometheus) and `/v1/models` endpoints enabled (default)

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Acknowledgments

Originally created by [Dennis Reichenberg](https://github.com/dennisreichenberg). Maintained as a fork by [Tomas Kirnak](https://github.com/tomaskir).

## License

MIT — see [LICENSE](LICENSE).
