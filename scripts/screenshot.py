"""Render a populated dashboard SVG for the README screenshot.

Outputs docs/screenshot.svg. Convert it to PNG (e.g. open in a browser
and screenshot the element) and save as docs/preview.png for the README.

Run from the repo root:  python scripts/screenshot.py
Uses synthetic demo data (no live server needed, no real URL/IP embedded).
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vllm_monitor.app import VllmMonitorApp  # noqa: E402
from vllm_monitor.metrics import HISTORY_SIZE, MetricsPoller, ModelInfo, VllmMetrics  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshot.svg"


def _wave(base: float, amp: float, periods: float) -> list[float]:
    return [
        base + amp * (0.5 + 0.5 * math.sin(i / HISTORY_SIZE * periods * 2 * math.pi))
        for i in range(HISTORY_SIZE)
    ]


def _demo_metrics() -> VllmMetrics:
    m = VllmMetrics(server_reachable=True)
    m.model_info = ModelInfo(
        model_id="deepseek-v4-flash",
        cache_dtype="fp8",
        num_gpu_blocks=94671,
        gpu_memory_utilization=0.92,
    )
    m.num_requests_running = 8
    m.num_requests_waiting = 3
    m.num_preemptions_total = 0
    m.latency_mean_s = {"e2e": 1.6, "ttft": 0.28, "tpot": 0.022, "queue": 0.14}
    m.prompt_tokens_per_sec = 1240.0
    m.generation_tokens_per_sec = 850.0
    m.gpu_cache_usage_perc = 42.0
    m.gpu_prefix_cache_hit_rate = 75.0
    m.spec_decode_active = True
    m.spec_acceptance_rate = 70.6
    m.spec_accept_length = 1.41
    m.finished_reasons = {"stop": 12000, "length": 387, "error": 0}
    m.prompt_tokens_total = 28_400_000
    m.generation_tokens_total = 510_000
    m.avg_prompt_tokens = 13_900
    m.avg_generation_tokens = 500
    m.avg_gen_tokens_per_sec = 838.0
    return m


async def _main() -> None:
    poller = MetricsPoller("http://localhost:8000")
    poller.history.requests_running.extend(_wave(6, 5, 2.5))
    poller.history.generation_tps.extend(_wave(700, 250, 3))
    poller.history.gpu_cache.extend(_wave(30, 18, 1.5))

    app = VllmMonitorApp(poller=poller)
    async with app.run_test(size=(104, 40)) as pilot:
        await pilot.pause()
        # The app defaults to ansi-dark (defers to the terminal palette), which
        # renders monochrome in a headless SVG export. Use a concrete RGB theme
        # so the screenshot shows the real colors.
        app.theme = "textual-dark"
        await pilot.pause()
        app._update_ui(_demo_metrics())
        await pilot.pause()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(app.export_screenshot(title="vllm-monitor"))
    await poller.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(_main())
