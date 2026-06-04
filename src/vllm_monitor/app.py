"""Textual TUI application for vllm-monitor."""

from __future__ import annotations

import asyncio
from typing import Optional

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, Static

from .metrics import MetricsPoller, VllmMetrics, sparkline

# Alert thresholds
GPU_MEM_WARN = 80.0
GPU_MEM_CRIT = 90.0
GPU_CACHE_WARN = 80.0
GPU_CACHE_CRIT = 95.0


def _color_pct(value: float, warn: float, crit: float) -> str:
    if value >= crit:
        return f"[bold red]{value:.1f}%[/bold red]"
    if value >= warn:
        return f"[bold yellow]{value:.1f}%[/bold yellow]"
    return f"[bold green]{value:.1f}%[/bold green]"


def _status_color(reachable: bool) -> str:
    return "[bold green]● ONLINE[/bold green]" if reachable else "[bold red]● OFFLINE[/bold red]"


class MetricCard(Static):
    """A single metric display card."""

    DEFAULT_CSS = """
    MetricCard {
        border: round $accent;
        padding: 0 1;
        height: 5;
    }
    MetricCard .card-title {
        color: $text-muted;
        text-style: bold;
    }
    MetricCard .card-value {
        text-align: center;
        text-style: bold;
        height: 2;
        padding-top: 1;
    }
    """

    def __init__(self, card_id: str, title: str, **kwargs) -> None:
        super().__init__(**kwargs, id=card_id)
        self._title = title

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="card-title")
        yield Label("—", classes="card-value", id=f"{self.id}-value")

    def update_value(self, markup: str) -> None:
        self.query_one(f"#{self.id}-value", Label).update(markup)


class SparklineCard(Static):
    """A card showing a labeled sparkline history chart."""

    DEFAULT_CSS = """
    SparklineCard {
        border: round $accent;
        padding: 0 1;
        height: 5;
    }
    SparklineCard .spark-title {
        color: $text-muted;
        text-style: bold;
    }
    SparklineCard .spark-line {
        color: $success;
    }
    SparklineCard .spark-label {
        color: $text-muted;
    }
    """

    def __init__(self, card_id: str, title: str, **kwargs) -> None:
        super().__init__(**kwargs, id=card_id)
        self._title = title

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="spark-title")
        yield Label("", classes="spark-line", id=f"{self.id}-spark")
        yield Label("", classes="spark-label", id=f"{self.id}-label")

    def update_spark(self, spark: str, label: str) -> None:
        self.query_one(f"#{self.id}-spark", Label).update(spark)
        self.query_one(f"#{self.id}-label", Label).update(label)


class ModelInfoPanel(Static):
    """Panel showing model info."""

    DEFAULT_CSS = """
    ModelInfoPanel {
        border: round $accent;
        padding: 0 1;
        height: 5;
    }
    ModelInfoPanel .model-title {
        color: $text-muted;
        text-style: bold;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Model Info", classes="model-title")
        yield Label("—", id="model-id")
        yield Label("", id="model-extra")

    def update_model(self, m: VllmMetrics) -> None:
        info = m.model_info
        # Server-provided; escape so markup metacharacters can't inject styling
        # or raise MarkupError and crash the render.
        model_id = escape(info.model_id or "unknown")
        self.query_one("#model-id", Label).update(f"[bold cyan]{model_id}[/bold cyan]")
        extras = []
        if info.max_model_len:
            extras.append(f"ctx={info.max_model_len}")
        if info.tensor_parallel_size:
            extras.append(f"tp={info.tensor_parallel_size}")
        self.query_one("#model-extra", Label).update("  ".join(extras) if extras else "")


class VllmMonitorApp(App):
    """Real-time vLLM monitoring dashboard."""

    TITLE = "vllm-monitor"
    SUB_TITLE = "vLLM Health Dashboard"

    CSS = """
    Screen {
        background: $surface;
    }
    #status-bar {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }
    #metrics-row {
        height: 5;
        margin: 0;
    }
    #sparklines-row {
        height: 5;
        margin: 0;
    }
    #model-row {
        height: 5;
        margin: 0;
    }
    MetricCard, SparklineCard, ModelInfoPanel {
        width: 1fr;
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh now"),
    ]

    metrics: reactive[Optional[VllmMetrics]] = reactive(None)

    def __init__(self, poller: MetricsPoller, interval: float = 2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._poller = poller
        self._interval = interval

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("", id="status-bar")
        with Horizontal(id="model-row"):
            yield ModelInfoPanel(id="model-panel")
            yield MetricCard("card-running", "Running Requests")
            yield MetricCard("card-waiting", "Queued Requests")
            yield MetricCard("card-latency", "Avg E2E Latency")
        with Horizontal(id="metrics-row"):
            yield MetricCard("card-prompt-tps", "Prompt Tokens/s")
            yield MetricCard("card-gen-tps", "Gen Tokens/s")
            yield MetricCard("card-gpu-cache", "GPU KV Cache")
            yield MetricCard("card-prefix-hit", "Prefix Cache Hit")
            yield MetricCard("card-gpu-mem", "GPU Memory")
        with Horizontal(id="sparklines-row"):
            yield SparklineCard("spark-running", "Active Requests (history)")
            yield SparklineCard("spark-gentps", "Gen Tokens/s (history)")
            yield SparklineCard("spark-cache", "GPU Cache % (history)")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "ansi-dark"
        self.set_interval(self._interval, self._tick)
        self.call_after_refresh(self._tick)

    async def _tick(self) -> None:
        # Never let a single bad sample or render error tear down the app —
        # log it and keep the dashboard running for the next tick.
        try:
            m = await self._poller.poll()
            self.metrics = m
            self._update_ui(m)
        except Exception as exc:  # noqa: BLE001 - last-resort UI guard
            self.log.error(f"tick failed: {exc!r}")

    def _update_ui(self, m: VllmMetrics) -> None:
        status = self.query_one("#status-bar", Label)
        status.update(
            f"{_status_color(m.server_reachable)}  "
            f"[dim]{escape(self._poller.base_url)}[/dim]  "
            f"[dim]interval={self._interval:.0f}s[/dim]"
        )

        self.query_one("#model-panel", ModelInfoPanel).update_model(m)

        self.query_one("#card-running", MetricCard).update_value(f"[bold cyan]{m.num_requests_running:.0f}[/bold cyan]")
        self.query_one("#card-waiting", MetricCard).update_value(f"[bold yellow]{m.num_requests_waiting:.0f}[/bold yellow]")

        if m.e2e_latency_mean_s > 0:
            latency_ms = m.e2e_latency_mean_s * 1000
            self.query_one("#card-latency", MetricCard).update_value(f"[bold white]{latency_ms:.0f}ms[/bold white]")
        else:
            self.query_one("#card-latency", MetricCard).update_value("[dim]—[/dim]")

        self.query_one("#card-prompt-tps", MetricCard).update_value(f"[bold white]{m.prompt_tokens_per_sec:.1f}[/bold white]")
        self.query_one("#card-gen-tps", MetricCard).update_value(f"[bold white]{m.generation_tokens_per_sec:.1f}[/bold white]")

        self.query_one("#card-gpu-cache", MetricCard).update_value(_color_pct(m.gpu_cache_usage_perc, GPU_CACHE_WARN, GPU_CACHE_CRIT))
        self.query_one("#card-prefix-hit", MetricCard).update_value(f"[bold cyan]{m.gpu_prefix_cache_hit_rate:.1f}%[/bold cyan]")

        # GPU memory
        if m.gpu_memory_total_bytes > 0:
            gpu_pct = m.gpu_memory_used_bytes / m.gpu_memory_total_bytes * 100
            used_gb = m.gpu_memory_used_bytes / 1e9
            total_gb = m.gpu_memory_total_bytes / 1e9
            gpu_markup = f"{_color_pct(gpu_pct, GPU_MEM_WARN, GPU_MEM_CRIT)}\n[dim]{used_gb:.1f}/{total_gb:.1f}GB[/dim]"
        else:
            gpu_markup = "[dim]—[/dim]"
        self.query_one("#card-gpu-mem", MetricCard).update_value(gpu_markup)

        # Sparklines
        h = self._poller.history
        self.query_one("#spark-running", SparklineCard).update_spark(
            sparkline(h.requests_running, 40),
            f"current={m.num_requests_running:.0f}"
        )
        self.query_one("#spark-gentps", SparklineCard).update_spark(
            sparkline(h.generation_tps, 40),
            f"current={m.generation_tokens_per_sec:.1f} tok/s"
        )
        self.query_one("#spark-cache", SparklineCard).update_spark(
            sparkline(h.gpu_cache, 40),
            f"current={m.gpu_cache_usage_perc:.1f}%"
        )

    def action_refresh(self) -> None:
        self.call_after_refresh(self._tick)
