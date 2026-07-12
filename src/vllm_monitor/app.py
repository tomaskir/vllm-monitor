"""Textual TUI application for vllm-monitor."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, Static

from .metrics import MetricsPoller, VllmMetrics, render_spark

# Vertical resolution of the history charts, in text rows.
CHART_HEIGHT = 5

# Alert thresholds
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


def _format_duration(seconds: float) -> str:
    """Human-friendly duration with a unit that fits the magnitude.

    <1s → ms, <1min → s, <1h → "Mm Ss", else "Hh Mm".
    """
    if seconds < 1:
        ms = seconds * 1000
        return f"{ms:.1f}ms" if ms < 10 else f"{ms:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes}m {secs}s"
    hours, rem = divmod(int(round(seconds)), 3600)
    return f"{hours}h {rem // 60}m"


def _format_count(n: float) -> str:
    """Compact count: 1234 → '1.2K', 14175549 → '14.2M'."""
    n = abs(n)
    if n < 1000:
        return f"{n:.0f}"
    if n < 1_000_000:
        return f"{n / 1e3:.1f}K"
    if n < 1_000_000_000:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1e9:.1f}B"


def _fmt_latency(seconds: float) -> str:
    """Format a latency value as markup; '—' when unknown (<= 0)."""
    if seconds <= 0:
        return "[dim]—[/dim]"
    return f"[bold white]{_format_duration(seconds)}[/bold white]"


class MetricCard(Static):
    """A metric tile. Its title lives in the top border; the value fills the
    body. Single-value tiles are 3 rows; two-value tiles use ``tall=True``."""

    DEFAULT_CSS = """
    MetricCard {
        border: round $accent;
        border-title-align: left;
        border-title-color: $text-muted;
        padding: 0 1;
        height: 3;
        width: 1fr;
        margin: 0 1;
    }
    MetricCard.tall {
        height: 4;
    }
    MetricCard .card-value {
        text-style: bold;
        height: 1fr;
        width: 1fr;
        content-align: center middle;
    }
    """

    def __init__(self, card_id: str, title: str, tall: bool = False, **kwargs) -> None:
        super().__init__(id=card_id, classes="tall" if tall else "", **kwargs)
        self._title = title

    def compose(self) -> ComposeResult:
        yield Label("—", classes="card-value", id=f"{self.id}-value")

    def on_mount(self) -> None:
        self.border_title = self._title

    def update_value(self, markup: str) -> None:
        self.query_one(f"#{self.id}-value", Label).update(markup)


class SparklineCard(Static):
    """A card showing a labeled sparkline history chart."""

    DEFAULT_CSS = """
    SparklineCard {
        border: round $accent;
        border-title-align: left;
        border-title-color: $text-muted;
        padding: 0 1;
        height: 8;
        width: 1fr;
        margin: 0 1;
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
        yield Label("", classes="spark-line", id=f"{self.id}-spark")
        yield Label("", classes="spark-label", id=f"{self.id}-label")

    def on_mount(self) -> None:
        self.border_title = self._title

    def update_spark(
        self, values: deque[float], caption: str, fmt: Callable[[float], str]
    ) -> None:
        """Render the history as a multi-row bar chart with a y-axis scale.

        `fmt` formats the axis numbers (peak at top, 0 at bottom); `caption`
        is the line shown beneath the chart (e.g. the current value).
        """
        lines = render_spark(values, self.content_size.width, fmt, CHART_HEIGHT)
        self.query_one(f"#{self.id}-spark", Label).update("\n".join(lines))
        self.query_one(f"#{self.id}-label", Label).update(caption)


def _model_bar_markup(m: VllmMetrics) -> str:
    """Build the model header line: name + cache config (markup-escaped)."""
    info = m.model_info
    parts = [f"[bold cyan]{escape(info.model_id or 'unknown')}[/bold cyan]"]
    if info.cache_dtype:
        parts.append(f"kv {escape(info.cache_dtype)}")
    if info.num_gpu_blocks:
        parts.append(f"{info.num_gpu_blocks} blks")
    if info.gpu_memory_utilization:
        parts.append(f"util {info.gpu_memory_utilization:.0%}")
    return "  ·  ".join(parts)


class VllmMonitorApp(App):
    """Real-time vLLM monitoring dashboard."""

    TITLE = "vllm-monitor"
    SUB_TITLE = "vLLM Health Dashboard"

    CSS = """
    Screen {
        background: $surface;
    }
    #header-panel {
        height: 4;
        border: round $accent;
        border-title-align: left;
        margin: 0 1;
        padding: 0 1;
    }
    #status-bar {
        height: 1;
        color: $text-muted;
    }
    #model-bar {
        height: 1;
        color: $text-muted;
    }
    #body {
        height: 1fr;
    }
    /* Command palette → centered, outlined popup window */
    CommandPalette {
        align: center middle;
    }
    CommandPalette > Vertical {
        margin-top: 0;
        width: 70%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        border: round $accent;
        background: $surface;
    }
    .section {
        color: $accent;
        text-style: bold;
        height: 1;
        margin: 1 0 0 1;
    }
    .tile-row {
        height: 3;
    }
    .tall-row {
        height: 4;
    }
    #sparklines-row {
        height: 8;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh now"),
    ]

    metrics: reactive[VllmMetrics | None] = reactive(None)

    def __init__(self, poller: MetricsPoller, interval: float = 2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._poller = poller
        self._interval = interval
        # Guards against overlapping polls: the periodic timer awaits each tick,
        # but a manual refresh (action_refresh) runs on a separate task and could
        # otherwise start a second poll() while one is still in flight.
        self._tick_busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="header-panel"):
            yield Label("", id="status-bar")
            yield Label("", id="model-bar")
        with VerticalScroll(id="body"):
            yield Label("LOAD", classes="section")
            with Horizontal(id="load-row", classes="tile-row"):
                yield MetricCard("card-running", "Running")
                yield MetricCard("card-waiting", "Queued")
                yield MetricCard("card-preemptions", "Preemptions")
            yield Label("LATENCY", classes="section")
            with Horizontal(id="latency-row", classes="tile-row"):
                yield MetricCard("card-latency", "E2E Latency")
                yield MetricCard("card-ttft", "TTFT")
                yield MetricCard("card-tpot", "TPOT")
                yield MetricCard("card-queue", "Queue Time")
            yield Label("THROUGHPUT & CACHE", classes="section")
            with Horizontal(id="throughput-row", classes="tile-row"):
                yield MetricCard("card-prompt-tps", "Prompt Tokens/s")
                yield MetricCard("card-gen-tps", "Gen Tokens/s")
                yield MetricCard("card-gpu-cache", "GPU KV Cache")
                yield MetricCard("card-prefix-hit", "Prefix Cache Hit")
            yield Label("STATS", classes="section")
            with Horizontal(id="stats-row", classes="tall-row"):
                yield MetricCard("card-spec", "Spec Accept (MTP)", tall=True)
                yield MetricCard("card-finished", "Completed", tall=True)
                yield MetricCard("card-avgreq", "Average Request", tall=True)
            yield Label("HISTORY", classes="section")
            with Horizontal(id="sparklines-row"):
                yield SparklineCard("spark-running", "Active Requests")
                yield SparklineCard("spark-gentps", "Gen Tokens/s")
                yield SparklineCard("spark-cache", "GPU Cache %")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "ansi-dark"
        self.query_one(Header).icon = ""  # drop the default ⭘ header icon
        self.set_interval(self._interval, self._tick)
        self.call_after_refresh(self._tick)

    async def on_unmount(self) -> None:
        # Release the HTTP connection pool on shutdown (avoids a ResourceWarning).
        await self._poller.close()

    async def _tick(self) -> None:
        # Drop overlapping triggers: if a poll is already in flight (e.g. a manual
        # refresh landing mid-tick), skip rather than corrupt the poller's
        # prev-sample / history state with a concurrent poll().
        if self._tick_busy:
            return
        self._tick_busy = True
        # Never let a single bad sample or render error tear down the app —
        # log it and keep the dashboard running for the next tick.
        try:
            m = await self._poller.poll()
            self.metrics = m
            self._update_ui(m)
        except Exception as exc:  # noqa: BLE001 - last-resort UI guard
            self.log.error(f"tick failed: {exc!r}")
        finally:
            self._tick_busy = False

    def _update_ui(self, m: VllmMetrics) -> None:
        status = self.query_one("#status-bar", Label)
        status.update(
            "  ·  ".join(
                [
                    _status_color(m.server_reachable),
                    f"[dim]{escape(self._poller.base_url)}[/dim]",
                    f"[dim]refresh {self._interval:.0f}s[/dim]",
                ]
            )
        )

        self.query_one("#model-bar", Label).update(_model_bar_markup(m))

        self.query_one("#card-running", MetricCard).update_value(
            f"[bold white]{m.num_requests_running:.0f}[/bold white]"
        )
        # Queued: green when empty, yellow once a backlog forms.
        waiting_color = "green" if m.num_requests_waiting == 0 else "yellow"
        self.query_one("#card-waiting", MetricCard).update_value(
            f"[bold {waiting_color}]{m.num_requests_waiting:.0f}[/bold {waiting_color}]"
        )

        # Preemptions: green at 0, yellow once any have occurred (KV pressure).
        preempt = m.num_preemptions_total
        preempt_color = "green" if preempt == 0 else "yellow"
        self.query_one("#card-preemptions", MetricCard).update_value(
            f"[bold {preempt_color}]{preempt:.0f}[/bold {preempt_color}]"
        )

        # Completed: total requests and tokens processed (prompt + generation),
        # with the truncated-by-length / errored split.
        reasons = m.finished_reasons
        if reasons:
            total = sum(reasons.values())
            length = reasons.get("length", 0.0)
            errs = reasons.get("error", 0.0) + reasons.get("abort", 0.0)
            err_color = "red" if errs else "dim"
            tokens = m.prompt_tokens_total + m.generation_tokens_total
            self.query_one("#card-finished", MetricCard).update_value(
                f"[bold white]{total:,.0f} req[/bold white] · "
                f"[white]{_format_count(tokens)} tok[/white]\n"
                f"[dim]len {length:.0f}[/dim] · [{err_color}]err {errs:.0f}[/{err_color}]"
            )
        else:
            self.query_one("#card-finished", MetricCard).update_value("[dim]—[/dim]")

        # Average request stats: token counts, throughput, and E2E latency.
        e2e = m.latency_mean_s.get("e2e", 0.0)
        if m.avg_prompt_tokens > 0 or m.avg_generation_tokens > 0 or e2e > 0:
            self.query_one("#card-avgreq", MetricCard).update_value(
                f"[bold white]{_format_count(m.avg_prompt_tokens)} in[/bold white] · "
                f"[white]{_format_count(m.avg_generation_tokens)} out[/white]\n"
                f"[bold white]{m.avg_gen_tokens_per_sec:.1f} tok/s[/bold white] · "
                f"[bold white]{_format_duration(e2e)} E2E[/bold white]"
            )
        else:
            self.query_one("#card-avgreq", MetricCard).update_value("[dim]—[/dim]")

        for card_id, key in (
            ("#card-latency", "e2e"),
            ("#card-ttft", "ttft"),
            ("#card-tpot", "tpot"),
            ("#card-queue", "queue"),
        ):
            mean = m.latency_mean_s.get(key, 0.0)
            self.query_one(card_id, MetricCard).update_value(_fmt_latency(mean))

        self.query_one("#card-prompt-tps", MetricCard).update_value(
            f"[bold white]{m.prompt_tokens_per_sec:.1f}[/bold white]"
        )
        self.query_one("#card-gen-tps", MetricCard).update_value(
            f"[bold white]{m.generation_tokens_per_sec:.1f}[/bold white]"
        )

        self.query_one("#card-gpu-cache", MetricCard).update_value(
            _color_pct(m.gpu_cache_usage_perc, GPU_CACHE_WARN, GPU_CACHE_CRIT)
        )
        self.query_one("#card-prefix-hit", MetricCard).update_value(
            f"[bold white]{m.gpu_prefix_cache_hit_rate:.1f}%[/bold white]"
        )

        if m.spec_decode_active:
            self.query_one("#card-spec", MetricCard).update_value(
                f"[bold white]{m.spec_acceptance_rate:.1f}%[/bold white]\n"
                f"[dim]{m.spec_accept_length:.2f} tok/step[/dim]"
            )
        else:
            self.query_one("#card-spec", MetricCard).update_value("[dim]—[/dim]")

        # Sparklines
        h = self._poller.history
        self.query_one("#spark-running", SparklineCard).update_spark(
            h.requests_running,
            f"current={m.num_requests_running:.0f}",
            lambda x: f"{x:.0f}",
        )
        self.query_one("#spark-gentps", SparklineCard).update_spark(
            h.generation_tps,
            f"current={m.generation_tokens_per_sec:.1f} tok/s",
            lambda x: f"{x:.0f}",
        )
        self.query_one("#spark-cache", SparklineCard).update_spark(
            h.gpu_cache,
            f"current={m.gpu_cache_usage_perc:.1f}%",
            lambda x: f"{x:.0f}%",
        )

    def action_refresh(self) -> None:
        self.call_after_refresh(self._tick)
