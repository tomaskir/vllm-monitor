"""Headless smoke tests for the Textual app.

These guard against crash-on-launch regressions: invalid CSS, duplicate
widget ids, and `_update_ui` querying ids that don't exist. They mount the
real app via Textual's test pilot but never touch the network — the poller
points at an unreachable address so poll() returns offline metrics.
"""

from __future__ import annotations

from vllm_monitor.app import ModelInfoPanel, VllmMonitorApp
from vllm_monitor.metrics import MetricsPoller, ModelInfo, VllmMetrics


def _make_app() -> VllmMonitorApp:
    poller = MetricsPoller(base_url="http://127.0.0.1:9")  # unreachable
    return VllmMonitorApp(poller=poller, interval=0.5)


async def test_app_composes_and_ticks():
    """The app mounts, composes its CSS, and a tick updates every card."""
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._tick()  # exercises _update_ui + every query_one
        await pilot.pause()

        # Card ids referenced by _update_ui must all resolve.
        for selector in (
            "#status-bar",
            "#model-panel",
            "#card-running",
            "#card-waiting",
            "#card-latency",
            "#card-prompt-tps",
            "#card-gen-tps",
            "#card-gpu-cache",
            "#card-prefix-hit",
            "#card-gpu-mem",
            "#spark-running",
            "#spark-gentps",
            "#spark-cache",
        ):
            assert app.query(selector), f"missing widget {selector}"

    await app._poller.close()


async def test_rows_lay_out_side_by_side():
    """Each row shows all its cards side-by-side, not just the first one.

    Regression for cards defaulting to full container width, which hid every
    sibling after the first.
    """
    expected = {"#model-row": 4, "#metrics-row": 5, "#sparklines-row": 3}
    # Narrow widths (44/80) used to overflow: any fixed min-width on the cards
    # makes the 1fr layout balloon once n*min-width exceeds the terminal width.
    for width in (44, 80, 120):
        app = _make_app()
        async with app.run_test(size=(width, 40)) as pilot:
            await pilot.pause()
            for row, count in expected.items():
                cards = app.query_one(row).children
                assert len(cards) == count, f"{row} has {len(cards)} cards, expected {count}"
                xs = [c.region.x for c in cards]
                assert all(c.region.width > 0 for c in cards), f"{row} has a zero-width card"
                assert len(set(xs)) == count, f"{row} cards overlap at w={width}: {xs}"
                right_edge = max(c.region.x + c.region.width for c in cards)
                assert right_edge <= width, f"{row} overflows at w={width}: right={right_edge}"
        await app._poller.close()


async def test_markup_in_model_name_does_not_crash():
    """A server-provided model name with markup metacharacters must not crash
    the render (it would raise rich MarkupError if not escaped)."""
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one("#model-panel", ModelInfoPanel)
        m = VllmMetrics()
        m.model_info = ModelInfo(model_id="evil[/]name [red]x")
        panel.update_model(m)  # must not raise
        await pilot.pause()
        label = app.query_one("#model-id")
        # The literal text is preserved (escaped), not interpreted as markup.
        assert "evil[/]name [red]x" in str(label.render())
    await app._poller.close()
