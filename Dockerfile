# syntax=docker/dockerfile:1

# --- build the wheel ---
FROM python:3.12-slim AS builder
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

# --- runtime ---
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/tomaskir/vllm-monitor"
LABEL org.opencontainers.image.description="Real-time terminal UI dashboard for monitoring vLLM"
LABEL org.opencontainers.image.licenses="MIT"

# TERM lets the Textual TUI pick colors; PYTHONUNBUFFERED keeps logs live.
ENV PYTHONUNBUFFERED=1 \
    TERM=xterm-256color

# vllm-monitor is a read-only client (connects out to vLLM); run unprivileged.
RUN useradd --create-home --uid 1000 monitor

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

USER monitor

# It's a TUI client — no ports are served. Pass --url (or VLLM_URL) to point
# it at a vLLM server, and run the container with -it for the terminal UI.
ENTRYPOINT ["vllm-monitor"]
