FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies
ARG TORCH_BACKEND=cpu
ENV UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
# The lock uses CPU wheels. CUDA builds replace only the pinned PyTorch pair.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra vision --no-install-project \
    && case "$TORCH_BACKEND" in \
        cpu) ;; \
        cu128) uv pip install --python /app/.venv/bin/python \
            --reinstall-package torch --reinstall-package torchvision \
            --index-url https://download.pytorch.org/whl/cu128 \
            torch==2.10.0 torchvision==0.25.0 ;; \
        *) echo "Unsupported TORCH_BACKEND: $TORCH_BACKEND" >&2; exit 2 ;; \
    esac \
    && uv pip check --python /app/.venv/bin/python

FROM base AS runtime
ENV PYTHONPATH=/app/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY --from=dependencies /app/.venv /app/.venv
COPY src ./src
COPY config ./config
COPY scripts ./scripts
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
ENTRYPOINT ["python", "-m", "multicamera_ai"]
CMD ["--config", "config/cameras.docker.yaml"]
