# syntax=docker/dockerfile:1

# ==========================================
# Stage 1: Builder — Installs dependencies
# ==========================================
FROM ghcr.io/prefix-dev/pixi:latest AS builder

WORKDIR /app
ENV CONDA_OVERRIDE_CUDA=12.0

# Install git if needed for dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY pixi.toml pixi.lock ./

# Install the environment
RUN pixi install --locked -e default


# ==========================================
# Stage 2: Runtime — Clean final image
# ==========================================
FROM ghcr.io/prefix-dev/pixi:latest AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Copy the pre-built .pixi environment and manifests from builder
COPY --from=builder /app/.pixi /app/.pixi
COPY --from=builder /app/pixi.toml /app/pixi.toml
COPY --from=builder /app/pixi.lock /app/pixi.lock

# Copy data and source code (changes here will NEVER re-trigger pixi install)
COPY ./data/ ./data/
COPY ./src/ ./src/

# Run the task directly using Pixi (since Pixi is built into this base image)
CMD ["pixi", "run", "-e", "default", "grid_run_local"]