# ==========================================
# Stage 1: Builder — Installs dependencies
# ==========================================
FROM ghcr.io/prefix-dev/pixi:latest AS builder

WORKDIR /app

# Install git if needed for dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests and data (if required for build tasks)
COPY pixi.toml pixi.lock ./
COPY ./data/ ./data/

# Install the environment into /app/.pixi
RUN CONDA_OVERRIDE_CUDA=12.0 pixi install --locked -e default

# Generate a shell-hook activation script so we don't need pixi binaries in runtime
RUN pixi shell-hook -e default > /shell-hook.sh \
    && echo 'exec "$@"' >> /shell-hook.sh

# ==========================================
# Stage 2: Runtime — Clean, lightweight image
# ==========================================
FROM ghcr.io/prefix-dev/pixi:latest AS runtime

WORKDIR /app

# Copy only the pre-compiled environment and shell hook from builder
COPY --from=builder /app/.pixi /app/.pixi
COPY --from=builder /shell-hook.sh /shell-hook.sh
COPY ./data/ ./data/
COPY ./src/ ./src/

ENV PYTHONUNBUFFERED=1

# Use the shell hook as entrypoint to activate the environment automatically
ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]

# Run your task
CMD ["pixi", "run", "-e", "default", "grid_run_local"]