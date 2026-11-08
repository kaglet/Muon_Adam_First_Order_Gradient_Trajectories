# syntax=docker/dockerfile:1
FROM ghcr.io/prefix-dev/pixi:latest

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Git pull for Muon optim
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the manifest and lockfile first so Docker can reuse the dependency-install layer.
COPY pixi.toml pixi.lock ./

# Copy the rest of the project.
COPY ./data/ ./data/     

# Install the Linux environment. This assumes pixi.lock is present and up to date.
RUN --mount=type=cache,target=/app/.pixi \
    CONDA_OVERRIDE_CUDA=12.0 pixi install --locked -e default

COPY ./src/ ./src/

# Run the task and leave its output visible in the terminal.
CMD ["pixi", "run", "-e", "default", "grid_run_local"]
