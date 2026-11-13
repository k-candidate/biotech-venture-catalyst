# =====================================================================
# STAGE 1: Dependency Compilation Builder Layer
# =====================================================================
FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim AS builder

WORKDIR /app

# Enable bytecode compilation for optimal startup speeds inside the container
ENV UV_COMPILE_BYTECODE=1

# Copy dependency definition configurations first to utilize layer caching
COPY pyproject.toml uv.lock ./

# Sync and install project packages to a local cache directory (.venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# =====================================================================
# STAGE 2: Lightweight Final Serving Runtime Layer
# =====================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Prevent Python from writing .pyc files and force unbuffered output logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the pre-compiled virtual environment packages from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Update environmental PATH to execute binaries straight from the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy the production application files and trained model artifacts.
COPY model.bin predict.py model.py ./

# Expose the network port matching your local API server configuration
EXPOSE 9696

# Define the production serving command using high-performance Uvicorn
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
