# syntax=docker/dockerfile:1
#
# Image for the screener (`ticker-screener`). The stock_analyzer repo's
# `make release-build` builds it and its compose file runs it in one-off
# containers, as the deploy user's uid, with the host data directory mounted at
# /data and DB_NAME and TIINGO_API_KEY set in the environment:
#
#   docker compose -f compose.production.yaml run --rm screener analyze
#
# The image holds no .env, database or spreadsheet.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.13.5

##############################################################################
# Stage: deps - the locked runtime dependencies, without the dev group.
##############################################################################
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM python:${PYTHON_VERSION}-slim AS deps

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

##############################################################################
# Stage: runtime
##############################################################################
FROM python:${PYTHON_VERSION}-slim AS runtime

WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
# main.py reads config/ and schema.sql relative to the working directory.
COPY main.py schema.sql ./
COPY app ./app
COPY config ./config

# The container runs as whatever uid compose gives it, which has no home
# directory, so point HOME, where libraries keep their caches, at /tmp.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

# Provenance: map the image back to the commit it was built from.
ARG GIT_REVISION=unknown
LABEL org.opencontainers.image.title="ticker-screener" \
      org.opencontainers.image.revision="${GIT_REVISION}"

USER 65532:65532
ENTRYPOINT ["python", "main.py"]
