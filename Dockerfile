FROM python:3.12-slim

ARG TINYCONTEXT_VERSION=dev

LABEL org.opencontainers.image.title="TinyContext" \
      org.opencontainers.image.description="Token-light SQLite hybrid memory for local agents" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/TinySuiteHQ/TinyContext" \
      org.opencontainers.image.version="${TINYCONTEXT_VERSION}" \
      io.modelcontextprotocol.server.name="io.github.TinySuiteHQ/tinycontext"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TINYCONTEXT_VERSION=${TINYCONTEXT_VERSION} \
    TINYCONTEXT_MEMORY_DB_PATH=/data/memories.db \
    TINYCONTEXT_MODELS_DIR=/data/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gosu \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip \
    && pip install ".[server,telemetry]" "msgpack>=1.2.1" "setuptools>=78.1.1" \
    && pip check \
    && pip uninstall --yes pip setuptools \
    && useradd --create-home --shell /usr/sbin/nologin tinycontext \
    && mkdir -p /data/models \
    && chown -R tinycontext:tinycontext /data \
    && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD tinycontext doctor || exit 1

# No USER on purpose: docker-entrypoint.sh needs root to re-own a bind-mounted
# /data before it drops to tinycontext via gosu. See .trivyignore.yaml.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["tinycontext", "mcp"]
