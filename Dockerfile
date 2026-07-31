FROM python:3.12.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/var/lib/rag/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        curl \
        libgomp1 \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-chi-tra \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 rag \
    && useradd --system --uid 10001 --gid rag --home /var/lib/rag rag

COPY requirements.lock pyproject.toml README.md ./
RUN python -m pip install --requirement requirements.lock

COPY alembic.ini ./
COPY migrations ./migrations
COPY rag_demo ./rag_demo
COPY profiles ./profiles
COPY scripts ./scripts
RUN python -m pip install --no-deps . \
    && chmod 0555 /app/scripts/docker-entrypoint.sh \
    && mkdir -p /var/lib/rag/huggingface \
    && chown -R rag:rag /var/lib/rag

USER 10001:10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=2)"

ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "rag_demo.production.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips=*"]
