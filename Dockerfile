FROM python:3.11.9-slim

# Build-time proxy only (ARG, not persisted). NO_PROXY is baked for runtime.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NO_PROXY=postgres,backend,host.docker.internal,127.0.0.1,localhost \
    no_proxy=postgres,backend,host.docker.internal,127.0.0.1,localhost

WORKDIR /app

# Java runtime for OpenDataLoader/Tika PDF fallback (Docling primary still
# needs a JRE for the fallback path).
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java
RUN java -version

# CPU-only torch first: default PyPI pulls multi-GB CUDA wheels even though
# inference is BGE-M3 CPU + Ollama.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Deps layer cached separately from code.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple .

RUN mkdir -p /app/uploads
ENV UPLOAD_DIR=/app/uploads

EXPOSE 8000

# Shell form so $PORT is honored (default 8000, see core/config.py + .env).
CMD ["sh", "-c", "uvicorn rip_maf.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
