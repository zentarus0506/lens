# Slim runtime image: no build toolchain, no test deps.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LENS_HOST=0.0.0.0 \
    LENS_PORT=8765

WORKDIR /app

# Dependencies first so a source edit does not invalidate the dependency layer.
# PIP_INDEX_URL stays overridable so builds behind a slow mirror can pass one in
# without baking a private registry into the image.
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY pyproject.toml README.md ./
COPY lens ./lens
RUN pip install --no-cache-dir --index-url ${PIP_INDEX_URL} . && rm -rf /root/.cache

# SQLite lives on a volume so traces survive a container restart.
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=2).status==200 else 1)"

CMD ["python", "-m", "lens.server.app"]
