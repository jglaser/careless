FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=300

WORKDIR /build
COPY . /build/

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:/opt/venv/lib/python3.13/site-packages/nvidia/cuda_nvcc/bin:/opt/venv/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH"

RUN uv pip install -U pip setuptools wheel \
    && uv pip install "/build[cuda]" "gemmi>=0.7.3" "h5py>=3.11" "tf-keras" \
    && find /opt/venv/lib/python3.13/site-packages/nvidia -maxdepth 2 -type d -name lib > /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig \
    && rm -rf /build

WORKDIR /app
CMD ["careless", "--help"]
