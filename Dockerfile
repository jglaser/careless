FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvcc/bin:/usr/local/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH"

WORKDIR /build
COPY . /build/

RUN uv pip install --system -U pip setuptools wheel \
    && uv pip install --system "/build[cuda]" "gemmi>=0.7.3" "h5py>=3.11" "tf-keras" \
    && find /usr/local/lib/python3.13/site-packages/nvidia -maxdepth 2 -type d -name lib > /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig \
    && rm -rf /build

WORKDIR /app
CMD ["careless", "--help"]
