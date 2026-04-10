# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn9-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/gridfm/gridfm-graphkit" \
      org.opencontainers.image.description="gridfm-graphkit" \
      org.opencontainers.image.version="0.0.6"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && python -m pip install --upgrade pip setuptools wheel

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app
COPY . /app

# Install torch with the matching CUDA index before the package so pip
# does not fall back to a CPU-only wheel.
RUN pip install --no-cache-dir \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir /app

ENTRYPOINT ["gridfm_graphkit"]
CMD ["--help"]
