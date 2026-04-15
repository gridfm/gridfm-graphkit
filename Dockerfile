# syntax=docker/dockerfile:1
FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/gridfm/gridfm-graphkit" \
      org.opencontainers.image.description="gridfm-graphkit" \
      org.opencontainers.image.version="0.0.6"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    HOME=/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        git \
        zip \
        unzip \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && python3 -m ensurepip --upgrade \
 && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# 1. Install Torch and PyG binaries
RUN pip install --no-cache-dir \
        "torch>=2.7.1,<2.9" \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir \
        torch-scatter \
        torch-sparse \
        torch-cluster \
        torch-spline-conv \
        -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 2. Install 'claimed' and the local app
# We install claimed separately to keep it in the cache layer
RUN pip install --no-cache-dir claimed

COPY . /app

# 3. Final app install
RUN pip install --no-cache-dir --ignore-installed \
    --extra-index-url https://download.pytorch.org/whl/cu128 /app

# 4. OPENSHIFT COMPATIBILITY: Set up permissions
# We ensure GID 0 (root group) owns the files and has write access.
# This allows OpenShift's random high-UIDs to run and write to the /app folder.
RUN chgrp -R 0 /app && \
    chmod -R g=u /app && \
    chmod -R 775 /app

# Use a non-privileged user (1001 is standard, but OpenShift will use a higher one)
USER 1001
