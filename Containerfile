# Containerfile for GridFM: installs latest gridfm-datakit and gridfm-graphkit from PyPI
# Build:  docker build -f Containerfile -t gridfm:latest .
# Run:    docker run --rm gridfm:latest
#
# gridfm-graphkit pulls in torch-scatter / torch-sparse, whose compiled C++ extensions must
# match the EXACT torch build in use. Two traps we avoid here:
#   1. Installing gridfm directly makes pip build torch-scatter under build isolation with no
#      torch visible  -> "No module named 'torch'".
#   2. Even once torch is present, a plain gridfm install re-resolves torch to another build,
#      leaving torch-scatter compiled against a now-uninstalled torch -> "undefined symbol".
# So: install CPU torch, compile torch-scatter/torch-sparse against it, then install the
# GridFM packages under a constraints file pinning those exact versions. Everything stays on
# the CPU wheel index to avoid multi-GB CUDA wheels.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# 1) CPU PyTorch stack. The released gridfm-graphkit 0.9.0 pins torch<2.13 (>=2.10), so we
#    stay in that range here; the newest CPU torch (2.14) would force pip to fall back to an
#    older gridfm release. torchvision/torchaudio are capped to their matching torch 2.12 line.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.10,<2.13" "torchvision<0.28" "torchaudio<2.13"

# 2) torch-scatter / torch-sparse compiled against the installed torch (guaranteed ABI match).
#    Try the matching prebuilt PyG wheel first, else build from source with torch visible.
RUN TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.__version__ + ('+cpu' if torch.version.cuda is None else ''))") \
    && echo "Resolving torch-scatter for torch-${TORCH_CUDA_VERSION}" \
    && (pip install --no-cache-dir torch-scatter torch-sparse \
            -f "https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html" \
        || pip install --no-cache-dir --no-build-isolation torch-scatter torch-sparse)

# 3) Pin the torch stack we just built, then install the latest GridFM releases under those
#    constraints so pip keeps (never re-resolves) torch/torch-scatter/torch-sparse.
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|torch-scatter|torch-sparse)==' > /tmp/torch-constraints.txt \
    && cat /tmp/torch-constraints.txt \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -c /tmp/torch-constraints.txt \
        gridfm-datakit gridfm-graphkit \
        "juliacall<0.9.35" "juliapkg<0.1.24"

# gridfm-datakit imports juliapkg.deps.run_julia, removed in juliapkg>=0.1.24. juliacall>=0.9.35
# hard-requires juliapkg>=0.1.24, so we cap juliacall at <0.9.35 (which allows juliapkg 0.1.23).
# Without the juliacall cap, pip backtracks on the gridfm packages instead and silently
# installs older gridfm releases rather than the requested latest.

# 4) Pre-install the Julia toolchain (Julia + PowerModels, Ipopt, Memento) that gridfm-datakit
#    uses for power-flow solving. Doing it at build time means `gridfm_datakit generate` works
#    offline and instantly; otherwise the first run downloads Julia into /root/.julia (needs
#    network, and is lost on a --rm container unless that path is a persisted volume).
RUN gridfm_datakit setup_pm

# MLflow 3.x rejects its default filesystem tracking backend unless this opt-out is set.
# gridfm-graphkit logs to a local ./mlruns file store by default, so enable it image-wide.
ENV MLFLOW_ALLOW_FILE_STORE=true

# Smoke test at runtime: import both packages (incl. torch-scatter C++ ext) and report versions
CMD ["python", "-c", "import importlib.metadata as m, gridfm_datakit, gridfm_graphkit, torch, torch_scatter; from torch_scatter import scatter_add; print('torch', torch.__version__); print('torch-scatter', m.version('torch-scatter')); print('gridfm-datakit', m.version('gridfm-datakit')); print('gridfm-graphkit', m.version('gridfm-graphkit')); print('OK')"]
