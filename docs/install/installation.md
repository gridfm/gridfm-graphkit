# Installation

The steps below mirror the [README](https://github.com/gridfm/gridfm-graphkit/blob/main/README.md#installation).

Create and activate a virtual environment (make sure you use the right python version = 3.10, 3.11 or 3.12. I highly recommend 3.12)

```bash
python -m venv venv
source venv/bin/activate
```

Install gridfm-graphkit from PyPI. `torch-scatter` wheels are not on PyPI.
pip does not read `[tool.uv] find-links` in `pyproject.toml`, so pass the PyG
index on the command line and pin torch to 2.12 (the newest release with
prebuilt scatter wheels). `pyproject.toml` allows `torch<2.14` for the `vllm`
extra; torch 2.13 has no scatter wheel, and an unpinned install selects that
torch plus a mismatched 2.12 scatter wheel.

```bash
pip install gridfm-graphkit "torch==2.12.*" \
    --find-links https://data.pyg.org/whl/torch-2.12.0+cpu.html
```

The URL above is the CPU index. For CUDA, swap `cpu` for the matching tag
(`cu126`, `cu118`, …). Then install `torch-scatter` for the torch version pip
selected:

```bash
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"
```

When no prebuilt wheel exists (including torch 2.13), install torch first and
build scatter against it. This needs a C++ compiler:

```bash
pip install "torch>=2.10,<2.14"
pip install torch-scatter --no-build-isolation
```

For documentation generation and unit testing, install the `dev` and `test`
extras the same way (`gdown` is in the `test` extra):

```bash
pip install "gridfm-graphkit[dev,test]" "torch==2.12.*" \
    --find-links https://data.pyg.org/whl/torch-2.12.0+cpu.html
```
