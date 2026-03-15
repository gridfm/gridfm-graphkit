# Installation

You can install `gridfm-graphkit` directly from PyPI:

```bash
pip install gridfm-graphkit
```

For GPU support and compatibility with PyTorch Geometric's scatter operations, install PyTorch (and optionally CUDA) first, then install the matching `torch-scatter` wheel. See [PyTorch and torch-scatter](#pytorch-and-torch-scatter-optional) below.

---

## Development Setup

To contribute or develop locally, clone the repository and install in editable mode. Use Python 3.10, 3.11, or 3.12 (3.12 is recommended).

```bash
git clone git@github.com:gridfm/gridfm-graphkit.git
cd gridfm-graphkit
python -m venv venv
source venv/bin/activate
pip install -e .
```

### PyTorch and torch-scatter (optional)

If you need GPU acceleration or PyTorch Geometric scatter ops (used by the library), install PyTorch and the matching `torch-scatter` wheel:

1. Install PyTorch (see [pytorch.org](https://pytorch.org/) for your platform and CUDA version).

2. Get your Torch + CUDA version string:
   ```bash
   TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.__version__ + ('+cpu' if torch.version.cuda is None else ''))")
   ```

3. Install the correct `torch-scatter` wheel:
   ```bash
   pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
   ```

---

## Optional extras

For documentation generation and unit testing, install with the optional `dev` and `test` extras:

```bash
pip install -e .[dev,test]
```
