# Installation

The steps below mirror the [README](https://github.com/gridfm/gridfm-graphkit/blob/main/README.md#installation).

Create and activate a virtual environment (Python 3.10–3.13; 3.12 is a good default)

```bash
python -m venv venv
source venv/bin/activate
```

Install gridfm-graphkit from PyPI

```bash
pip install gridfm-graphkit
```

**`torch-scatter` is a required dependency.** It cannot be bundled in `pyproject.toml` because the correct wheel depends on your PyTorch and CUDA versions, so it must be installed separately.

Get PyTorch + CUDA version for torch-scatter

```bash
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.__version__ + ('+cpu' if torch.version.cuda is None else ''))")
```

Install the correct torch-scatter wheel

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
```


For documentation generation and unit testing, install with the optional `dev` and `test` extras:

```bash
pip install "gridfm-graphkit[dev,test]"
```
