# Installation

The steps below match the [README](https://github.com/gridfm/gridfm-graphkit/blob/main/README.md#installation).
To use a container instead, see [Docker / containers](docker.md).

Python 3.10, 3.11 or 3.12.

```bash
python -m venv venv
source venv/bin/activate
```

On Windows, run `venv\Scripts\activate` instead.

Then run one of these:

CPU (Linux, Windows, macOS on Apple silicon):

```bash
pip install gridfm-graphkit "torch==2.12.*" torch-scatter --extra-index-url https://download.pytorch.org/whl/cpu -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
```

NVIDIA GPU (Linux, Windows; `nvidia-smi` must report CUDA Version 12.6 or higher):

```bash
pip install gridfm-graphkit "torch==2.12.*" torch-scatter --extra-index-url https://download.pytorch.org/whl/cu126 -f https://data.pyg.org/whl/torch-2.12.0+cu126.html
```

RTX 50xx and other Blackwell GPUs need CUDA 13: replace `cu126` with `cu130` in both URLs (Linux only, `nvidia-smi` must report CUDA Version 13.0 or higher).

For RRWP positional encodings, add `torch-sparse` after `torch-scatter`.

Check the install:

```bash
python -c "import torch, torch_scatter, gridfm_graphkit; print(torch.__version__, torch.cuda.is_available())"
```

Set `MLFLOW_ALLOW_FILE_STORE=true` before running `gridfm_graphkit` (`export` on Linux and macOS, `set` on Windows).

To run the tests, clone the repository and install with `".[test]"` in place of `gridfm-graphkit` in the command above, then run `pytest tests/`.
