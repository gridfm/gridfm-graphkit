# Docker / container image

The repository ships a [`Containerfile`](https://github.com/gridfm/gridfm-graphkit/blob/main/Containerfile)
that installs the latest **gridfm-datakit** and **gridfm-graphkit** from PyPI on top of a
CPU-only PyTorch stack. It is a plain OCI image, so it builds and runs identically with
either `docker` or `podman` — substitute whichever you use for the `docker` calls below.

The image is self-contained: the Julia toolchain that `gridfm-datakit` uses for power-flow
solving (PowerModels, Ipopt, Memento) is installed at build time, and MLflow's local file
tracking store is enabled, so both data generation and training work out of the box.

> **Why a dedicated Containerfile?** `torch-scatter`/`torch-sparse` ship compiled extensions
> that must match the exact PyTorch build, and `gridfm-datakit` currently needs
> `juliapkg < 0.1.24`. The Containerfile pins these precisely; see its inline comments for the
> full reasoning.

## Build the image

```bash
git clone https://github.com/gridfm/gridfm-graphkit.git
cd gridfm-graphkit
docker build -f Containerfile -t gridfm:latest .
```

The build downloads the CPU PyTorch wheels and the Julia packages, so the first build takes
several minutes. The result is a CPU-only image (no CUDA wheels).

Running the image with no arguments executes a built-in smoke test that imports both packages
and prints their versions:

```bash
docker run --rm gridfm:latest
```

```text
torch 2.12.1+cpu
torch-scatter 2.1.2
gridfm-datakit 1.1.0
gridfm-graphkit 0.9.0
OK
```

## Hello world: generate data, then train

This runs the complete `datakit → graphkit` pipeline end to end on a tiny IEEE case-14 grid.
It finishes in well under a minute on CPU and is meant purely to prove the pipeline works — the
model is not trained to convergence.

The two config files live in the repo under
[`examples/config/`](https://github.com/gridfm/gridfm-graphkit/tree/main/examples/config).
Copy them into a working directory that we will mount into the container as `/work`:

```bash
mkdir -p work
cp examples/config/hello_world_datagen_case14.yaml work/
cp examples/config/hello_world_train_case14.yaml   work/
```

> **SELinux note (Fedora/RHEL):** the `:Z` suffix on the volume relabels it so the container
> can read/write it. On other systems it is harmless, but you can drop it: use
> `-v "$PWD/work:/work"`.

**Step 1 — generate a small power-flow dataset with gridfm-datakit:**

```bash
docker run --rm -v "$PWD/work:/work:Z" gridfm:latest \
    gridfm_datakit generate /work/hello_world_datagen_case14.yaml
```

This writes Hive-partitioned parquet data to `work/data/case14_ieee/raw/`
(`bus_data.parquet/`, `gen_data.parquet/`, `branch_data.parquet/`, …) — exactly the layout
gridfm-graphkit expects under `<data_path>/<network>/raw/`.

**Step 2 — train a gridfm-graphkit model for one epoch on that data:**

```bash
docker run --rm -v "$PWD/work:/work:Z" gridfm:latest \
    gridfm_graphkit train \
        --config /work/hello_world_train_case14.yaml \
        --data_path /work/data \
        --log_dir /work/mlruns
```

You should see a Lightning training summary, one training epoch, and a validation/test pass,
ending with `Trainer.fit stopped: max_epochs=1 reached`. MLflow run artifacts are written to
`work/mlruns/`.

That's the whole loop: **`gridfm_datakit generate` → `gridfm_graphkit train`**, with the
generated dataset handed straight to training via `--data_path`. Scale it up by raising
`scenarios` in the datagen config (and matching `data.scenarios` in the training config),
adding networks, or switching to one of the fuller configs in `examples/config/`.

## Use the image as a VS Code Dev Container

You can develop *inside* this image with VS Code's
[Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) extension, so
your editor, terminal, and debugger all run against the exact same pinned environment.

1. Install the **Dev Containers** extension (`ms-vscode-remote.remote-containers`). It works
   with Docker or Podman (for Podman, set *Dev › Containers: Docker Path* to `podman` in
   settings).

2. The repository already contains a
   [`.devcontainer/devcontainer.json`](https://github.com/gridfm/gridfm-graphkit/blob/main/.devcontainer/devcontainer.json)
   that builds the `Containerfile` and mounts your checkout into `/workspaces`:

    ```json
    {
      "name": "gridfm",
      "build": { "dockerfile": "../Containerfile", "context": ".." },
      "workspaceFolder": "/workspaces/gridfm-graphkit",
      "customizations": {
        "vscode": {
          "extensions": ["ms-python.python", "ms-toolsai.jupyter"],
          "settings": { "python.defaultInterpreterPath": "/usr/local/bin/python" }
        }
      }
    }
    ```

3. Open the repository folder in VS Code and run **Dev Containers: Reopen in Container** from
   the command palette (`F1`). VS Code builds the image (first time only) and drops you into a
   shell inside it, with your working copy live-mounted.

4. Inside the container everything from the hello-world above is on `PATH`. The shipped
   datagen config writes to its `settings.data_dir` (`/work/data`), so point the `train`
   step at the same location — the two stay consistent as a copy-paste run:

    ```bash
    gridfm_datakit generate examples/config/hello_world_datagen_case14.yaml
    gridfm_graphkit train \
        --config examples/config/hello_world_train_case14.yaml \
        --data_path /work/data \
        --log_dir /work/mlruns
    ```

    Data and MLflow runs land under `/work` **inside the container**, not in your mounted
    workspace. To keep them in the workspace instead, copy the config and set
    `settings.data_dir` to a workspace-relative path (e.g. `data`), then use a matching
    `--data_path data`.

To iterate on the GridFM source itself rather than the released wheels, add an editable install
as a `postCreateCommand` in `devcontainer.json`, e.g.
`"postCreateCommand": "pip install -e ."`.
