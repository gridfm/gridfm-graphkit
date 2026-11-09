# Apptainer image (HPC)

[Apptainer](https://apptainer.org/) (formerly Singularity) is the container runtime available on
most HPC clusters, where Docker is usually not allowed. The repository ships
[`gridfm-graphkit.def`](https://github.com/gridfm/gridfm-graphkit/blob/main/gridfm-graphkit.def),
which builds a **GPU** image of gridfm-graphkit from the local checkout.

The definition only uses public sources (Docker Hub `ubuntu:24.04`, PyPI,
`download.pytorch.org` and `data.pyg.org`), so anyone with Apptainer can build it.

## Prerequisites

On a cluster these are normally already set up by the administrators:

* **Apptainer**: `sudo apt install apptainer` (or a release `.deb` from
  <https://github.com/apptainer/apptainer/releases>). A `.deb` built for another Ubuntu release
  may depend on library names that do not exist on yours (e.g. `libfuse3-3` vs `libfuse3-4`);
  prefer the distribution package in that case.
* **`--fakeroot` support**: install `uidmap` (`sudo apt install uidmap`) and make sure your user
  has an entry in `/etc/subuid` and `/etc/subgid`. Without `newuidmap`, the build stops with
  `newuidmap was not found in PATH`.
* **Ubuntu 24.04 and later**: AppArmor blocks unprivileged user namespaces, and the build fails with
  `Failed to create mount namespace`. Allow them for Apptainer only:

    ```bash
    printf '%s\n' 'abi <abi/4.0>,' 'include <tunables/global>' '' \
        'profile apptainer /usr/lib/x86_64-linux-gnu/apptainer/bin/starter{,-suid} flags=(unconfined) {' \
        '  userns,' '  include if exists <local/apptainer>' '}' \
        | sudo tee /etc/apparmor.d/apptainer
    sudo systemctl reload apparmor
    ```

* **Disk space and bandwidth**: the CUDA PyTorch wheels are over 3 GB. On a slow or unstable
  connection `uv` can time out (`operation timed out` on an `nvidia-*` wheel); raise
  `UV_HTTP_TIMEOUT` and `UV_HTTP_RETRIES` or build on a node with a fast connection.
* **Optional**: `sudo apt install squashfuse` lets Apptainer mount the `.sif` directly instead of
  unpacking it to a temporary sandbox at every launch.
* **GPU**: only the NVIDIA driver is needed on the host. Without a GPU the image still runs on
  CPU, and `apptainer run` reports `CUDA device available: False`.

## Build the image

```bash
git clone https://github.com/gridfm/gridfm-graphkit.git
cd gridfm-graphkit
apptainer build --fakeroot gridfm-graphkit.sif gridfm-graphkit.def
```

* Run the build from the repository root: the definition copies `pyproject.toml`, `README.md`,
  `LICENSE` and `gridfm_graphkit/` from the current directory.
* `--fakeroot` builds without root privileges. If your site does not enable it, build on a
  machine where you have root (`sudo apptainer build ...`) and copy the `.sif` file.
* On a cluster, build on a node with internet access and enough space in `APPTAINER_TMPDIR`
  and `APPTAINER_CACHEDIR` (set them to a scratch directory if your home quota is small).
* The PyTorch version and CUDA flavour can be changed at build time. They must match the
  wheels published on <https://data.pyg.org/whl/> for `torch-scatter`:

    ```bash
    apptainer build --fakeroot \
        --build-arg TORCH_VERSION=2.12.0 --build-arg CUDA_TAG=cu126 \
        gridfm-graphkit.sif gridfm-graphkit.def
    ```

## Run

Use `--nv` to expose the host NVIDIA driver; only the driver is needed on the host, the CUDA
runtime is part of the PyTorch wheels.

```bash
# Smoke test (versions, CUDA availability)
apptainer run --nv gridfm-graphkit.sif

# Train
apptainer exec --nv gridfm-graphkit.sif \
    gridfm_graphkit train --config examples/config/HGNS_PF_datakit_case14.yaml \
        --data_path /path/to/data --log_dir /path/to/mlruns
```

The image is read-only: write data, checkpoints and MLflow runs to a bind-mounted directory
(`--bind /scratch/me:/scratch/me`). `$HOME`, `/tmp` and the current directory are mounted by
default.

## Slurm example

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

srun apptainer exec --nv gridfm-graphkit.sif \
    gridfm_graphkit train --config examples/config/HGNS_PF_datakit_case14.yaml \
        --data_path "$PWD/data" --log_dir "$PWD/mlruns"
```

## Notes

* Python 3.12 and the torch stack are pinned as in `pyproject.toml` (`torch>=2.10,<2.13`).
  `torch`, `torch-scatter` and `torch-sparse` (RRWP encodings) are installed first and frozen in
  `/opt/venv/torch-constraints.txt`, so installing gridfm-graphkit cannot replace them.
* `gridfm-datakit` is installed as a dependency of graphkit, but this image does not include
  Julia. Generating data with `gridfm_datakit generate` needs the Docker `Containerfile`
  (or a Julia installation); training on existing datasets works.
* Packages are installed with [uv](https://docs.astral.sh/uv/) (`uv pip`), but `uv.lock` is not used:
  it pins CPU wheels for torch-scatter on Linux, so the GPU image resolves from PyPI at build time.
  The exact result is saved in `/opt/venv/pip-freeze.txt`.
* `MLFLOW_ALLOW_FILE_STORE=true` is set in the image, as in the Docker image.
