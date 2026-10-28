# Apptainer image (HPC)

[Apptainer](https://apptainer.org/) (formerly Singularity) is the container runtime available on
most HPC clusters, where Docker is usually not allowed. The repository ships
[`gridfm-graphkit.def`](https://github.com/gridfm/gridfm-graphkit/blob/main/gridfm-graphkit.def),
which builds a **GPU** image of gridfm-graphkit from the local checkout.

The definition only uses public sources (Docker Hub `ubuntu:24.04`, PyPI,
`download.pytorch.org` and `data.pyg.org`), so anyone with Apptainer can build it.

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

srun apptainer exec --nv --bind "$PWD" gridfm-graphkit.sif \
    gridfm_graphkit train --config examples/config/HGNS_PF_datakit_case14.yaml \
        --data_path "$PWD/data" --log_dir "$PWD/mlruns"
```

## Notes

* Python 3.12 and the torch stack are pinned as in `pyproject.toml` (`torch>=2.10,<2.13`).
  `torch` and `torch-scatter` are installed first and frozen in
  `/opt/venv/torch-constraints.txt`, so installing gridfm-graphkit cannot replace them.
* `gridfm-datakit` is installed as a dependency of graphkit, but this image does not include
  Julia. Generating data with `gridfm_datakit generate` needs the Docker `Containerfile`
  (or a Julia installation); training on existing datasets works.
* `MLFLOW_ALLOW_FILE_STORE=true` is set in the image, as in the Docker image.
