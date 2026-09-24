"""Copy a fixed prefix of processed samples to local tmp for disk-backed benchmarks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def local_tmp_root(network: str) -> Path:
    tmp_base = Path(os.environ.get("TMPDIR", "/tmp"))
    user = os.environ.get("USER", "user")
    return tmp_base / f"{user}_gridfm_{network}_benchmark_data"


def prepare_local_tmp_dataset(
    network: str,
    source_data_root: Path,
    *,
    max_samples: int,
) -> Path:
    local_root = local_tmp_root(network)
    local_processed = local_root / network / "processed"
    local_processed.mkdir(parents=True, exist_ok=True)

    source_processed = source_data_root / network / "processed"
    if not source_processed.exists():
        raise FileNotFoundError(f"Source processed directory not found: {source_processed}")

    copied = 0
    for idx in range(max_samples):
        src = source_processed / f"data_index_{idx}.pt"
        if not src.exists():
            break
        dst = local_processed / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        copied += 1

    if copied == 0:
        raise RuntimeError(f"No data_index_*.pt copied from {source_processed}")
    if copied < max_samples:
        raise RuntimeError(
            f"Expected at least {max_samples} processed samples for {network}, "
            f"but only found {copied} in {source_processed}",
        )

    print(f"Local tmp dataset ready at: {local_root}")
    print(f"Copied/verified {copied} processed sample files.")
    return local_root


def cleanup_local_tmp_dataset(network: str) -> None:
    local_root = local_tmp_root(network)
    if local_root.exists():
        shutil.rmtree(local_root)
        print(f"Removed local tmp dataset: {local_root}")
