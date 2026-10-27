"""Dataset integrity + preprocessing-reproducibility helpers for the integration tests.

Two concerns live here, kept out of ``test_base_set.py`` so both the PF and OPF
tests share one implementation:

1. ``ensure_dataset`` — download (once) and verify the on-disk layout of a
   gdrive-shipped dataset, re-downloading a single time if the first copy is
   incomplete, and failing hard if it is still broken.

2. ``verify_preprocessing`` — re-run the real raw->processed pipeline from the
   shipped ``raw/`` parquet tables and assert the freshly computed scenario
   tensors match the pre-generated ``processed/`` tensors shipped in the zip.
   This catches silent regressions in the preprocessing code that the
   metric-bounds assertions (which train on the shipped processed tensors) would
   never see.

The dataset layout produced by the gdrive zips is::

    <data_dir>/<grid_name>/raw/{bus,gen,branch}_data.parquet
    <data_dir>/<grid_name>/processed/processed_raw_files.done
    <data_dir>/<grid_name>/processed/data_index_<scenario>.pt
    <data_dir>/<grid_name>/processed/load_scenarios.pt   (optional)
"""

import os
import os.path as osp
import glob
import shutil
import zipfile

import gdown


RAW_FILES = ["bus_data.parquet", "gen_data.parquet", "branch_data.parquet"]
PROCESSED_DONE = "processed_raw_files.done"


def _grid_dirs(data_dir: str) -> list:
    """Return the per-grid subdirectories (each holding raw/ + processed/)."""
    if not osp.isdir(data_dir):
        return []
    return [
        d
        for d in sorted(glob.glob(osp.join(data_dir, "*")))
        if osp.isdir(d) and osp.isdir(osp.join(d, "raw"))
    ]


def _layout_problems(data_dir: str) -> list:
    """Return a list of human-readable problems with the dataset layout.

    Empty list means the layout is valid: at least one grid directory, each with
    the three raw parquet tables and a completed ``processed/`` directory
    containing the ``.done`` marker and at least one ``data_index_*.pt`` tensor.
    """
    problems = []
    grids = _grid_dirs(data_dir)
    if not grids:
        return [f"no grid directory with a raw/ subdir found under {data_dir!r}"]
    for grid in grids:
        name = osp.basename(grid)
        raw = osp.join(grid, "raw")
        for rf in RAW_FILES:
            # Accept either a flat parquet file or a Hive-partitioned directory.
            if not osp.exists(osp.join(raw, rf)):
                problems.append(f"{name}: missing raw table {rf}")
        processed = osp.join(grid, "processed")
        if not osp.isdir(processed):
            problems.append(f"{name}: missing processed/ directory")
            continue
        if not osp.exists(osp.join(processed, PROCESSED_DONE)):
            problems.append(f"{name}: missing processed marker {PROCESSED_DONE}")
        if not glob.glob(osp.join(processed, "data_index_*.pt")):
            problems.append(f"{name}: no processed data_index_*.pt tensors")
    return problems


def _download_and_extract(gdrive_file_id: str, zip_filename: str, data_dir: str) -> None:
    """Download the gdrive zip and extract it so its ``data_out`` becomes ``data_dir``.

    The zips contain a top-level ``data_out`` directory; this normalises it to
    whatever ``data_dir`` the caller uses (``data_out`` for PF, ``data_out_opf``
    for OPF).
    """
    gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    print(f"Downloading {zip_filename} from Google Drive...")
    gdown.download(gdrive_url, zip_filename, quiet=False)

    print(f"Extracting {zip_filename}...")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_filename, "r") as zf:
            zf.extractall(tmpdir)
        extracted = osp.join(tmpdir, "data_out")
        src = extracted if osp.isdir(extracted) else tmpdir
        if osp.exists(data_dir):
            shutil.rmtree(data_dir, ignore_errors=True)
        shutil.move(src, data_dir)
    os.remove(zip_filename)
    print(f"Data extracted to '{data_dir}'.")


def ensure_dataset(data_dir: str, gdrive_file_id: str, zip_filename: str) -> None:
    """Make sure ``data_dir`` holds a valid dataset, downloading at most twice.

    Logic:
      * If ``data_dir`` is absent/empty, download once.
      * Verify the layout. If it is broken, re-download a single time (the copy
        on disk may be a truncated/partial earlier download) and re-verify.
      * If it is still broken after the re-download, raise ``AssertionError``
        with the list of problems — we do not loop forever on a genuinely bad
        source zip.
    """
    need_download = not osp.exists(data_dir) or not os.listdir(data_dir)
    if need_download:
        print(f"Dataset '{data_dir}' not found or empty — downloading.")
        _download_and_extract(gdrive_file_id, zip_filename, data_dir)
    else:
        print(f"Dataset '{data_dir}' present — verifying integrity.")

    problems = _layout_problems(data_dir)
    if problems:
        print(
            "Dataset integrity check failed:\n  "
            + "\n  ".join(problems)
            + "\nRe-downloading once...",
        )
        _download_and_extract(gdrive_file_id, zip_filename, data_dir)
        problems = _layout_problems(data_dir)

    assert not problems, (
        f"Dataset at {data_dir!r} is still invalid after a re-download:\n  "
        + "\n  ".join(problems)
        + "\nThe shipped Google Drive zip may be corrupt or its layout changed."
    )
    print(f"Dataset integrity check passed for '{data_dir}'.")


def _load_scenario_tensors(processed_dir: str) -> dict:
    """Load every ``data_index_<scenario>.pt`` under ``processed_dir`` as HeteroData.

    Returns ``{scenario_id: HeteroData}``. Each file stores the output of
    ``HeteroData.to_dict()``; we rebuild with ``HeteroData.from_dict`` so the
    comparison is on tensor contents, not pickle bytes.
    """
    import torch
    from torch_geometric.data import HeteroData

    out = {}
    for path in glob.glob(osp.join(processed_dir, "data_index_*.pt")):
        scenario = osp.basename(path)[len("data_index_") : -len(".pt")]
        raw = torch.load(path, weights_only=False)
        data = HeteroData.from_dict(raw) if isinstance(raw, dict) else raw
        out[scenario] = data
    return out


def _compare_hetero(golden, fresh, scenario: str, rtol: float, atol: float) -> list:
    """Compare two HeteroData objects tensor-by-tensor; return list of diffs."""
    import torch

    diffs = []
    g_stores = {s._key: s for s in golden.stores if getattr(s, "_key", None)}
    f_stores = {s._key: s for s in fresh.stores if getattr(s, "_key", None)}
    if set(g_stores) != set(f_stores):
        diffs.append(
            f"scenario {scenario}: store keys differ "
            f"(golden={sorted(map(str, g_stores))} fresh={sorted(map(str, f_stores))})",
        )
        return diffs
    for key, g_store in g_stores.items():
        f_store = f_stores[key]
        g_attrs = {k for k, v in g_store.items() if torch.is_tensor(v)}
        f_attrs = {k for k, v in f_store.items() if torch.is_tensor(v)}
        if g_attrs != f_attrs:
            diffs.append(
                f"scenario {scenario} store {key}: tensor attrs differ "
                f"(golden={sorted(g_attrs)} fresh={sorted(f_attrs)})",
            )
            continue
        for attr in sorted(g_attrs):
            gt, ft = g_store[attr], f_store[attr]
            if gt.shape != ft.shape:
                diffs.append(
                    f"scenario {scenario} {key}.{attr}: shape {tuple(gt.shape)} "
                    f"vs {tuple(ft.shape)}",
                )
                continue
            if not torch.allclose(gt.float(), ft.float(), rtol=rtol, atol=atol):
                max_abs = (gt.float() - ft.float()).abs().max().item()
                diffs.append(
                    f"scenario {scenario} {key}.{attr}: max abs diff {max_abs:.3e} "
                    f"exceeds rtol={rtol} atol={atol}",
                )
    return diffs


def verify_preprocessing(
    data_dir: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Re-run preprocessing from raw and assert it reproduces the shipped tensors.

    For each grid under ``data_dir``:
      1. Snapshot the shipped ``processed/`` directory as the golden reference.
      2. Delete ``processed/`` and re-instantiate ``HeteroGridDatasetDisk``,
         which triggers ``process()`` to rebuild the tensors from ``raw/``.
      3. Compare every regenerated ``data_index_*.pt`` against the golden copy
         via ``torch.allclose`` and fail on any divergence.
      4. Restore the shipped ``processed/`` so the subsequent training step uses
         the original tensors (and no re-download is needed next run).

    The dataset's normalizer is only used at access time (``get()``), never in
    ``process()``, so ``None`` is passed as the normalizer here.
    """
    from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk

    grids = _grid_dirs(data_dir)
    assert grids, f"No grid directory found under {data_dir!r} to preprocess."

    all_diffs = []
    for grid in grids:
        name = osp.basename(grid)
        processed = osp.join(grid, "processed")
        golden_backup = osp.join(grid, "processed_golden")

        print(f"\n[preprocess-check] grid '{name}': snapshotting shipped tensors.")
        if osp.exists(golden_backup):
            shutil.rmtree(golden_backup, ignore_errors=True)
        shutil.copytree(processed, golden_backup)

        try:
            # Force a fresh process() by removing the processed cache.
            shutil.rmtree(processed, ignore_errors=True)
            print(f"[preprocess-check] grid '{name}': re-running process() from raw.")
            HeteroGridDatasetDisk(root=grid, data_normalizer=None)

            golden = _load_scenario_tensors(golden_backup)
            fresh = _load_scenario_tensors(processed)

            assert golden, f"grid '{name}': no golden data_index_*.pt tensors found"
            if set(golden) != set(fresh):
                all_diffs.append(
                    f"grid '{name}': scenario set differs "
                    f"(golden has {len(golden)}, fresh has {len(fresh)})",
                )
            for scenario in sorted(set(golden) & set(fresh)):
                all_diffs.extend(
                    _compare_hetero(
                        golden[scenario],
                        fresh[scenario],
                        scenario,
                        rtol,
                        atol,
                    ),
                )
            print(
                f"[preprocess-check] grid '{name}': compared {len(golden)} scenarios.",
            )
        finally:
            # Restore the shipped processed tensors for the training step.
            shutil.rmtree(processed, ignore_errors=True)
            shutil.move(golden_backup, processed)

    assert not all_diffs, (
        "Preprocessing from raw did NOT reproduce the shipped processed tensors "
        f"(rtol={rtol}, atol={atol}). This indicates a regression in the "
        "raw->processed pipeline:\n  " + "\n  ".join(all_diffs[:50])
        + ("\n  ..." if len(all_diffs) > 50 else "")
    )
    print("\n[preprocess-check] PASSED: preprocessing reproduces shipped tensors.")
