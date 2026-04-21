"""
Utilities for writing MLflow artifacts that work with both local and remote
MLflow tracking servers.

Two helpers are provided:

``artifact_context`` – context manager::

    with artifact_context(logger, "stats") as local_dir:
        torch.save(my_data, os.path.join(local_dir, "data.pt"))

``artifact_write_ctx`` – imperative style (no indentation change required)::

    artifact_dir, _upload = artifact_write_ctx(logger)
    # … write files into artifact_dir …
    _upload()   # uploads to MLflow; no-op for other loggers

For MLflow loggers both helpers write to a temporary directory then call
``MlflowClient.log_artifacts``, which works for **local and remote** tracking
servers alike.  For other loggers files are written directly under
``logger.save_dir``.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Callable, Tuple

from lightning.pytorch.loggers import MLFlowLogger


@contextmanager
def artifact_context(logger, artifact_subpath: str = ""):
    """Context manager that yields a local directory for writing artifacts.

    On exit the directory contents are committed to the artifact store:

    * **MLflow logger** – files are written to a temporary directory and then
      uploaded via ``MlflowClient.log_artifacts``.  This works with both
      local file-based tracking (``mlruns/``) and remote tracking servers
      (HTTP/HTTPS, Databricks, etc.).
    * **Other loggers** – files are written directly to
      ``os.path.join(logger.save_dir, artifact_subpath)``.

    Parameters
    ----------
    logger:
        The Lightning logger attached to the trainer.
    artifact_subpath:
        Sub-directory within the artifact store to place the files under
        (e.g. ``"stats"``, ``"model"``, ``"test"``).  An empty string
        targets the artifact root.
    """
    if isinstance(logger, MLFlowLogger):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
            logger.experiment.log_artifacts(
                logger.run_id,
                tmpdir,
                artifact_subpath if artifact_subpath else None,
            )
    else:
        local_dir = (
            os.path.join(logger.save_dir, artifact_subpath)
            if artifact_subpath
            else logger.save_dir
        )
        os.makedirs(local_dir, exist_ok=True)
        yield local_dir


def artifact_write_ctx(logger) -> Tuple[str, Callable[[], None]]:
    """Imperative alternative to :func:`artifact_context`.

    Returns a ``(local_dir, upload_fn)`` tuple.  Write all artifacts into
    ``local_dir`` (creating sub-directories as needed), then call
    ``upload_fn()`` to commit them to the artifact store.

    * **MLflow logger** – ``local_dir`` is a fresh temporary directory;
      ``upload_fn`` uploads its contents via ``MlflowClient.log_artifacts``
      and then removes the temporary directory.
    * **Other loggers** – ``local_dir`` is ``logger.save_dir``; ``upload_fn``
      is a no-op.

    Parameters
    ----------
    logger:
        The Lightning logger attached to the trainer.
    """
    if isinstance(logger, MLFlowLogger):
        tmpdir = tempfile.mkdtemp()

        def _upload() -> None:
            logger.experiment.log_artifacts(logger.run_id, tmpdir)
            shutil.rmtree(tmpdir, ignore_errors=True)

        return tmpdir, _upload
    else:
        local_dir = logger.save_dir
        os.makedirs(local_dir, exist_ok=True)
        return local_dir, lambda: None
