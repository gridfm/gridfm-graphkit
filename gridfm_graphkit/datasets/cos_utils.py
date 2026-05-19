"""
Cloud Object Storage (COS) utilities for gridfm-graphkit.

Supports URLs of the form:
    cos://ACCESS_KEY:SECRET_KEY@ENDPOINT:PORT/BUCKET/path/to/data

Files are downloaded **lazily, one at a time**, as each sample is accessed
by the dataloader.  Nothing is downloaded up-front.

Example
-------
    >>> from gridfm_graphkit.datasets.cos_utils import COSHeteroGridDataset
    >>> dataset = COSHeteroGridDataset(
    ...     cos_url="cos://mykey:mysecret@s3.us-south.cloud-object-storage.appdomain.cloud:443/mybucket/case14",
    ...     data_normalizer=normalizer,
    ... )
    >>> sample = dataset[0]  # downloads only data_index_0.pt from COS
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse, unquote

_COS_SCHEME = "cos"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def is_cos_url(path: str) -> bool:
    """Return ``True`` if *path* is a COS URL (``cos://…``)."""
    return isinstance(path, str) and path.startswith(f"{_COS_SCHEME}://")


def parse_cos_url(url: str) -> dict:
    """Parse a ``cos://`` URL into its components.

    URL format::

        cos://ACCESS_KEY:SECRET_KEY@ENDPOINT:PORT/BUCKET/optional/prefix

    Returns
    -------
    dict with keys: access_key, secret_key, endpoint_url, bucket, prefix
    """
    parsed = urlparse(url)
    if parsed.scheme != _COS_SCHEME:
        raise ValueError(f"Expected scheme 'cos', got '{parsed.scheme}'")

    access_key = unquote(parsed.username or "")
    secret_key = unquote(parsed.password or "")

    host = parsed.hostname or ""
    port = parsed.port
    endpoint_url = f"https://{host}" if port is None else f"https://{host}:{port}"

    # Path is "/BUCKET/prefix/…" — strip leading slash
    path_parts = parsed.path.lstrip("/").split("/", 1)
    bucket = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""

    return {
        "access_key": access_key,
        "secret_key": secret_key,
        "endpoint_url": endpoint_url,
        "bucket": bucket,
        "prefix": prefix,
    }


def _make_cache_dir(endpoint_url: str, bucket: str, prefix: str, base_dir: str | None) -> Path:
    """Return (and create) a deterministic local cache directory."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{endpoint_url}_{bucket}_{prefix}")
    if base_dir is None:
        base_dir = os.path.join(tempfile.gettempdir(), "gridfm_cos_cache")
    cache_path = Path(base_dir) / safe
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _get_s3_client(cos_params: dict):
    """Return a boto3 S3 client configured for the given COS params."""
    try:
        import boto3
        from botocore.client import Config
    except ImportError as exc:
        raise ImportError(
            "The 'boto3' package is required for COS support. "
            "Install it with:  pip install 'gridfm-graphkit[cos]'"
        ) from exc

    return boto3.client(
        "s3",
        endpoint_url=cos_params["endpoint_url"],
        aws_access_key_id=cos_params["access_key"],
        aws_secret_access_key=cos_params["secret_key"],
        config=Config(signature_version="s3v4"),
    )


# ---------------------------------------------------------------------------
# Single-file download
# ---------------------------------------------------------------------------

def download_file_from_cos(
    cos_params: dict,
    relative_path: str,
    local_dir: Path,
    force: bool = False,
) -> Path:
    """Download a single object from COS identified by *relative_path*.

    The object key is constructed as ``<prefix>/<relative_path>``.

    Parameters
    ----------
    cos_params:
        Dict returned by :func:`parse_cos_url`.
    relative_path:
        Path of the file relative to the COS prefix (e.g. ``processed/data_index_0.pt``).
    local_dir:
        Local root directory that mirrors the COS prefix.
    force:
        Re-download even when the local file already exists.

    Returns
    -------
    Path
        Absolute path to the local file.
    """
    local_file = local_dir / relative_path
    if local_file.exists() and not force:
        return local_file

    bucket = cos_params["bucket"]
    prefix = cos_params["prefix"].rstrip("/")
    key = f"{prefix}/{relative_path}" if prefix else relative_path

    local_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[COS] Downloading s3://{bucket}/{key} → {local_file}")
    _get_s3_client(cos_params).download_file(bucket, key, str(local_file))
    return local_file

