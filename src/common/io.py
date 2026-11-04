"""Table I/O shared by pipeline stages; the format is chosen by file extension."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.constants import SUPPORTED_TABLE_FORMATS


def _check_format(path: Path, direction: str) -> None:
    """Raise if ``path`` has an extension outside ``SUPPORTED_TABLE_FORMATS``."""
    if path.suffix not in SUPPORTED_TABLE_FORMATS:
        raise ValueError(f"Unsupported {direction} format '{path.suffix}'; use {SUPPORTED_TABLE_FORMATS}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` as Parquet or CSV (no index), creating parent folders.

    Raises:
        ValueError: If the extension is not supported.
    """
    _check_format(path, "output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    """Read a Parquet or CSV table written by ``write_table``.

    Raises:
        ValueError: If the extension is not supported.
        FileNotFoundError: If ``path`` does not exist.
    """
    _check_format(path, "input")
    if not path.is_file():
        raise FileNotFoundError(f"Table not found: {path} (has the previous pipeline stage been run?)")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
