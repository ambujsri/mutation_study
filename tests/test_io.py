"""Unit tests for ``src.common.io`` table reading and writing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common import constants as const
from src.common.io import read_table, write_table


@pytest.mark.parametrize("suffix", list(const.SUPPORTED_TABLE_FORMATS))
def test_round_trip(tmp_path: Path, labeled: pd.DataFrame, suffix: str) -> None:
    """A table written in each supported format reads back with identical values."""
    path = tmp_path / "nested" / f"table{suffix}"

    write_table(labeled, path)

    pd.testing.assert_frame_equal(read_table(path), labeled, check_dtype=False)


@pytest.mark.parametrize("func_name", ["write", "read"])
def test_unsupported_extension_raises(tmp_path: Path, labeled: pd.DataFrame, func_name: str) -> None:
    """Unsupported extensions are rejected on both read and write, listing supported ones."""
    path = tmp_path / "table.xlsx"

    with pytest.raises(ValueError, match=r"Unsupported .* format '\.xlsx'"):
        if func_name == "write":
            write_table(labeled, path)
        else:
            read_table(path)


def test_read_missing_file_raises(tmp_path: Path) -> None:
    """Reading a table that has not been produced yet points at the missing upstream stage."""
    with pytest.raises(FileNotFoundError, match="previous pipeline stage"):
        read_table(tmp_path / "absent.parquet")
