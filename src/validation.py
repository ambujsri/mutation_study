"""Cross-validation strategies that control protein- and family-level leakage.

Phase 2 of the Medvolt.AI mutation-effect pipeline.

Why grouping matters here: the labeled set has many mutations per protein and
several proteins per family, and proteins in a family share similar embeddings.
A random mutation-level split puts sibling mutations of the same protein in
both training and validation, so a model can score well by recognising the
protein rather than by learning mutation effects. A third of the candidates
come from families absent from training, so the honest estimate of deployment
performance is the family-grouped one.

Strategies (``src.common.constants.SplitStrategy``):
    * ``random``         — ``KFold`` over mutations; optimistic baseline only.
    * ``protein_group``  — ``GroupKFold`` by ``protein_id``; tests transfer to
                           unseen proteins from seen families.
    * ``family_group``   — ``GroupKFold`` by ``protein_family``; tests transfer to
                           unseen families (the OOD scenario of the candidates).

Every set of folds is checked before it is returned: each row is validated
exactly once, train and validation never share a row, and grouped strategies
never share a group. Fold indices are **positional** (use ``.iloc``).

Run directly::

    python -m src.validation --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table, write_table

logger = logging.getLogger(__name__)

IndexArray = npt.NDArray[np.intp]


class LeakageError(RuntimeError):
    """Raised when folds share rows or, for grouped strategies, share groups."""


@dataclass(frozen=True)
class Fold:
    """One train/validation split.

    Attributes:
        strategy: Strategy that produced the split.
        index: Zero-based fold number.
        train_idx: Positional row indices used for training.
        val_idx: Positional row indices used for validation.
    """

    strategy: const.SplitStrategy
    index: int
    train_idx: IndexArray
    val_idx: IndexArray


# =============================================================================
# Splitting
# =============================================================================
def group_labels(df: pd.DataFrame, strategy: const.SplitStrategy) -> pd.Series | None:
    """Return the grouping column for ``strategy`` (``None`` for random splits).

    Raises:
        KeyError: If the grouping column is absent from ``df``.
    """
    column = const.SPLIT_GROUP_COLUMN[strategy]
    if column is None:
        return None
    if column not in df.columns:
        raise KeyError(f"Strategy '{strategy.value}' needs column '{column}', absent from data")
    return df[column]


def check_feasible(df: pd.DataFrame, strategy: const.SplitStrategy, n_splits: int) -> None:
    """Ensure there are enough rows / groups to form ``n_splits`` non-empty folds.

    Raises:
        ValueError: If the data cannot support the requested number of folds.
    """
    groups = group_labels(df, strategy)
    n_units = len(df) if groups is None else groups.nunique()
    unit = "rows" if groups is None else f"distinct {groups.name} groups"
    if n_units < n_splits:
        raise ValueError(
            f"Strategy '{strategy.value}' needs at least n_splits={n_splits} {unit}, found {n_units}. "
            "Lower validation.n_splits in config."
        )
    if groups is not None and groups.isna().any():
        raise ValueError(f"Strategy '{strategy.value}': {int(groups.isna().sum())} rows have no group")


def make_splitter(
    strategy: const.SplitStrategy, n_splits: int, shuffle: bool, seed: int
) -> KFold | GroupKFold:
    """Build the scikit-learn splitter for ``strategy``.

    The seed is passed only when ``shuffle`` is true; scikit-learn rejects a
    ``random_state`` without shuffling because it would have no effect.
    """
    random_state = seed if shuffle else None
    if strategy is const.SplitStrategy.RANDOM:
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def generate_folds(
    df: pd.DataFrame, strategy: const.SplitStrategy, n_splits: int, shuffle: bool, seed: int
) -> list[Fold]:
    """Split ``df`` into folds with ``strategy`` and verify them.

    Args:
        df: Training table (one row per labeled mutation).
        strategy: Splitting strategy.
        n_splits: Number of folds.
        shuffle: Whether to shuffle rows (random) or groups (grouped) first.
        seed: Random seed used when shuffling.

    Returns:
        Verified folds, in order.

    Raises:
        ValueError: If the split is infeasible.
        LeakageError: If verification fails.
    """
    check_feasible(df, strategy, n_splits)
    splitter = make_splitter(strategy, n_splits, shuffle, seed)
    groups = group_labels(df, strategy)
    folds = [
        Fold(strategy, i, np.asarray(train, dtype=np.intp), np.asarray(val, dtype=np.intp))
        for i, (train, val) in enumerate(splitter.split(df, groups=groups))
    ]
    verify_folds(df, folds)
    return folds


# =============================================================================
# Verification & statistics
# =============================================================================
def shared_groups(df: pd.DataFrame, fold: Fold, column: str) -> set[Any]:
    """Return the values of ``column`` present in both the train and validation rows of ``fold``."""
    values = df[column].to_numpy()
    return set(values[fold.train_idx]) & set(values[fold.val_idx])


def verify_folds(df: pd.DataFrame, folds: list[Fold]) -> None:
    """Assert the folds form a clean partition with no leakage for their strategy.

    Checks:
        * no row is in both train and validation of the same fold;
        * train + validation cover every row in each fold;
        * across folds, every row is validated exactly once;
        * grouped strategies share no group between train and validation.

    Raises:
        LeakageError: Describing the first violation found.
    """
    n_rows = len(df)
    val_counts = np.zeros(n_rows, dtype=int)
    for fold in folds:
        label = f"{fold.strategy.value} fold {fold.index}"
        if np.intersect1d(fold.train_idx, fold.val_idx).size:
            raise LeakageError(f"{label}: rows in both train and validation")
        if fold.train_idx.size + fold.val_idx.size != n_rows:
            raise LeakageError(f"{label}: train + validation do not cover all {n_rows} rows")
        val_counts[fold.val_idx] += 1
        column = const.SPLIT_GROUP_COLUMN[fold.strategy]
        if column is not None:
            leaked = shared_groups(df, fold, column)
            if leaked:
                raise LeakageError(f"{label}: {column} values in train and validation: {sorted(leaked)}")
    if not (val_counts == 1).all():
        raise LeakageError(
            f"{folds[0].strategy.value}: {int((val_counts != 1).sum())} rows not validated exactly once"
        )


def summarize_folds(df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Tabulate per-fold sizes, group coverage, cross-split group overlap and target means.

    The ``leaked_*`` columns quantify how much protein / family identity a
    strategy lets the model see at validation time; they are expected to be
    non-zero for ``random`` and zero for the matching grouped strategy.
    """
    rows = []
    for fold in folds:
        val = df.iloc[fold.val_idx]
        row: dict[str, Any] = {
            const.CV_COL_STRATEGY: fold.strategy.value,
            const.CV_COL_FOLD: fold.index,
            const.CV_COL_N_TRAIN: fold.train_idx.size,
            const.CV_COL_N_VAL: fold.val_idx.size,
            const.CV_COL_VAL_FRACTION: fold.val_idx.size / len(df),
        }
        for level, column in const.GROUP_COLUMNS.items():
            row[f"{const.CV_PREFIX_N_VAL_GROUPS}{level}"] = val[column].nunique()
        row[const.CV_COL_LEAKED_PROTEINS] = len(shared_groups(df, fold, const.COL_PROTEIN_ID))
        row[const.CV_COL_LEAKED_FAMILIES] = len(shared_groups(df, fold, const.COL_PROTEIN_FAMILY))
        for target in const.TARGETS.values():
            if target in df.columns:
                row[f"{const.CV_PREFIX_VAL_MEAN}{target}"] = float(val[target].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def log_fold_summary(summary: pd.DataFrame) -> None:
    """Log one line per fold plus the across-fold spread of validation sizes."""
    for record in summary.to_dict(orient="records"):
        logger.info(
            "[%s] fold %d: train=%d val=%d (%.1f%%) | val proteins=%d families=%d | "
            "proteins in both=%d families in both=%d",
            record[const.CV_COL_STRATEGY],
            record[const.CV_COL_FOLD],
            record[const.CV_COL_N_TRAIN],
            record[const.CV_COL_N_VAL],
            100 * record[const.CV_COL_VAL_FRACTION],
            record[f"{const.CV_PREFIX_N_VAL_GROUPS}protein"],
            record[f"{const.CV_PREFIX_N_VAL_GROUPS}family"],
            record[const.CV_COL_LEAKED_PROTEINS],
            record[const.CV_COL_LEAKED_FAMILIES],
        )
    sizes = summary[const.CV_COL_N_VAL]
    logger.info(
        "[%s] validation size min=%d max=%d (imbalance ratio %.2f)",
        summary[const.CV_COL_STRATEGY].iloc[0],
        sizes.min(),
        sizes.max(),
        sizes.max() / sizes.min(),
    )


# =============================================================================
# Harness
# =============================================================================
def build_cv_splits(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[dict[const.SplitStrategy, list[Fold]], pd.DataFrame]:
    """Generate, verify and summarise folds for every strategy listed in config.

    Args:
        df: Processed training table.
        cfg: Validated configuration.

    Returns:
        ``(splits, summary)``: folds keyed by strategy, and one summary row per
        fold across all strategies.
    """
    v = cfg["validation"]
    seed = cfg["project"]["random_seed"]
    splits: dict[const.SplitStrategy, list[Fold]] = {}
    summaries = []
    for name in v["strategies"]:
        strategy = const.SplitStrategy(name)
        splits[strategy] = generate_folds(df, strategy, v["n_splits"], v["shuffle"], seed)
        summary = summarize_folds(df, splits[strategy])
        log_fold_summary(summary)
        summaries.append(summary)
    logger.info(
        "Built %d strategies x %d folds; primary strategy: %s",
        len(splits),
        v["n_splits"],
        v["primary_strategy"],
    )
    return splits, pd.concat(summaries, ignore_index=True)


def iter_folds(
    splits: dict[const.SplitStrategy, list[Fold]],
) -> Iterator[tuple[const.SplitStrategy, int, IndexArray, IndexArray]]:
    """Yield ``(strategy, fold_index, train_idx, val_idx)`` across all strategies."""
    for strategy, folds in splits.items():
        for fold in folds:
            yield strategy, fold.index, fold.train_idx, fold.val_idx


def run_validation_harness(
    cfg: dict[str, Any],
) -> tuple[dict[const.SplitStrategy, list[Fold]], pd.DataFrame]:
    """Load the processed training table, build all CV splits and write the fold summary.

    Returns:
        ``(splits, summary)`` as from ``build_cv_splits``.
    """
    df = read_table(resolve_path(cfg, "processed_dir", "processed_train"))
    logger.info("Loaded processed training table shape=%s", df.shape)
    splits, summary = build_cv_splits(df, cfg)
    out_path = resolve_path(cfg, "reports_dir", "cv_fold_summary")
    write_table(summary, out_path)
    logger.info("Wrote CV fold summary -> %s", out_path)
    return splits, summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    run_validation_harness(config)
