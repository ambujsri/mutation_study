"""Unit and integration tests for ``src.validation`` (Phase 2 cross-validation).

The fixture table mimics the leakage structure of the real data — several
families, several proteins per family, several mutations per protein, with
uneven group sizes — so grouped splitters have something to get wrong.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import validation as val
from src.common import constants as const
from src.common.io import read_table, write_table

# Synthetic layout: 7 families (as in the real labeled set) x 3 proteins, uneven mutation counts.
N_FAMILIES: int = 7
PROTEINS_PER_FAMILY: int = 3
MUTATIONS_PER_PROTEIN: tuple[int, ...] = (4, 6, 9)

ALL_STRATEGIES: list[const.SplitStrategy] = list(const.SplitStrategy)
GROUPED_STRATEGIES: list[const.SplitStrategy] = [
    s for s in const.SplitStrategy if const.SPLIT_GROUP_COLUMN[s] is not None
]


@pytest.fixture()
def grouped_df(rng: np.random.Generator) -> pd.DataFrame:
    """Training-like table with nested family → protein → mutation structure and both targets."""
    rows = []
    for f in range(N_FAMILIES):
        for p in range(PROTEINS_PER_FAMILY):
            for m in range(MUTATIONS_PER_PROTEIN[p]):
                rows.append(
                    {
                        const.COL_MUTATION_ID: f"M_{f}_{p}_{m}",
                        const.COL_PROTEIN_ID: f"P_{f}_{p}",
                        const.COL_PROTEIN_FAMILY: f"F_{f}",
                    }
                )
    df = pd.DataFrame(rows)
    df[const.TARGET_STABILITY] = rng.normal(size=len(df))
    df[const.TARGET_FUNCTION] = rng.uniform(size=len(df))
    return df


@pytest.fixture()
def cv_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config with all strategies enabled and the project's fold count and shuffle setting."""
    cfg["validation"]["strategies"] = [s.value for s in ALL_STRATEGIES]
    return cfg


def folds_for(
    df: pd.DataFrame, strategy: const.SplitStrategy, cfg: dict[str, Any], seed: int | None = None
) -> list[val.Fold]:
    """Generate folds using the fold count / shuffle / seed from ``cfg`` (seed overridable)."""
    v = cfg["validation"]
    return val.generate_folds(
        df, strategy, v["n_splits"], v["shuffle"], cfg["project"]["random_seed"] if seed is None else seed
    )


def as_lists(folds: list[val.Fold]) -> list[tuple[list[int], list[int]]]:
    """Convert folds to plain lists for equality comparison."""
    return [(f.train_idx.tolist(), f.val_idx.tolist()) for f in folds]


# =============================================================================
# Fold structure (all strategies)
# =============================================================================
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_folds_partition_rows(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], strategy: const.SplitStrategy
) -> None:
    """Each fold covers every row once; across folds every row is validated exactly once."""
    folds = folds_for(grouped_df, strategy, cfg)

    assert len(folds) == cfg["validation"]["n_splits"]
    assert [f.index for f in folds] == list(range(len(folds)))
    for fold in folds:
        assert fold.strategy is strategy
        assert not set(fold.train_idx) & set(fold.val_idx)
        assert sorted([*fold.train_idx, *fold.val_idx]) == list(range(len(grouped_df)))
        assert fold.val_idx.size > 0
    all_val = np.concatenate([f.val_idx for f in folds])
    assert sorted(all_val.tolist()) == list(range(len(grouped_df)))


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_folds_are_positional_int_arrays(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], strategy: const.SplitStrategy
) -> None:
    """Indices are positional ``np.intp`` arrays, valid for ``.iloc`` even with a non-default index."""
    shifted = grouped_df.set_index(grouped_df.index + 1000)

    folds = folds_for(shifted, strategy, cfg)

    for fold in folds:
        assert fold.train_idx.dtype == np.intp
        assert fold.val_idx.max() < len(shifted)
        shifted.iloc[fold.val_idx]  # must not raise


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_same_seed_reproduces_folds(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], strategy: const.SplitStrategy
) -> None:
    """The configured seed (20261102) makes every strategy's folds exactly reproducible."""
    assert as_lists(folds_for(grouped_df, strategy, cfg)) == as_lists(folds_for(grouped_df, strategy, cfg))


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_different_seed_changes_shuffled_folds(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], seed: int, strategy: const.SplitStrategy
) -> None:
    """With shuffling on, the seed actually drives the assignment (it is not ignored)."""
    cfg["validation"]["shuffle"] = True

    assert as_lists(folds_for(grouped_df, strategy, cfg, seed)) != as_lists(
        folds_for(grouped_df, strategy, cfg, seed + 1)
    )


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_unshuffled_folds_ignore_seed(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], seed: int, strategy: const.SplitStrategy
) -> None:
    """Without shuffling the seed is not passed to scikit-learn and folds are seed-independent."""
    cfg["validation"]["shuffle"] = False

    assert as_lists(folds_for(grouped_df, strategy, cfg, seed)) == as_lists(
        folds_for(grouped_df, strategy, cfg, seed + 1)
    )


def test_random_folds_are_balanced(grouped_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Random K-fold validation sizes differ by at most one row."""
    sizes = [f.val_idx.size for f in folds_for(grouped_df, const.SplitStrategy.RANDOM, cfg)]

    assert max(sizes) - min(sizes) <= 1


# =============================================================================
# Leakage control
# =============================================================================
@pytest.mark.parametrize("strategy", GROUPED_STRATEGIES)
def test_grouped_strategies_never_share_groups(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], strategy: const.SplitStrategy
) -> None:
    """No group of the strategy's grouping column appears in both train and validation."""
    column = const.SPLIT_GROUP_COLUMN[strategy]
    assert column is not None

    for fold in folds_for(grouped_df, strategy, cfg):
        train_groups = set(grouped_df[column].iloc[fold.train_idx])
        val_groups = set(grouped_df[column].iloc[fold.val_idx])
        assert not train_groups & val_groups


@pytest.mark.parametrize("strategy", GROUPED_STRATEGIES)
def test_each_group_validated_in_exactly_one_fold(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], strategy: const.SplitStrategy
) -> None:
    """Every group is held out whole, in exactly one fold."""
    column = const.SPLIT_GROUP_COLUMN[strategy]
    assert column is not None
    seen: list[str] = []

    for fold in folds_for(grouped_df, strategy, cfg):
        seen.extend(grouped_df[column].iloc[fold.val_idx].unique())

    assert sorted(seen) == sorted(grouped_df[column].unique())


def test_family_split_also_isolates_proteins(grouped_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Family grouping is strictly stronger: it isolates proteins as well as families."""
    for fold in folds_for(grouped_df, const.SplitStrategy.FAMILY_GROUP, cfg):
        assert not val.shared_groups(grouped_df, fold, const.COL_PROTEIN_ID)


def test_random_split_leaks_groups(grouped_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """The random baseline does place sibling mutations on both sides; this is why it is optimistic."""
    folds = folds_for(grouped_df, const.SplitStrategy.RANDOM, cfg)

    assert any(val.shared_groups(grouped_df, f, const.COL_PROTEIN_ID) for f in folds)


# =============================================================================
# Feasibility checks
# =============================================================================
def test_too_few_groups_raises(grouped_df: pd.DataFrame) -> None:
    """Asking for more folds than families fails with guidance to lower ``n_splits``."""
    with pytest.raises(ValueError, match=r"n_splits=8 .*found 7.*Lower validation.n_splits"):
        val.generate_folds(grouped_df, const.SplitStrategy.FAMILY_GROUP, N_FAMILIES + 1, True, 0)


def test_n_splits_equal_to_groups_is_leave_one_group_out(grouped_df: pd.DataFrame) -> None:
    """With ``n_splits`` equal to the family count, each fold holds out exactly one family."""
    folds = val.generate_folds(grouped_df, const.SplitStrategy.FAMILY_GROUP, N_FAMILIES, True, 0)

    for fold in folds:
        assert len(set(grouped_df[const.COL_PROTEIN_FAMILY].iloc[fold.val_idx])) == 1


def test_too_few_rows_raises(grouped_df: pd.DataFrame) -> None:
    """Random K-fold needs at least ``n_splits`` rows."""
    with pytest.raises(ValueError, match="rows"):
        val.generate_folds(grouped_df.head(2), const.SplitStrategy.RANDOM, 3, True, 0)


@pytest.mark.parametrize("strategy", GROUPED_STRATEGIES)
def test_missing_group_column_raises(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], strategy: const.SplitStrategy
) -> None:
    """A grouped strategy without its grouping column fails, naming the column."""
    column = const.SPLIT_GROUP_COLUMN[strategy]
    assert column is not None

    with pytest.raises(KeyError, match=column):
        folds_for(grouped_df.drop(columns=column), strategy, cfg)


@pytest.mark.parametrize("strategy", GROUPED_STRATEGIES)
def test_missing_group_value_raises(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], strategy: const.SplitStrategy
) -> None:
    """Rows without a group label cannot be assigned safely and are rejected."""
    column = const.SPLIT_GROUP_COLUMN[strategy]
    assert column is not None
    grouped_df.loc[0, column] = None

    with pytest.raises(ValueError, match="no group"):
        folds_for(grouped_df, strategy, cfg)


def test_random_strategy_has_no_groups(grouped_df: pd.DataFrame) -> None:
    """The random strategy uses no grouping column."""
    assert val.group_labels(grouped_df, const.SplitStrategy.RANDOM) is None


# =============================================================================
# verify_folds catches hand-crafted violations
# =============================================================================
def make_fold(
    strategy: const.SplitStrategy, train: list[int], val_rows: list[int], index: int = 0
) -> val.Fold:
    """Build a ``Fold`` from plain lists."""
    return val.Fold(strategy, index, np.asarray(train, dtype=np.intp), np.asarray(val_rows, dtype=np.intp))


def test_verify_detects_row_overlap(grouped_df: pd.DataFrame) -> None:
    """A row in both train and validation is reported as leakage."""
    n = len(grouped_df)
    fold = make_fold(const.SplitStrategy.RANDOM, list(range(n)), [0])

    with pytest.raises(val.LeakageError, match="both train and validation"):
        val.verify_folds(grouped_df, [fold])


def test_verify_detects_incomplete_coverage(grouped_df: pd.DataFrame) -> None:
    """A fold that drops rows is reported."""
    fold = make_fold(const.SplitStrategy.RANDOM, [1, 2], [0])

    with pytest.raises(val.LeakageError, match="do not cover"):
        val.verify_folds(grouped_df, [fold])


def test_verify_detects_rows_never_validated(grouped_df: pd.DataFrame) -> None:
    """Folds whose validation sets do not together cover every row are reported."""
    n = len(grouped_df)
    fold = make_fold(const.SplitStrategy.RANDOM, list(range(1, n)), [0])

    with pytest.raises(val.LeakageError, match="not validated exactly once"):
        val.verify_folds(grouped_df, [fold])


@pytest.mark.parametrize("strategy", GROUPED_STRATEGIES)
def test_verify_detects_group_leakage(grouped_df: pd.DataFrame, strategy: const.SplitStrategy) -> None:
    """Splitting one group across train and validation is reported for grouped strategies."""
    n = len(grouped_df)
    fold = make_fold(strategy, list(range(1, n)), [0])  # row 0's group also sits in train

    with pytest.raises(val.LeakageError, match="in train and validation"):
        val.verify_folds(grouped_df, [fold])


# =============================================================================
# Summary statistics
# =============================================================================
def test_summary_schema_and_totals(grouped_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """The summary has the documented columns and its validation sizes add up to the table."""
    folds = folds_for(grouped_df, const.SplitStrategy.PROTEIN_GROUP, cfg)

    summary = val.summarize_folds(grouped_df, folds)

    expected_cols = {
        const.CV_COL_STRATEGY,
        const.CV_COL_FOLD,
        const.CV_COL_N_TRAIN,
        const.CV_COL_N_VAL,
        const.CV_COL_VAL_FRACTION,
        const.CV_COL_LEAKED_PROTEINS,
        const.CV_COL_LEAKED_FAMILIES,
        *(f"{const.CV_PREFIX_N_VAL_GROUPS}{level}" for level in const.GROUP_COLUMNS),
        *(f"{const.CV_PREFIX_VAL_MEAN}{t}" for t in const.TARGETS.values()),
    }
    assert set(summary.columns) == expected_cols
    assert len(summary) == len(folds)
    assert summary[const.CV_COL_N_VAL].sum() == len(grouped_df)
    assert (summary[const.CV_COL_N_TRAIN] + summary[const.CV_COL_N_VAL] == len(grouped_df)).all()
    assert summary[const.CV_COL_VAL_FRACTION].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("strategy", "proteins_leak", "families_leak"),
    [
        (const.SplitStrategy.RANDOM, True, True),
        (const.SplitStrategy.PROTEIN_GROUP, False, True),
        (const.SplitStrategy.FAMILY_GROUP, False, False),
    ],
)
def test_summary_leakage_counts_match_strategy(
    grouped_df: pd.DataFrame,
    cfg: dict[str, Any],
    strategy: const.SplitStrategy,
    proteins_leak: bool,
    families_leak: bool,
) -> None:
    """Leaked-group counts are non-zero exactly where the strategy permits sharing."""
    summary = val.summarize_folds(grouped_df, folds_for(grouped_df, strategy, cfg))

    assert bool(summary[const.CV_COL_LEAKED_PROTEINS].sum() > 0) is proteins_leak
    assert bool(summary[const.CV_COL_LEAKED_FAMILIES].sum() > 0) is families_leak


def test_summary_without_targets_omits_target_means(grouped_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Target-mean columns are only produced when targets are present (e.g. not for candidates)."""
    unlabeled = grouped_df.drop(columns=list(const.TARGETS.values()))
    folds = folds_for(unlabeled, const.SplitStrategy.RANDOM, cfg)

    summary = val.summarize_folds(unlabeled, folds)

    assert not [c for c in summary.columns if c.startswith(const.CV_PREFIX_VAL_MEAN)]


def test_log_fold_summary_logs_every_fold(
    grouped_df: pd.DataFrame, cfg: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """One log line per fold plus an imbalance line is emitted."""
    folds = folds_for(grouped_df, const.SplitStrategy.FAMILY_GROUP, cfg)

    with caplog.at_level(logging.INFO, logger=val.__name__):
        val.log_fold_summary(val.summarize_folds(grouped_df, folds))

    messages = [r.getMessage() for r in caplog.records]
    assert sum("fold" in m and "train=" in m for m in messages) == len(folds)
    assert any("imbalance ratio" in m for m in messages)


# =============================================================================
# Harness
# =============================================================================
def test_build_cv_splits_runs_configured_strategies(grouped_df: pd.DataFrame, cv_cfg: dict[str, Any]) -> None:
    """Every configured strategy is run, in config order, with one summary row per fold."""
    splits, summary = val.build_cv_splits(grouped_df, cv_cfg)

    assert list(splits) == ALL_STRATEGIES
    assert len(summary) == len(ALL_STRATEGIES) * cv_cfg["validation"]["n_splits"]
    assert summary[const.CV_COL_STRATEGY].unique().tolist() == [s.value for s in ALL_STRATEGIES]


def test_build_cv_splits_respects_strategy_subset(grouped_df: pd.DataFrame, cv_cfg: dict[str, Any]) -> None:
    """Only the strategies listed in config are run."""
    cv_cfg["validation"]["strategies"] = [const.SplitStrategy.FAMILY_GROUP.value]

    splits, _ = val.build_cv_splits(grouped_df, cv_cfg)

    assert list(splits) == [const.SplitStrategy.FAMILY_GROUP]


def test_iter_folds_yields_every_fold(grouped_df: pd.DataFrame, cv_cfg: dict[str, Any]) -> None:
    """``iter_folds`` flattens all strategies' folds into (strategy, index, train, val) tuples."""
    splits, _ = val.build_cv_splits(grouped_df, cv_cfg)

    items = list(val.iter_folds(splits))

    assert len(items) == sum(len(f) for f in splits.values())
    strategy, index, train_idx, val_idx = items[0]
    assert strategy is ALL_STRATEGIES[0]
    assert index == 0
    assert train_idx.size + val_idx.size == len(grouped_df)


def test_run_validation_harness_end_to_end(grouped_df: pd.DataFrame, cv_cfg: dict[str, Any]) -> None:
    """The harness reads the processed table, builds folds and writes a matching summary report."""
    processed = Path(cv_cfg["paths"]["processed_dir"]) / cv_cfg["files"]["processed_train"]
    write_table(grouped_df, processed)

    splits, summary = val.run_validation_harness(cv_cfg)

    report = Path(cv_cfg["paths"]["metrics_dir"]) / cv_cfg["files"]["cv_fold_summary"]
    assert report.is_file()
    pd.testing.assert_frame_equal(read_table(report), summary, check_dtype=False)
    assert set(splits) == set(ALL_STRATEGIES)


def test_run_validation_harness_without_preprocessing_raises(cv_cfg: dict[str, Any]) -> None:
    """Running Phase 2 before Phase 1 fails with a pointer to the missing upstream stage."""
    with pytest.raises(FileNotFoundError, match="previous pipeline stage"):
        val.run_validation_harness(cv_cfg)
