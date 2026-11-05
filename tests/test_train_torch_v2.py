"""Unit tests for ``src.train_torch_v2``: the dual-branch training wrapper.

The architecture is tested in ``tests/test_torch_model_v2.py``; what matters
here is everything around it, and above all the two guarantees a neural
network makes easy to break:

* **the leakage boundaries** — both preprocessors and the target scaler are
  fitted on inner-training rows only, and the outer validation fold is touched
  exactly once, at predict time; and
* **reproducibility** — the same config and the same data give the same
  out-of-fold predictions, which is what seed 20261102 is for.

Runs are kept to a handful of epochs on the synthetic data from
``conftest.py``: these assert that the wiring is right, not that a 24-row fit
is any good.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import data_processing as pp
from src import train as tr
from src import train_torch_v2 as v2
from src.common import constants as const
from src.common.config import validate_config
from src.common.io import read_table
from src.model import FeatureSet
from src.torch_model_v2 import dual_branch_config_from
from src.validation import generate_folds

# The synthetic data has a single labeled protein in one family, so a family-grouped
# training fold can hold one embedding vector; PCA then has zero variance and numpy
# warns while computing explained-variance ratios. Expected for this fixture only.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

TEST_N_SPLITS: int = 2  # synthetic data has only 2 families
TEST_N_COMPONENTS: int = 2
TEST_MAX_EPOCHS: int = 3
TEST_PATIENCE: int = 2
TEST_BOTTLENECK: int = 4


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture()
def v2_cfg(cfg: dict[str, Any], raw_files: dict[str, Path]) -> dict[str, Any]:
    """Config with Phase 1 run and a network small enough to fit in a test."""
    cfg["validation"]["n_splits"] = TEST_N_SPLITS
    cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS
    cfg["models"]["candidates"] = {const.ModelName.RIDGE.value: {"alpha": 1.0}}
    cfg["torch_v2"]["tabular_branch"]["hidden_dims"] = [8]
    cfg["torch_v2"]["embedding_branch"]["hidden_dims"] = [8]
    cfg["torch_v2"]["embedding_branch"]["bottleneck_dim"] = TEST_BOTTLENECK
    cfg["torch_v2"]["fusion"]["hidden_dim"] = 8
    cfg["torch_v2"]["fusion"]["n_residual_blocks"] = 1
    cfg["torch_v2"]["head_hidden_dim"] = None
    cfg["torch_v2"]["batch_size"] = 8
    cfg["torch_v2"]["max_epochs"] = TEST_MAX_EPOCHS
    cfg["torch_v2"]["early_stopping"]["patience"] = TEST_PATIENCE
    cfg["evaluation"]["figures"] = False
    validate_config(cfg)
    pp.run_preprocessing(cfg)
    return cfg


@pytest.fixture()
def fold_inputs(v2_cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[Any], FeatureSet, Any]:
    """The data, folds, feature set and resolved config one fold needs."""
    df, features = v2.load_training_data(v2_cfg)
    config = dual_branch_config_from(v2_cfg)
    folds = generate_folds(
        df,
        const.SplitStrategy(v2_cfg["validation"]["primary_strategy"]),
        v2_cfg["validation"]["n_splits"],
        v2_cfg["validation"]["shuffle"],
        v2_cfg["project"]["random_seed"],
    )
    return df, folds, features, config


# =============================================================================
# Branch preprocessing
# =============================================================================
def test_each_branch_gets_its_own_preprocessor(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """The two blocks are transformed separately, which is what the branches need."""
    df, _, features, config = fold_inputs

    tabular_pre, embedding_pre = v2.build_branch_preprocessors(v2_cfg, features, config)
    x_tab = tabular_pre.fit_transform(df[features.tabular])
    x_emb = embedding_pre.fit_transform(df[features.embedding])

    assert x_tab.shape[1] == len(features.tabular)
    assert x_tab.shape[0] == x_emb.shape[0] == len(df)


def test_embedding_branch_skips_pca_by_default(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """With ``embedding_pca: false`` the full embedding reaches the learned bottleneck.

    PCA keeps the directions of greatest variance, which on a protein-level
    vector is largely variance between families — the component that cannot
    transfer to a held-out family. The bottleneck is learned instead.
    """
    df, _, features, config = fold_inputs
    assert config.embedding_pca is False

    _, embedding_pre = v2.build_branch_preprocessors(v2_cfg, features, config)
    x_emb = embedding_pre.fit_transform(df[features.embedding])

    assert x_emb.shape[1] == len(features.embedding)
    assert x_emb.shape[1] != TEST_N_COMPONENTS


def test_embedding_pca_can_be_turned_back_on(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """The flag is real: true restores the fold-local PCA ahead of the bottleneck."""
    df, _, features, _ = fold_inputs
    v2_cfg["torch_v2"]["embedding_pca"] = True
    config = dual_branch_config_from(v2_cfg)

    _, embedding_pre = v2.build_branch_preprocessors(v2_cfg, features, config)
    x_emb = embedding_pre.fit_transform(df[features.embedding])

    assert x_emb.shape[1] == TEST_N_COMPONENTS


def test_building_preprocessors_does_not_mutate_the_config(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """Disabling PCA for the embedding branch must not disable it project-wide."""
    _, _, features, config = fold_inputs
    before = v2_cfg["features"]["embedding_reduction"]["enabled"]

    v2.build_branch_preprocessors(v2_cfg, features, config)

    assert v2_cfg["features"]["embedding_reduction"]["enabled"] == before


def test_both_feature_blocks_are_required(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """A single-block feature set cannot drive a two-branch model."""
    _, _, features, config = fold_inputs

    with pytest.raises(ValueError, match="needs both feature blocks"):
        v2.build_branch_preprocessors(
            v2_cfg, FeatureSet(tabular=features.tabular, embedding=[]), config
        )


# =============================================================================
# Leakage boundaries
# =============================================================================
def test_preprocessors_never_see_the_outer_validation_fold(
    v2_cfg: dict[str, Any],
    fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rows each preprocessor is fitted on are a subset of the training fold.

    Recorded by intercepting ``fit_transform`` rather than inferred from the
    numbers, so the assertion is about what the code did, not about what the
    result looks like.
    """
    df, folds, features, config = fold_inputs
    fold = folds[0]
    seen: list[pd.Index] = []

    original = v2.build_branch_preprocessors

    def recording(*args: Any, **kwargs: Any) -> Any:
        tabular_pre, embedding_pre = original(*args, **kwargs)
        for transformer in (tabular_pre, embedding_pre):
            inner_fit = transformer.fit_transform

            def wrapped(frame: pd.DataFrame, *rest: Any, _fit: Any = inner_fit) -> Any:
                seen.append(frame.index)
                return _fit(frame, *rest)

            transformer.fit_transform = wrapped  # type: ignore[method-assign]
        return tabular_pre, embedding_pre

    monkeypatch.setattr(v2, "build_branch_preprocessors", recording)
    v2.train_one_fold(df, fold, features, v2_cfg, config)

    train_labels = set(df.index[fold.train_idx])
    val_labels = set(df.index[fold.val_idx])
    assert len(seen) == 2, "both branches must be fitted exactly once"
    for index in seen:
        assert set(index) <= train_labels
        assert not set(index) & val_labels


def test_the_inner_split_is_carved_out_of_the_training_fold(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """Early stopping watches held-out rows that are still training-fold rows.

    Stopping on the outer validation fold would pick the epoch that scores
    best on the very rows used to report performance.
    """
    df, folds, features, config = fold_inputs
    fold = folds[0]

    _, _, summary = v2.train_one_fold(df, fold, features, v2_cfg, config)

    n_inner = summary[const.TORCH_COL_N_INNER_TRAIN] + summary[const.TORCH_COL_N_INNER_VAL]
    assert n_inner == fold.train_idx.size
    assert summary[const.TORCH_COL_N_INNER_VAL] > 0


def test_the_target_scaler_is_fitted_on_inner_training_rows_only(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """The stability mean/std come from a strict subset of the training fold."""
    df, folds, features, config = fold_inputs
    fold = folds[0]

    (_, _, _, scaler), _, summary = v2.train_one_fold(df, fold, features, v2_cfg, config)

    values = df[const.TARGET_STABILITY].to_numpy(dtype=float)
    assert scaler.enabled
    assert scaler.mean != pytest.approx(float(values.mean()))
    assert summary[const.TORCH_COL_N_INNER_TRAIN] < fold.train_idx.size


def test_predictions_are_returned_in_target_units(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """Stability is inverse-transformed to kcal/mol; function stays inside 0-1."""
    df, folds, features, config = fold_inputs
    fold = folds[0]
    artifacts, _, _ = v2.train_one_fold(df, fold, features, v2_cfg, config)

    stability, function = v2.predict_rows(artifacts, df.iloc[fold.val_idx], features, config)

    assert stability.shape == (fold.val_idx.size,)
    assert np.all(function >= 0) and np.all(function <= 1)
    assert np.isfinite(stability).all()


# =============================================================================
# Early stopping
# =============================================================================
def test_history_records_one_row_per_epoch(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """The learning-curve artifact is one row per epoch with both losses."""
    df, folds, features, config = fold_inputs

    _, history, summary = v2.train_one_fold(df, folds[0], features, v2_cfg, config)

    assert list(history[const.TORCH_COL_EPOCH]) == list(range(1, len(history) + 1))
    assert len(history) <= TEST_MAX_EPOCHS
    assert summary[const.TORCH_COL_STOPPED_EPOCH] == len(history)
    assert history[const.TORCH_COL_TRAIN_LOSS].notna().all()
    assert history[const.TORCH_COL_VAL_LOSS].notna().all()


def test_the_kept_epoch_is_the_best_one_not_the_last(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """Weights are restored from the best inner-validation loss, within ``min_delta``.

    Not necessarily the exact argmin: an epoch only counts as an improvement
    if it beats the incumbent by more than ``min_delta``, so a later epoch
    that is lower by a hair deliberately does not displace it.
    """
    df, folds, features, config = fold_inputs

    _, history, summary = v2.train_one_fold(df, folds[0], features, v2_cfg, config)

    losses = history.set_index(const.TORCH_COL_EPOCH)[const.TORCH_COL_VAL_LOSS]
    kept = float(losses.loc[summary[const.TORCH_COL_BEST_EPOCH]])
    assert summary[const.TORCH_COL_VAL_LOSS] == pytest.approx(kept)
    assert kept <= float(losses.min()) + config.min_delta
    # Nothing before the kept epoch was better, or it would have been kept.
    earlier = losses.loc[: summary[const.TORCH_COL_BEST_EPOCH]].iloc[:-1]
    assert earlier.empty or kept < float(earlier.min())


# =============================================================================
# Reproducibility
# =============================================================================
def test_two_runs_give_the_same_out_of_fold_predictions(
    v2_cfg: dict[str, Any], fold_inputs: tuple[pd.DataFrame, list[Any], FeatureSet, Any]
) -> None:
    """Seed 20261102 makes the whole fold deterministic, weights included."""
    df, folds, features, config = fold_inputs

    first, *_ = v2.cross_validate_dual_branch(df, folds, features, v2_cfg, config)
    second, *_ = v2.cross_validate_dual_branch(df, folds, features, v2_cfg, config)

    assert np.allclose(
        first[const.COL_Y_PRED].to_numpy(dtype=float),
        second[const.COL_Y_PRED].to_numpy(dtype=float),
    )


# =============================================================================
# Artifacts and orchestration
# =============================================================================
def test_run_writes_every_configured_artifact(v2_cfg: dict[str, Any]) -> None:
    """The stage emits metrics, history, weights and a provenance record."""
    tr.run_training(v2_cfg)

    summary, comparison = v2.run_dual_branch_training(v2_cfg)

    metrics_dir = Path(v2_cfg["paths"]["metrics_dir"])
    for key in (
        "torch_v2_oof_predictions",
        "torch_v2_cv_metrics",
        "torch_v2_fold_summary",
        "torch_v2_history",
        "torch_v2_comparison",
        "torch_v2_comparison_markdown",
    ):
        path = metrics_dir / v2_cfg["files"][key]
        assert path.is_file() and path.stat().st_size > 0, key

    assert set(summary[const.COL_TARGET]) == set(const.TARGETS.values())
    assert set(summary[const.COL_MODEL]) == {const.TORCH_V2_MODEL_NAME}
    assert const.TORCH_V2_MODEL_NAME in set(comparison[const.COL_MODEL])


def test_metadata_records_capacity_per_branch(v2_cfg: dict[str, Any]) -> None:
    """Provenance names the seed, the versions and how capacity is split."""
    tr.run_training(v2_cfg)
    v2.run_dual_branch_training(v2_cfg)

    path = (
        Path(v2_cfg["paths"]["models_dir"])
        / const.TORCH_V2_MODELS_SUBDIR
        / const.TORCH_V2_METADATA_FILE
    )
    metadata = json.loads(path.read_text(encoding=const.CONFIG_ENCODING))

    assert metadata["model"] == const.TORCH_V2_MODEL_NAME
    assert metadata["random_seed"] == v2_cfg["project"]["random_seed"]
    assert metadata["architecture"]["bottleneck_dim"] == TEST_BOTTLENECK
    assert metadata["architecture"]["embedding_pca"] is False
    assert set(metadata["parameters_by_branch"]) == {
        const.TORCH_V2_BRANCH_TABULAR,
        const.TORCH_V2_BRANCH_EMBEDDING,
        const.TORCH_V2_BRANCH_FUSION,
        const.TORCH_V2_BRANCH_HEADS,
    }
    assert metadata["versions"]["torch"]


def test_oof_schema_matches_the_other_models(v2_cfg: dict[str, Any]) -> None:
    """The long-format columns line up, so the tables concatenate without reconciliation."""
    tr.run_training(v2_cfg)
    v2.run_dual_branch_training(v2_cfg)

    metrics_dir = Path(v2_cfg["paths"]["metrics_dir"])
    baseline = read_table(metrics_dir / v2_cfg["files"]["oof_predictions"])
    network = read_table(metrics_dir / v2_cfg["files"]["torch_v2_oof_predictions"])

    assert set(network.columns) == set(baseline.columns)
    assert set(network[const.COL_MODEL]) == {const.TORCH_V2_MODEL_NAME}


def test_comparison_is_skipped_without_baselines(v2_cfg: dict[str, Any]) -> None:
    """Run before the baselines, the stage still trains and says what is missing."""
    summary, comparison = v2.run_dual_branch_training(v2_cfg)

    assert not summary.empty
    assert comparison.empty
    assert not (
        Path(v2_cfg["paths"]["metrics_dir"]) / v2_cfg["files"]["torch_v2_comparison"]
    ).exists()
