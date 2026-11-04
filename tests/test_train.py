"""Unit and integration tests for ``src.train`` (Phase 3).

The most important tests here are the leakage tests: they inspect the fitted
imputer / scaler / PCA inside each fold pipeline and prove their statistics
come from the training fold only, and that they do not change when the
validation rows are altered.

Training data is produced by running the real Phase 1 on the synthetic raw
tables from ``conftest.py``; models are shrunk so the suite stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import data_processing as pp
from src import evaluate as ev
from src import model as md
from src import train as tr
from src.common import constants as const
from src.common.config import validate_config
from src.common.io import read_table
from src.validation import Fold, build_cv_splits, generate_folds

# The synthetic data has a single labeled protein in one family, so a family-grouped
# training fold can hold one embedding vector; PCA then has zero variance and numpy
# warns while computing explained-variance ratios. Expected for this fixture only.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

TEST_N_SPLITS: int = 2  # synthetic data has only 2 families
TEST_N_COMPONENTS: int = 3  # must not exceed the smallest training fold
SMALL_MODELS: dict[str, dict[str, Any]] = {
    const.ModelName.RIDGE.value: {"alpha": 1.0},
    const.ModelName.RANDOM_FOREST.value: {
        "n_estimators": 5,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "max_features": 0.5,
        "n_jobs": 1,
    },
    const.ModelName.GRADIENT_BOOSTING.value: {
        "n_estimators": 5,
        "learning_rate": 0.1,
        "max_depth": 2,
        "subsample": 1.0,
    },
}


# =============================================================================
# Fixtures & helpers
# =============================================================================
@pytest.fixture()
def train_cfg(cfg: dict[str, Any], raw_files: dict[str, Path]) -> dict[str, Any]:
    """Config for fast training on synthetic data, with Phase 1 already run in ``tmp_path``."""
    cfg["validation"]["n_splits"] = TEST_N_SPLITS
    cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS
    cfg["models"]["candidates"] = {name: dict(params) for name, params in SMALL_MODELS.items()}
    validate_config(cfg)
    pp.run_preprocessing(cfg)
    return cfg


@pytest.fixture()
def data(train_cfg: dict[str, Any]) -> tuple[pd.DataFrame, md.FeatureSet]:
    """Processed training table and feature lists, as Phase 3 loads them."""
    return tr.load_training_data(train_cfg)


def protein_folds(df: pd.DataFrame, cfg: dict[str, Any]) -> list[Fold]:
    """Protein-grouped folds using the test fold count and project seed."""
    return generate_folds(
        df, const.SplitStrategy.PROTEIN_GROUP, TEST_N_SPLITS, True, cfg["project"]["random_seed"]
    )


def block(pipeline: Pipeline, name: str) -> Pipeline:
    """Return the fitted sub-pipeline for a feature block (tabular / embedding)."""
    block_pipeline: Pipeline = pipeline.named_steps[const.STEP_PREPROCESS].named_transformers_[name]
    return block_pipeline


# =============================================================================
# Data loading
# =============================================================================
def test_load_training_data_returns_manifest_features(data: tuple[pd.DataFrame, md.FeatureSet]) -> None:
    """Features come from the Phase 1 manifest; embeddings are the 256 constant columns."""
    df, features = data

    assert features.embedding == list(const.EMBEDDING_COLUMNS)
    assert set(const.NUMERIC_FEATURE_COLUMNS) <= set(features.tabular)
    assert features.all == [*features.tabular, *features.embedding]
    assert set(features.all) <= set(df.columns)


def test_load_training_data_requires_manifest(train_cfg: dict[str, Any]) -> None:
    """Missing manifest means Phase 1 did not complete; fail with a pointer to it."""
    Path(train_cfg["paths"]["processed_dir"], train_cfg["files"]["feature_manifest"]).unlink()

    with pytest.raises(FileNotFoundError, match="run Phase 1"):
        tr.load_training_data(train_cfg)


def test_load_training_data_detects_missing_columns(train_cfg: dict[str, Any]) -> None:
    """A manifest feature absent from the table is reported rather than silently dropped."""
    path = Path(train_cfg["paths"]["processed_dir"], train_cfg["files"]["processed_train"])
    read_table(path).drop(columns=const.COL_CONSERVATION).to_parquet(path, index=False)

    with pytest.raises(KeyError, match="lacks 1 columns"):
        tr.load_training_data(train_cfg)


# =============================================================================
# Fold-level diagnostics
# =============================================================================
def test_low_protein_count_warns(
    train_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet], caplog: pytest.LogCaptureFixture
) -> None:
    """PCA components beyond the number of distinct training proteins are flagged as uninformative."""
    df, features = data
    train_cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS

    with caplog.at_level("WARNING", logger=tr.__name__):
        tr.cross_validate_model(
            df,
            protein_folds(df, train_cfg),
            train_cfg,
            const.ModelName.RIDGE,
            const.TARGET_STABILITY,
            features,
        )

    assert any("distinct training proteins" in r.getMessage() for r in caplog.records)



# =============================================================================
# Leakage prevention (the critical guarantees)
# =============================================================================
def test_fold_transformers_fit_on_training_rows_only(
    train_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet]
) -> None:
    """Imputer, scaler and PCA statistics equal those of the training fold, not of all rows."""
    df, features = data
    folds = protein_folds(df, train_cfg)

    _, _, pipelines = tr.cross_validate_model(
        df, folds, train_cfg, const.ModelName.RIDGE, const.TARGET_STABILITY, features
    )

    for fold, pipeline in zip(folds, pipelines, strict=True):
        train_rows = df.iloc[fold.train_idx]
        tab = block(pipeline, const.BLOCK_TABULAR)
        emb = block(pipeline, const.BLOCK_EMBEDDING)

        np.testing.assert_allclose(
            tab.named_steps[const.STEP_SCALE].mean_, train_rows[features.tabular].mean()
        )
        np.testing.assert_allclose(
            tab.named_steps[const.STEP_IMPUTE].statistics_, train_rows[features.tabular].median()
        )
        np.testing.assert_allclose(
            emb.named_steps[const.STEP_SCALE].mean_, train_rows[features.embedding].mean()
        )
        assert emb.named_steps[const.STEP_REDUCE].n_samples_ == fold.train_idx.size
        # Sanity: the training-fold statistics differ from whole-table ones, so the test has teeth.
        assert not np.allclose(tab.named_steps[const.STEP_SCALE].mean_, df[features.tabular].mean())


def test_validation_rows_cannot_influence_fitted_pipeline(
    train_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet]
) -> None:
    """Corrupting validation features and targets leaves every fitted transformer and model unchanged."""
    df, features = data
    folds = protein_folds(df, train_cfg)
    fold = folds[0]
    corrupted = df.copy()
    corrupted[features.all] = corrupted[features.all].astype(float)
    rows = corrupted.index[fold.val_idx]
    corrupted.loc[rows, features.all] = corrupted.loc[rows, features.all] * 1000 + 7
    corrupted.loc[rows, const.TARGET_STABILITY] = -999.0

    _, _, clean = tr.cross_validate_model(
        df, [fold], train_cfg, const.ModelName.RIDGE, const.TARGET_STABILITY, features
    )
    _, _, dirty = tr.cross_validate_model(
        corrupted, [fold], train_cfg, const.ModelName.RIDGE, const.TARGET_STABILITY, features
    )

    clean_emb, dirty_emb = block(clean[0], const.BLOCK_EMBEDDING), block(dirty[0], const.BLOCK_EMBEDDING)
    np.testing.assert_array_equal(
        clean_emb.named_steps[const.STEP_REDUCE].components_,
        dirty_emb.named_steps[const.STEP_REDUCE].components_,
    )
    np.testing.assert_array_equal(
        block(clean[0], const.BLOCK_TABULAR).named_steps[const.STEP_SCALE].mean_,
        block(dirty[0], const.BLOCK_TABULAR).named_steps[const.STEP_SCALE].mean_,
    )
    np.testing.assert_array_equal(
        clean[0].named_steps[const.STEP_MODEL].coef_, dirty[0].named_steps[const.STEP_MODEL].coef_
    )


def test_each_fold_gets_a_fresh_pipeline(
    train_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet]
) -> None:
    """Folds never share a fitted object (no state carried from one fold to the next)."""
    df, features = data

    _, _, pipelines = tr.cross_validate_model(
        df, protein_folds(df, train_cfg), train_cfg, const.ModelName.RIDGE, const.TARGET_STABILITY, features
    )

    assert len({id(p) for p in pipelines}) == len(pipelines)
    assert len({id(p.named_steps[const.STEP_MODEL]) for p in pipelines}) == len(pipelines)


# =============================================================================
# Out-of-fold predictions
# =============================================================================
def test_oof_predictions_cover_every_row_once(
    train_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet]
) -> None:
    """Per model and target, each labeled mutation gets one OOF prediction with its true label."""
    df, features = data

    oof, records, _ = tr.cross_validate_model(
        df, protein_folds(df, train_cfg), train_cfg, const.ModelName.RIDGE, const.TARGET_FUNCTION, features
    )

    assert sorted(oof[const.COL_MUTATION_ID]) == sorted(df[const.COL_MUTATION_ID])
    truth = df.set_index(const.COL_MUTATION_ID)[const.TARGET_FUNCTION]
    np.testing.assert_allclose(oof[const.COL_Y_TRUE], truth.loc[oof[const.COL_MUTATION_ID]].to_numpy())
    assert np.isfinite(oof[const.COL_Y_PRED]).all()
    assert [r[const.CV_COL_FOLD] for r in records] == list(range(TEST_N_SPLITS))
    assert sum(r[const.COL_N_SAMPLES] for r in records) == len(df)


# =============================================================================
# Cross-validation grid and the train -> evaluate seam
# =============================================================================
@pytest.fixture()
def cv_run(
    train_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet]
) -> tuple[md.CVResults, pd.DataFrame]:
    """Cross-validation results and metric summary for all strategies, models and targets."""
    df, features = data
    splits, _ = build_cv_splits(df, train_cfg)
    results = tr.run_cross_validation(df, splits, train_cfg, features)
    return results, ev.summarize_metrics(results, train_cfg)


def test_cv_covers_full_grid(cv_run: tuple[md.CVResults, pd.DataFrame], train_cfg: dict[str, Any]) -> None:
    """Every strategy × model × target × fold is evaluated."""
    results, summary = cv_run
    n_strategies = len(train_cfg["validation"]["strategies"])
    n_models = len(train_cfg["models"]["candidates"])
    n_targets = len(const.TARGETS)

    assert len(results.fold_metrics) == n_strategies * n_models * n_targets * TEST_N_SPLITS
    assert len(summary) == n_strategies * n_models * n_targets


def test_summary_pooled_and_fold_metrics_are_consistent(
    cv_run: tuple[md.CVResults, pd.DataFrame],
    train_cfg: dict[str, Any],
    data: tuple[pd.DataFrame, md.FeatureSet],
) -> None:
    """Pooled metrics recompute from OOF rows; fold means recompute from fold metrics."""
    results, summary = cv_run
    keys = [const.CV_COL_STRATEGY, const.COL_MODEL, const.COL_TARGET]
    metrics = [const.Metric(m) for m in train_cfg["validation"]["metrics"]]

    for _, row in summary.iterrows():
        mask = (results.oof[keys] == row[keys].to_numpy()).all(axis=1)
        oof = results.oof[mask]
        assert row[const.COL_N_SAMPLES] == len(data[0])
        expected = ev.compute_metrics(
            oof[const.COL_Y_TRUE].to_numpy(), oof[const.COL_Y_PRED].to_numpy(), metrics
        )
        fold_rows = results.fold_metrics[(results.fold_metrics[keys] == row[keys].to_numpy()).all(axis=1)]
        for metric in metrics:
            assert row[f"{const.POOLED_PREFIX}{metric.value}"] == pytest.approx(
                expected[metric.value], nan_ok=True
            )
            assert row[f"{const.FOLD_MEAN_PREFIX}{metric.value}"] == pytest.approx(
                fold_rows[metric.value].mean(), nan_ok=True
            )


def test_fold_pipelines_kept_for_primary_strategy_only(
    cv_run: tuple[md.CVResults, pd.DataFrame], train_cfg: dict[str, Any]
) -> None:
    """Only primary-strategy fold pipelines are retained, one per model × target × fold."""
    results, _ = cv_run

    assert (
        len(results.fold_pipelines)
        == len(train_cfg["models"]["candidates"]) * len(const.TARGETS) * TEST_N_SPLITS
    )


# =============================================================================
# End-to-end
# =============================================================================
def test_run_training_writes_all_artifacts(
    train_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet]
) -> None:
    """Reports, final models, selected-model fold pipelines and metadata are written and usable."""
    df, features = data

    _, summary, selection = tr.run_training(train_cfg)

    reports = Path(train_cfg["paths"]["metrics_dir"])
    models = Path(train_cfg["paths"]["models_dir"])
    for key in ("oof_predictions", "cv_fold_metrics", "cv_metrics_summary", "model_selection"):
        assert (reports / train_cfg["files"][key]).is_file()
    pd.testing.assert_frame_equal(
        read_table(reports / train_cfg["files"]["cv_metrics_summary"]), summary, check_dtype=False
    )

    metadata = json.loads((models / train_cfg["files"]["training_metadata"]).read_text())
    assert metadata["random_seed"] == train_cfg["project"]["random_seed"]
    assert metadata["selection"] == json.loads((reports / train_cfg["files"]["model_selection"]).read_text())
    assert set(metadata["versions"]) >= {"python", "numpy", "pandas", "scikit-learn"}

    for target, filename in metadata["final_models"].items():
        assert selection[target][const.COL_MODEL] in filename
        pipeline = joblib.load(models / filename)
        predictions = pipeline.predict(df[features.all])
        assert predictions.shape == (len(df),)
        assert np.isfinite(predictions).all()

    assert len(metadata["fold_pipelines"]) == len(const.TARGETS) * TEST_N_SPLITS
    for relative in metadata["fold_pipelines"]:
        assert (models / relative).is_file()
        assert train_cfg["validation"]["primary_strategy"] in relative


def test_run_training_is_deterministic(train_cfg: dict[str, Any]) -> None:
    """Two runs with the same config and data give identical predictions and selections."""
    results_a, summary_a, selection_a = tr.run_training(train_cfg)
    results_b, summary_b, selection_b = tr.run_training(train_cfg)

    pd.testing.assert_frame_equal(results_a.oof, results_b.oof)
    pd.testing.assert_frame_equal(summary_a, summary_b)
    assert selection_a == selection_b


def test_run_training_respects_refit_and_fold_saving_flags(train_cfg: dict[str, Any]) -> None:
    """With refit and fold saving disabled, no model files are written (reports still are)."""
    train_cfg["models"]["refit_on_full_data"] = False
    train_cfg["models"]["save_fold_pipelines"] = False

    tr.run_training(train_cfg)

    models = Path(train_cfg["paths"]["models_dir"])
    assert not list(models.rglob("*.joblib"))
    metadata = json.loads((models / train_cfg["files"]["training_metadata"]).read_text())
    assert metadata["final_models"] == {}
    assert metadata["fold_pipelines"] == []
