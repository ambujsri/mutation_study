"""Unit tests for ``src.model``: pipeline architecture and the model registry.

These tests assert the *shape* of what is built — which blocks exist, which
steps they contain, which columns reach them and which hyperparameters and seed
are applied. Whether that structure actually prevents leakage when it is fit
fold by fold is asserted in ``tests/test_train.py``; here we only guarantee that
the structure is the one the config asked for.

Column names and step names always come from ``src.common.constants``, never
string literals, so a rename in the pipeline cannot pass silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src import data_processing as pp
from src import model as md
from src import train as tr
from src.common import constants as const
from src.common.config import validate_config

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
def model_cfg(cfg: dict[str, Any], raw_files: dict[str, Path]) -> dict[str, Any]:
    """Config for cheap pipelines on synthetic data, with Phase 1 already run."""
    cfg["validation"]["n_splits"] = TEST_N_SPLITS
    cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS
    cfg["models"]["candidates"] = {name: dict(params) for name, params in SMALL_MODELS.items()}
    validate_config(cfg)
    pp.run_preprocessing(cfg)
    return cfg


@pytest.fixture()
def data(model_cfg: dict[str, Any]) -> tuple[pd.DataFrame, md.FeatureSet]:
    """Processed training table and feature lists, as Phase 3 loads them."""
    return tr.load_training_data(model_cfg)


def block(pipeline: Pipeline, name: str) -> Pipeline:
    """Return the fitted sub-pipeline for a feature block (tabular / embedding)."""
    block_pipeline: Pipeline = pipeline.named_steps[const.STEP_PREPROCESS].named_transformers_[name]
    return block_pipeline


# =============================================================================
# Feature set
# =============================================================================
def test_feature_set_orders_tabular_before_embedding() -> None:
    """``FeatureSet.all`` is the model's column order and is tabular-first."""
    features = md.FeatureSet(tabular=["a", "b"], embedding=["e1", "e2"])

    assert features.all == ["a", "b", "e1", "e2"]


def test_feature_set_is_immutable() -> None:
    """The feature set is frozen so a stage cannot silently redefine the model inputs."""
    features = md.FeatureSet(tabular=["a"], embedding=["e1"])

    with pytest.raises(AttributeError):
        features.tabular = ["b"]  # type: ignore[misc]


# =============================================================================
# Pipeline construction
# =============================================================================
def test_pipeline_structure(model_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet]) -> None:
    """Pipeline = preprocess(tabular: impute→scale, embedding: impute→scale→PCA) → model."""
    _, features = data

    pipeline = md.build_pipeline(model_cfg, const.ModelName.RIDGE, features)

    assert [name for name, _ in pipeline.steps] == [const.STEP_PREPROCESS, const.STEP_MODEL]
    transformers = {
        name: (steps, cols) for name, steps, cols in pipeline.named_steps[const.STEP_PREPROCESS].transformers
    }
    tab_steps, tab_cols = transformers[const.BLOCK_TABULAR]
    emb_steps, emb_cols = transformers[const.BLOCK_EMBEDDING]
    assert tab_cols == features.tabular
    assert emb_cols == features.embedding
    assert list(tab_steps.named_steps) == [const.STEP_IMPUTE, const.STEP_SCALE]
    assert list(emb_steps.named_steps) == [const.STEP_IMPUTE, const.STEP_SCALE, const.STEP_REDUCE]
    pca = emb_steps.named_steps[const.STEP_REDUCE]
    assert isinstance(pca, PCA)
    assert pca.n_components == TEST_N_COMPONENTS
    assert pca.random_state == model_cfg["project"]["random_seed"]


@pytest.mark.parametrize(
    ("scaler", "expected"),
    [(const.Scaler.STANDARD, StandardScaler), (const.Scaler.ROBUST, RobustScaler), (const.Scaler.NONE, None)],
)
def test_scaler_follows_config(
    model_cfg: dict[str, Any],
    data: tuple[pd.DataFrame, md.FeatureSet],
    scaler: const.Scaler,
    expected: type | None,
) -> None:
    """The configured scaler (or none) is used in both feature blocks."""
    model_cfg["models"]["scaler"] = scaler.value

    preprocessor = md.build_preprocessor(model_cfg, data[1])

    for _, steps, _ in preprocessor.transformers:
        if expected is None:
            assert const.STEP_SCALE not in steps.named_steps
        else:
            assert isinstance(steps.named_steps[const.STEP_SCALE], expected)


def test_embedding_reduction_can_be_disabled(
    model_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet]
) -> None:
    """With reduction disabled the full embedding block is passed to the model."""
    df, features = data
    model_cfg["features"]["embedding_reduction"]["enabled"] = False

    pipeline = md.build_pipeline(model_cfg, const.ModelName.RIDGE, features)
    pipeline.fit(df[features.all], df[const.TARGET_STABILITY])

    assert const.STEP_REDUCE not in block(pipeline, const.BLOCK_EMBEDDING).named_steps
    assert pipeline.named_steps[const.STEP_MODEL].n_features_in_ == len(features.all)


def test_reduced_feature_count(model_cfg: dict[str, Any], data: tuple[pd.DataFrame, md.FeatureSet]) -> None:
    """With PCA on, the model sees tabular features plus ``n_components`` embedding components."""
    df, features = data

    pipeline = md.build_pipeline(model_cfg, const.ModelName.RIDGE, features)
    pipeline.fit(df[features.all], df[const.TARGET_STABILITY])

    assert pipeline.named_steps[const.STEP_MODEL].n_features_in_ == len(features.tabular) + TEST_N_COMPONENTS


# =============================================================================
# Single-block feature sets (used by the ablation study)
# =============================================================================
def test_empty_block_is_omitted_from_the_preprocessor(model_cfg: dict[str, Any]) -> None:
    """A block with no columns is dropped, not left to fit a scaler on zero features."""
    preprocessor = md.build_preprocessor(model_cfg, md.FeatureSet(tabular=[], embedding=["e1", "e2"]))

    assert isinstance(preprocessor, ColumnTransformer)
    assert [name for name, _, _ in preprocessor.transformers] == [const.BLOCK_EMBEDDING]


def test_empty_feature_set_is_rejected(model_cfg: dict[str, Any]) -> None:
    """A feature set with no columns at all cannot produce a model and fails loudly."""
    with pytest.raises(ValueError, match="FeatureSet is empty"):
        md.build_preprocessor(model_cfg, md.FeatureSet(tabular=[], embedding=[]))


# =============================================================================
# Estimators and the registry
# =============================================================================
@pytest.mark.parametrize(
    ("model", "cls"),
    [
        (const.ModelName.RIDGE, Ridge),
        (const.ModelName.RANDOM_FOREST, RandomForestRegressor),
        (const.ModelName.GRADIENT_BOOSTING, GradientBoostingRegressor),
    ],
)
def test_build_estimator_applies_params_and_seed(
    model: const.ModelName, cls: type[BaseEstimator], seed: int
) -> None:
    """Config hyperparameters are passed through and the project seed is set where supported."""
    estimator = md.build_estimator(model, SMALL_MODELS[model.value], seed)

    assert isinstance(estimator, cls)
    params = estimator.get_params()
    for key, value in SMALL_MODELS[model.value].items():
        assert params[key] == value
    assert params["random_state"] == seed


def test_build_estimator_is_a_fresh_instance_each_call(seed: int) -> None:
    """Two calls never share an estimator object, so folds cannot contaminate each other."""
    first = md.build_estimator(const.ModelName.RIDGE, SMALL_MODELS[const.ModelName.RIDGE.value], seed)
    second = md.build_estimator(const.ModelName.RIDGE, SMALL_MODELS[const.ModelName.RIDGE.value], seed)

    assert first is not second


def test_every_registered_model_has_config_choice() -> None:
    """The registry and the ``ModelName`` enum list exactly the same models."""
    assert set(md.MODEL_REGISTRY) == set(const.ModelName)


def test_every_scaler_choice_is_registered() -> None:
    """Every ``Scaler`` the config accepts maps to a class (or an explicit ``None``)."""
    assert set(md.SCALER_REGISTRY) == set(const.Scaler)
