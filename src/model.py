"""Model architectures: feature blocks, preprocessing and regressor assembly.

Every model in this project is one scikit-learn ``Pipeline``::

    ColumnTransformer
      ├─ tabular   : impute → scale
      └─ embedding : impute → scale → PCA          (256-d → n_components)
    → regressor

Keeping imputation, scaling and PCA *inside* the pipeline is what makes the
leakage guarantee structural rather than procedural: ``fit`` on a training fold
is the only place their statistics are learned, and the validation fold is only
ever passed to ``predict``. ``src.train`` builds one such pipeline per fold and
``tests/test_train.py`` asserts the guarantee holds.

This module is deliberately free of I/O and of cross-validation logic: it
answers "what is the model?", not "how is it fit or scored?". Adding a new
architecture means adding a ``ModelName`` member, a ``CONFIG_BOUNDS`` entry and
one line in ``MODEL_REGISTRY`` — no other module changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.common import constants as const

FloatArray = npt.NDArray[np.float64]

# Registry: config model name -> estimator class. Add LightGBM / XGBoost here
# (plus a ModelName member and its CONFIG_BOUNDS) once they are dependencies.
MODEL_REGISTRY: dict[const.ModelName, type[BaseEstimator]] = {
    const.ModelName.RIDGE: Ridge,
    const.ModelName.RANDOM_FOREST: RandomForestRegressor,
    const.ModelName.GRADIENT_BOOSTING: GradientBoostingRegressor,
}

SCALER_REGISTRY: dict[const.Scaler, type[StandardScaler] | type[RobustScaler] | None] = {
    const.Scaler.STANDARD: StandardScaler,
    const.Scaler.ROBUST: RobustScaler,
    const.Scaler.NONE: None,
}


# =============================================================================
# Modelling data structures
# =============================================================================
@dataclass(frozen=True)
class FeatureSet:
    """Feature columns read from the Phase 1 feature manifest.

    Attributes:
        tabular: Physicochemical, structural, encoded and engineered columns.
        embedding: Protein embedding columns (reduced inside each fold).
    """

    tabular: list[str]
    embedding: list[str]

    @property
    def all(self) -> list[str]:
        """All model input columns, tabular first."""
        return [*self.tabular, *self.embedding]


@dataclass
class CVResults:
    """Everything produced by cross-validation.

    Attributes:
        oof: Long-format out-of-fold predictions (one row per strategy × model ×
            target × labeled mutation).
        fold_metrics: One row per strategy × model × target × fold.
        fold_pipelines: Fitted fold pipelines for the primary strategy, keyed
            by ``(target, model, fold)``; empty if not requested.
    """

    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_pipelines: dict[tuple[str, str, int], Pipeline] = field(default_factory=dict)


# =============================================================================
# Pipeline construction
# =============================================================================
def _block_pipeline(cfg: dict[str, Any], reduce: bool) -> Pipeline:
    """Impute → (scale) → (PCA) steps for one feature block."""
    seed = cfg["project"]["random_seed"]
    steps: list[tuple[str, Any]] = [
        (const.STEP_IMPUTE, SimpleImputer(strategy=cfg["quality"]["numeric_imputation"]))
    ]
    scaler_cls = SCALER_REGISTRY[const.Scaler(cfg["models"]["scaler"])]
    if scaler_cls is not None:
        steps.append((const.STEP_SCALE, scaler_cls()))
    if reduce:
        n_components = cfg["features"]["embedding_reduction"]["n_components"]
        steps.append((const.STEP_REDUCE, PCA(n_components=n_components, random_state=seed)))
    return Pipeline(steps)


def build_preprocessor(cfg: dict[str, Any], features: FeatureSet) -> ColumnTransformer:
    """Build the fold-local preprocessing for tabular and embedding blocks.

    Embeddings are standardised before PCA so that no single dimension
    dominates the components (when a scaler is configured). Blocks whose
    column list is empty are omitted, so single-block feature sets work.

    Raises:
        ValueError: If the feature set has no columns at all.
    """
    reduction = cfg["features"]["embedding_reduction"]
    reduce = bool(reduction["enabled"])
    if reduce:
        const.ReductionMethod(reduction["method"])  # only PCA today; fails loudly otherwise

    # A block with no columns is omitted entirely: an empty ColumnTransformer
    # block would leave PCA/scaler fitting on zero features. Ablation runs use
    # single-block feature sets, so this path is exercised there.
    blocks = [
        (const.BLOCK_TABULAR, _block_pipeline(cfg, reduce=False), features.tabular),
        (const.BLOCK_EMBEDDING, _block_pipeline(cfg, reduce=reduce), features.embedding),
    ]
    present = [(name, steps, columns) for name, steps, columns in blocks if columns]
    if not present:
        raise ValueError("FeatureSet is empty: at least one of tabular/embedding must have columns")
    return ColumnTransformer(present, remainder="drop", verbose_feature_names_out=True)


def build_estimator(name: const.ModelName, params: dict[str, Any], seed: int) -> BaseEstimator:
    """Instantiate a registered regressor with config hyperparameters and the project seed.

    The seed is applied to every estimator that exposes ``random_state``, so
    forests and boosting are reproducible.
    """
    estimator = MODEL_REGISTRY[name](**params)
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    return estimator


def build_pipeline(cfg: dict[str, Any], model: const.ModelName, features: FeatureSet) -> Pipeline:
    """Return an unfitted preprocessing + regressor pipeline for ``model``."""
    params = dict(cfg["models"]["candidates"][model.value])
    return Pipeline(
        [
            (const.STEP_PREPROCESS, build_preprocessor(cfg, features)),
            (const.STEP_MODEL, build_estimator(model, params, cfg["project"]["random_seed"])),
        ]
    )
