"""Model training, dual-target cross-validation and model selection.

Phase 3 of the Medvolt.AI mutation-effect pipeline.

For every CV strategy (``validation.strategies``), every candidate model
(``models.candidates``) and both targets (``stability_ddG``,
``functional_score``) the module fits one scikit-learn ``Pipeline`` per fold:

    ColumnTransformer
      ├─ tabular   : impute → scale
      └─ embedding : impute → scale → PCA          (256-d → n_components)
    → regressor

Because imputation, scaling and PCA live *inside* the pipeline, ``fit`` on the
training fold is the only place their statistics are learned; the validation
fold is only ever passed to ``predict``. This is the leakage guarantee, and it
is asserted in ``tests/test_train.py``.

Metrics are reported two ways:
    * per fold, then mean ± std across folds; and
    * **pooled** over all out-of-fold (OOF) predictions at once. Family-grouped
      folds are small and uneven (1–2 families each), so pooled OOF metrics are
      the primary, lower-variance estimate.

The best model per target is chosen by the pooled ``models.selection_metric``
under ``validation.primary_strategy``, then optionally refit on all labeled
data for Phase 4.

Run directly::

    python -m src.train --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table, write_table
from src.validation import Fold, build_cv_splits

logger = logging.getLogger(__name__)

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
# Data
# =============================================================================
def load_training_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, FeatureSet]:
    """Load the processed training table and its feature lists from Phase 1.

    Raises:
        FileNotFoundError: If Phase 1 has not been run.
        KeyError: If manifest features or targets are missing from the table.
    """
    df = read_table(resolve_path(cfg, "processed_dir", "processed_train"))
    manifest_path = resolve_path(cfg, "processed_dir", "feature_manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Feature manifest not found: {manifest_path} (run Phase 1 first)")
    manifest = json.loads(manifest_path.read_text(encoding=const.CONFIG_ENCODING))
    features = FeatureSet(manifest["tabular_features"], manifest["embedding_features"])

    missing = [c for c in [*features.all, *const.TARGETS.values()] if c not in df.columns]
    if missing:
        raise KeyError(f"Processed training table lacks {len(missing)} columns, e.g. {missing[:5]}")
    logger.info(
        "Loaded training data shape=%s: %d tabular + %d embedding features",
        df.shape,
        len(features.tabular),
        len(features.embedding),
    )
    return df, features


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


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(y_true: FloatArray, y_pred: FloatArray, metrics: list[const.Metric]) -> dict[str, float]:
    """Compute the requested regression metrics.

    Spearman is ``NaN`` when either vector is constant (rank correlation is
    undefined); it is never silently replaced by 0.
    """
    calculators = {
        const.Metric.RMSE: lambda: float(np.sqrt(mean_squared_error(y_true, y_pred))),
        const.Metric.MAE: lambda: float(mean_absolute_error(y_true, y_pred)),
        const.Metric.R2: lambda: float(r2_score(y_true, y_pred)),
        const.Metric.SPEARMAN: lambda: (
            float("nan")
            if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0
            else float(spearmanr(y_true, y_pred).statistic)
        ),
    }
    return {metric.value: calculators[metric]() for metric in metrics}


# =============================================================================
# Cross-validation
# =============================================================================
def cross_validate_model(
    df: pd.DataFrame,
    folds: list[Fold],
    cfg: dict[str, Any],
    model: const.ModelName,
    target: str,
    features: FeatureSet,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[Pipeline]]:
    """Fit one fresh pipeline per fold and predict its validation rows.

    Returns:
        ``(oof, fold_metrics, pipelines)``: OOF predictions for every row,
        metric records per fold, and the fitted pipelines in fold order.
    """
    metrics = [const.Metric(m) for m in cfg["validation"]["metrics"]]
    x_all, y = df[features.all], df[target].to_numpy(dtype=float)
    oof_frames, records, pipelines = [], [], []
    reduction = cfg["features"]["embedding_reduction"]
    for fold in folds:
        n_proteins = df[const.COL_PROTEIN_ID].iloc[fold.train_idx].nunique()
        if reduction["enabled"] and reduction["n_components"] >= n_proteins:
            # Embeddings are per protein, so the embedding block has rank <= n_proteins.
            logger.warning(
                "[%s] fold %d: n_components=%d >= %d distinct training proteins; "
                "extra PCA components carry no information",
                fold.strategy.value,
                fold.index,
                reduction["n_components"],
                n_proteins,
            )
        pipeline = build_pipeline(cfg, model, features)
        pipeline.fit(x_all.iloc[fold.train_idx], y[fold.train_idx])
        y_pred = np.asarray(pipeline.predict(x_all.iloc[fold.val_idx]), dtype=float)
        y_true = y[fold.val_idx]

        scores = compute_metrics(y_true, y_pred, metrics)
        records.append(
            {
                const.CV_COL_STRATEGY: fold.strategy.value,
                const.COL_MODEL: model.value,
                const.COL_TARGET: target,
                const.CV_COL_FOLD: fold.index,
                const.COL_N_SAMPLES: fold.val_idx.size,
                **scores,
            }
        )
        logger.debug("[%s|%s|%s] fold %d %s", fold.strategy.value, model.value, target, fold.index, scores)

        val_rows = df.iloc[fold.val_idx]
        oof_frames.append(
            pd.DataFrame(
                {
                    const.COL_MUTATION_ID: val_rows[const.COL_MUTATION_ID].to_numpy(),
                    const.COL_PROTEIN_ID: val_rows[const.COL_PROTEIN_ID].to_numpy(),
                    const.COL_PROTEIN_FAMILY: val_rows[const.COL_PROTEIN_FAMILY].to_numpy(),
                    const.CV_COL_STRATEGY: fold.strategy.value,
                    const.COL_MODEL: model.value,
                    const.COL_TARGET: target,
                    const.CV_COL_FOLD: fold.index,
                    const.COL_Y_TRUE: y_true,
                    const.COL_Y_PRED: y_pred,
                }
            )
        )
        pipelines.append(pipeline)
    return pd.concat(oof_frames, ignore_index=True), records, pipelines


def run_cross_validation(
    df: pd.DataFrame,
    splits: dict[const.SplitStrategy, list[Fold]],
    cfg: dict[str, Any],
    features: FeatureSet,
) -> CVResults:
    """Cross-validate every configured model on both targets under every strategy.

    Fold pipelines are kept only for the primary strategy (and only if
    ``models.save_fold_pipelines``), since only they are used downstream.
    """
    primary = const.SplitStrategy(cfg["validation"]["primary_strategy"])
    keep_folds = bool(cfg["models"]["save_fold_pipelines"])
    oof_frames, records = [], []
    fold_pipelines: dict[tuple[str, str, int], Pipeline] = {}
    for strategy, folds in splits.items():
        for model_name in cfg["models"]["candidates"]:
            model = const.ModelName(model_name)
            for target in const.TARGETS.values():
                oof, fold_records, pipelines = cross_validate_model(df, folds, cfg, model, target, features)
                oof_frames.append(oof)
                records.extend(fold_records)
                pooled = compute_metrics(
                    oof[const.COL_Y_TRUE].to_numpy(),
                    oof[const.COL_Y_PRED].to_numpy(),
                    [const.Metric(m) for m in cfg["validation"]["metrics"]],
                )
                logger.info(
                    "[%s|%s|%s] pooled OOF: %s",
                    strategy.value,
                    model.value,
                    target,
                    ", ".join(f"{k}={v:.3f}" for k, v in pooled.items()),
                )
                if keep_folds and strategy is primary:
                    for fold, pipeline in zip(folds, pipelines, strict=True):
                        fold_pipelines[(target, model.value, fold.index)] = pipeline
    return CVResults(pd.concat(oof_frames, ignore_index=True), pd.DataFrame(records), fold_pipelines)


def summarize_metrics(results: CVResults, cfg: dict[str, Any]) -> pd.DataFrame:
    """Combine pooled-OOF metrics with per-fold mean and std per strategy × model × target."""
    metrics = [const.Metric(m) for m in cfg["validation"]["metrics"]]
    keys = [const.CV_COL_STRATEGY, const.COL_MODEL, const.COL_TARGET]
    rows = []
    for key_values, group in results.oof.groupby(keys, sort=False):
        pooled = compute_metrics(
            group[const.COL_Y_TRUE].to_numpy(), group[const.COL_Y_PRED].to_numpy(), metrics
        )
        rows.append(
            {
                **dict(zip(keys, key_values, strict=True)),
                const.COL_N_SAMPLES: len(group),
                **{f"{const.POOLED_PREFIX}{k}": v for k, v in pooled.items()},
            }
        )
    summary = pd.DataFrame(rows)
    per_fold = results.fold_metrics.groupby(keys, sort=False)[[m.value for m in metrics]]
    spread = (
        per_fold.mean()
        .add_prefix(const.FOLD_MEAN_PREFIX)
        .join(per_fold.std().add_prefix(const.FOLD_STD_PREFIX))
    )
    return summary.merge(spread.reset_index(), on=keys, how="left")


def select_best_models(summary: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pick the best model per target by the pooled selection metric under the primary strategy.

    Ties are broken by model order in ``models.candidates`` (first wins), which
    keeps selection deterministic.

    Returns:
        ``{target: {"model", "metric", "value", "strategy", "ranking"}}`` where
        ``ranking`` lists every candidate's score, best first.

    Raises:
        ValueError: If no model has a finite score for a target.
    """
    metric = const.Metric(cfg["models"]["selection_metric"])
    primary = cfg["validation"]["primary_strategy"]
    column = f"{const.POOLED_PREFIX}{metric.value}"
    ascending = not const.METRIC_GREATER_IS_BETTER[metric]
    selection: dict[str, dict[str, Any]] = {}
    for target in const.TARGETS.values():
        rows = summary[(summary[const.CV_COL_STRATEGY] == primary) & (summary[const.COL_TARGET] == target)]
        rows = rows[np.isfinite(rows[column])]
        if rows.empty:
            raise ValueError(f"No model has a finite {column} for target '{target}' under '{primary}'")
        ranked = rows.sort_values(column, ascending=ascending, kind="stable")
        best = ranked.iloc[0]
        selection[target] = {
            const.COL_MODEL: best[const.COL_MODEL],
            "metric": column,
            "value": float(best[column]),
            const.CV_COL_STRATEGY: primary,
            "ranking": {r[const.COL_MODEL]: float(r[column]) for _, r in ranked.iterrows()},
        }
        logger.info(
            "Selected %s for %s: %s=%.3f under %s",
            best[const.COL_MODEL],
            target,
            column,
            best[column],
            primary,
        )
    return selection


def log_optimism_gap(summary: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Log how much better each model looks under the random split than under the primary one.

    A large gap means the random split rewards memorising protein/family
    identity, i.e. it would over-promise performance on the OOD candidates.
    """
    metric = f"{const.POOLED_PREFIX}{cfg['models']['selection_metric']}"
    primary = cfg["validation"]["primary_strategy"]
    random_ = const.SplitStrategy.RANDOM.value
    if random_ not in set(summary[const.CV_COL_STRATEGY]) or primary == random_:
        return
    pivot = summary.pivot_table(
        index=[const.COL_MODEL, const.COL_TARGET], columns=const.CV_COL_STRATEGY, values=metric
    )
    for (model, target), row in pivot.iterrows():
        logger.info(
            "Optimism gap [%s|%s] %s: random=%.3f vs %s=%.3f (diff %+.3f)",
            model,
            target,
            metric,
            row[random_],
            primary,
            row[primary],
            row[random_] - row[primary],
        )


# =============================================================================
# Final models & artifacts
# =============================================================================
def fit_final_models(
    df: pd.DataFrame, selection: dict[str, dict[str, Any]], cfg: dict[str, Any], features: FeatureSet
) -> dict[str, Pipeline]:
    """Refit each target's selected pipeline on all labeled data."""
    finals = {}
    for target, choice in selection.items():
        pipeline = build_pipeline(cfg, const.ModelName(choice[const.COL_MODEL]), features)
        pipeline.fit(df[features.all], df[target].to_numpy(dtype=float))
        finals[target] = pipeline
        logger.info("Refit %s for %s on all %d labeled rows", choice[const.COL_MODEL], target, len(df))
    return finals


def _dump(obj: Any, path: Path) -> None:
    """Persist ``obj`` with joblib, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=const.JOBLIB_COMPRESSION)


def save_artifacts(
    results: CVResults,
    summary: pd.DataFrame,
    selection: dict[str, dict[str, Any]],
    finals: dict[str, Pipeline],
    features: FeatureSet,
    cfg: dict[str, Any],
) -> dict[str, Path]:
    """Write reports, fitted pipelines and a provenance record.

    Each saved pipeline contains its own fitted imputer, scaler and PCA, so a
    fold's transformers are always stored with the model they were fit for.
    Fold pipelines are saved for the primary strategy and the selected model
    per target only (the fold ensemble used for uncertainty in Phase 4).

    Returns:
        Mapping of artifact name to path (fold pipelines under one key).
    """
    paths: dict[str, Path] = {}
    for key, table in (
        ("oof_predictions", results.oof),
        ("cv_fold_metrics", results.fold_metrics),
        ("cv_metrics_summary", summary),
    ):
        paths[key] = resolve_path(cfg, "reports_dir", key)
        write_table(table, paths[key])

    paths["model_selection"] = resolve_path(cfg, "reports_dir", "model_selection")
    paths["model_selection"].write_text(json.dumps(selection, indent=2), encoding=const.CONFIG_ENCODING)

    models_dir = resolve_path(cfg, "models_dir")
    final_files = {}
    for target, pipeline in finals.items():
        name = const.FINAL_MODEL_TEMPLATE.format(target=target, model=selection[target][const.COL_MODEL])
        _dump(pipeline, models_dir / name)
        final_files[target] = name
        paths[f"final_model[{target}]"] = models_dir / name

    fold_files = []
    if results.fold_pipelines:
        fold_dir = models_dir / const.FOLD_MODELS_SUBDIR
        strategy = cfg["validation"]["primary_strategy"]
        for (target, model, fold), pipeline in sorted(results.fold_pipelines.items()):
            if model != selection[target][const.COL_MODEL]:
                continue  # only the selected model's folds are used downstream
            name = const.FOLD_MODEL_TEMPLATE.format(strategy=strategy, fold=fold, target=target, model=model)
            _dump(pipeline, fold_dir / name)
            fold_files.append(f"{const.FOLD_MODELS_SUBDIR}/{name}")
        paths["fold_pipelines"] = fold_dir

    metadata = {
        "random_seed": cfg["project"]["random_seed"],
        "primary_strategy": cfg["validation"]["primary_strategy"],
        "selection": selection,
        "final_models": final_files,
        "fold_pipelines": fold_files,
        "features": {"tabular": features.tabular, "embedding": features.embedding},
        "config": cfg,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
    }
    paths["training_metadata"] = resolve_path(cfg, "models_dir", "training_metadata")
    paths["training_metadata"].parent.mkdir(parents=True, exist_ok=True)
    paths["training_metadata"].write_text(
        json.dumps(metadata, indent=2, default=str), encoding=const.CONFIG_ENCODING
    )
    for key, path in paths.items():
        logger.info("Wrote %s -> %s", key, path)
    return paths


# =============================================================================
# Orchestration
# =============================================================================
def run_training(cfg: dict[str, Any]) -> tuple[CVResults, pd.DataFrame, dict[str, dict[str, Any]]]:
    """Execute Phase 3: CV all models on both targets, select, refit and save.

    Returns:
        ``(cv_results, metrics_summary, selection)``.
    """
    df, features = load_training_data(cfg)
    splits, _ = build_cv_splits(df, cfg)
    results = run_cross_validation(df, splits, cfg, features)
    summary = summarize_metrics(results, cfg)
    log_optimism_gap(summary, cfg)
    selection = select_best_models(summary, cfg)
    finals = fit_final_models(df, selection, cfg, features) if cfg["models"]["refit_on_full_data"] else {}
    save_artifacts(results, summary, selection, finals, features, cfg)
    logger.info("Phase 3 complete: %d fold fits", len(results.fold_metrics))
    return results, summary, selection


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    run_training(config)
