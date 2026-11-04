"""Training and cross-validation execution for both mutation-effect targets.

Phase 3 of the Medvolt.AI mutation-effect pipeline. This module is the
*execution* layer: it owns the fold loop, the artifacts and the provenance
record, while the architecture lives in ``src.model`` and every metric,
aggregation and selection rule lives in ``src.evaluate``.

For every CV strategy (``validation.strategies``), every candidate model
(``models.candidates``) and both targets (``stability_ddG``,
``functional_score``) a fresh ``src.model.build_pipeline`` is fit on each
training fold and used to predict that fold's held-out rows. Because
imputation, scaling and PCA live inside the pipeline, ``fit`` on the training
fold is the only place their statistics are learned; the validation fold is
only ever passed to ``predict``. This is the leakage guarantee, and it is
asserted in ``tests/test_train.py``.

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
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.pipeline import Pipeline

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table, write_table
from src.evaluate import (
    compute_metrics,
    configured_metrics,
    log_optimism_gap,
    pooled_metrics,
    select_best_models,
    summarize_metrics,
)
from src.model import CVResults, FeatureSet, build_pipeline
from src.validation import Fold, build_cv_splits

logger = logging.getLogger(__name__)


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
    metrics = configured_metrics(cfg)
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
                pooled = pooled_metrics(oof, configured_metrics(cfg))
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
        paths[key] = resolve_path(cfg, "metrics_dir", key)
        write_table(table, paths[key])

    paths["model_selection"] = resolve_path(cfg, "metrics_dir", "model_selection")
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
