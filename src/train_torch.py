"""Fold-local training of the multitask neural network (assignment Task 4).

Runs :class:`src.torch_model.ProteinMutationMultitaskNN` over exactly the folds
`src/validation.py` produces, with exactly the fold-local preprocessing
`src/model.py` builds for the sklearn baselines. Holding the data pipeline
fixed is what makes the comparison in `torch_vs_baselines.md` a statement about
the *model* rather than about the plumbing around it.

Run directly (requires Phase 1)::

    python -m src.train_torch --config config/config.yaml

**Two leakage boundaries, not one.** The sklearn baselines have a single
boundary: the outer validation fold is untouched until ``predict``. A neural
network needs a second one, because early stopping is itself a fitting
decision. Choosing the stopping epoch on the outer validation fold would select
the epoch that happens to score best on the very rows used to report
performance, and the resulting out-of-fold metric would be optimistic for the
same reason a random split is. So this module carves an **inner** validation
split out of the *training fold only*:

    outer fold ──┬── training rows ──┬── inner train   (gradient steps)
                 │                   └── inner val     (early stopping)
                 └── validation rows ─── untouched until the final predict

The inner split is grouped the same way the outer one is, so the stopping
decision is made under the same kind of generalisation the metric reports.

**Preprocessing** (median imputation, scaling, PCA on the 256-d embedding) is
fitted on the training fold and applied unchanged to the inner-validation and
outer-validation rows. The stability target is standardised on the inner
training rows alone and predictions are inverse-transformed before any metric
is computed, so reported numbers are always in kcal/mol.
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
import torch

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table, write_table
from src.evaluate import configured_metrics, make_torch_figures
from src.model import FeatureSet, build_preprocessor
from src.torch_model import (
    ProteinMutationMultitaskNN,
    TorchConfig,
    count_parameters,
    torch_config_from,
)
from src.torch_training import FoldInputs, FoldResult, FoldTrainer
from src.train import load_training_data
from src.validation import generate_folds

logger = logging.getLogger(__name__)

TARGET_STABILITY = const.TARGET_STABILITY
TARGET_FUNCTION = const.TARGET_FUNCTION

# =============================================================================
# The trainer
# =============================================================================
class MultitaskTrainer(FoldTrainer[TorchConfig, ProteinMutationMultitaskNN]):
    """Fold-local trainer for the shared-backbone network.

    All the machinery — seeding, the inner split, the target scaler, the epoch
    loop and early stopping — is inherited from :class:`FoldTrainer`. The two
    things specific to this architecture are supplied here: it consumes a
    single concatenated input block, built by the same
    :func:`src.model.build_preprocessor` the sklearn baselines use, and it
    constructs a :class:`~src.torch_model.ProteinMutationMultitaskNN`.
    """

    model_name = const.TORCH_MODEL_NAME

    def fit_fold_inputs(
        self, df: pd.DataFrame, inner_train_idx: np.ndarray, inner_val_idx: np.ndarray
    ) -> FoldInputs:
        """Fit one ColumnTransformer on the inner-training rows.

        A single block: the tabular columns and the PCA-reduced embedding are
        concatenated before the network sees them, which is exactly what the
        dual-branch architecture was written to avoid.
        """
        preprocessor = build_preprocessor(self.cfg, self.features)
        x_all = df[self.features.all]
        train = preprocessor.fit_transform(x_all.iloc[inner_train_idx])
        val = preprocessor.transform(x_all.iloc[inner_val_idx])
        return FoldInputs(
            train=(train,),
            val=(val,),
            transform=lambda frame: (preprocessor.transform(frame[self.features.all]),),
            fitted=(preprocessor,),
        )

    def build_model(self, widths: tuple[int, ...]) -> ProteinMutationMultitaskNN:
        """Construct the shared-backbone network for the single input block."""
        return ProteinMutationMultitaskNN(widths[0], self.config)


def compare_with_baselines(
    torch_summary: pd.DataFrame,
    cfg: dict[str, Any],
    model_name: str = const.TORCH_MODEL_NAME,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Place a network beside the other models on the same folds.

    Reads the baseline metrics `src/train.py` saved rather than refitting, so
    the comparison cannot disagree with the published baseline numbers.

    Args:
        torch_summary: Pooled metrics for the network being placed.
        cfg: Loaded configuration.
        model_name: The network's name in the metrics tables; every other row
            counts as a reference model it is measured against.
        reference: Extra reference rows to compare against on top of the
            sklearn baselines — used by the dual-branch run to measure itself
            against the first network as well.

    Returns:
        One row per target per model under the evaluated strategy, with the
        network's signed difference against the **best** reference model for
        that target. An empty frame if no reference metrics exist.
    """
    path = resolve_path(cfg, "metrics_dir", "cv_metrics_summary")
    if not path.is_file():
        logger.warning("Baseline metrics not found at %s; skipping comparison", path)
        return pd.DataFrame()

    baselines = read_table(path)
    if reference is not None and not reference.empty:
        baselines = pd.concat([baselines, reference], ignore_index=True)
    strategy = torch_summary[const.CV_COL_STRATEGY].iloc[0]
    baselines = baselines[baselines[const.CV_COL_STRATEGY] == strategy]
    combined = pd.concat([baselines, torch_summary], ignore_index=True)

    metrics = configured_metrics(cfg)
    selection = const.Metric(cfg["models"]["selection_metric"])
    column = f"{const.POOLED_PREFIX}{selection.value}"
    greater_is_better = const.METRIC_GREATER_IS_BETTER[selection]

    rows = []
    for target in const.TARGETS.values():
        target_rows = combined[combined[const.COL_TARGET] == target]
        baseline_rows = target_rows[target_rows[const.COL_MODEL] != model_name]
        if baseline_rows.empty:
            continue
        best = (
            baseline_rows.loc[baseline_rows[column].idxmax()]
            if greater_is_better
            else baseline_rows.loc[baseline_rows[column].idxmin()]
        )
        for _, row in target_rows.iterrows():
            record = {
                const.COL_TARGET: target,
                const.COL_MODEL: row[const.COL_MODEL],
                const.CV_COL_STRATEGY: strategy,
                **{
                    f"{const.POOLED_PREFIX}{m.value}": float(row[f"{const.POOLED_PREFIX}{m.value}"])
                    for m in metrics
                },
            }
            for metric in metrics:
                name = f"{const.POOLED_PREFIX}{metric.value}"
                record[f"{const.TORCH_COL_DELTA_PREFIX}{metric.value}"] = float(row[name]) - float(
                    best[name]
                )
            rows.append(record)
        logger.info(
            "[%s] best baseline is %s (%s=%.3f); %s scores %.3f",
            target,
            best[const.COL_MODEL],
            column,
            best[column],
            model_name,
            float(torch_summary[torch_summary[const.COL_TARGET] == target][column].iloc[0]),
        )
    return pd.DataFrame(rows)


def to_markdown(comparison: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Render the model comparison as a Markdown section for the technical report."""
    selection = cfg["models"]["selection_metric"]
    strategy = comparison[const.CV_COL_STRATEGY].iloc[0] if not comparison.empty else "n/a"
    lines = [
        "# Multitask neural network vs. the classical baselines",
        "",
        f"- Cross-validation: `{strategy}`, identical folds and identical fold-local "
        "preprocessing for every model",
        f"- Delta columns are measured against the **best baseline for that target** by "
        f"`{const.POOLED_PREFIX}{selection}`",
        "",
        "The network shares one backbone across both targets and is early-stopped on a split "
        "carved out of the training fold, so the out-of-fold rows below were never used to pick "
        "the stopping epoch.",
        "",
    ]
    for target in const.TARGETS.values():
        rows = comparison[comparison[const.COL_TARGET] == target]
        if rows.empty:
            continue
        lines += [
            f"## {target}",
            "",
            rows.drop(columns=[const.COL_TARGET, const.CV_COL_STRATEGY]).to_markdown(
                index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT
            ),
            "",
        ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Artifacts
# =============================================================================
def save_artifacts(
    oof: pd.DataFrame,
    summary: pd.DataFrame,
    history: pd.DataFrame,
    fold_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    artifacts: dict[int, FoldResult[ProteinMutationMultitaskNN]],
    features: FeatureSet,
    cfg: dict[str, Any],
    config: TorchConfig,
) -> dict[str, Path]:
    """Write metrics, training history, fold weights and a provenance record.

    Each fold's weights are saved beside the preprocessor that produced its
    inputs, because the two are only meaningful together.

    Returns:
        Mapping of artifact name to written path.
    """
    paths: dict[str, Path] = {}
    for key, table in (
        ("torch_oof_predictions", oof),
        ("torch_cv_metrics", summary),
        ("torch_fold_summary", fold_summary),
        ("torch_history", history),
    ):
        paths[key] = resolve_path(cfg, "metrics_dir", key)
        write_table(table, paths[key])

    if not comparison.empty:
        paths["torch_comparison"] = resolve_path(cfg, "metrics_dir", "torch_comparison")
        write_table(comparison, paths["torch_comparison"])
        paths["torch_comparison_markdown"] = resolve_path(
            cfg, "metrics_dir", "torch_comparison_markdown"
        )
        paths["torch_comparison_markdown"].parent.mkdir(parents=True, exist_ok=True)
        paths["torch_comparison_markdown"].write_text(
            to_markdown(comparison, cfg), encoding=const.CONFIG_ENCODING
        )

    saved_folds: list[str] = []
    if cfg["torch"]["save_fold_models"]:
        models_dir = resolve_path(cfg, "models_dir") / const.TORCH_MODELS_SUBDIR
        models_dir.mkdir(parents=True, exist_ok=True)
        strategy = oof[const.CV_COL_STRATEGY].iloc[0]
        for fold_index, result in sorted(artifacts.items()):
            model, preprocessor, scaler = result.model, result.inputs.fitted[0], result.scaler
            weights = const.TORCH_FOLD_MODEL_TEMPLATE.format(strategy=strategy, fold=fold_index)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "n_features": model.n_features,
                    "target_mean": scaler.mean,
                    "target_std": scaler.std,
                    "target_scaled": scaler.enabled,
                },
                models_dir / weights,
            )
            joblib.dump(
                preprocessor,
                models_dir
                / const.TORCH_PREPROCESSOR_TEMPLATE.format(strategy=strategy, fold=fold_index),
                compress=const.JOBLIB_COMPRESSION,
            )
            saved_folds.append(f"{const.TORCH_MODELS_SUBDIR}/{weights}")
        paths["torch_fold_models"] = models_dir

    example = next(iter(artifacts.values())).model if artifacts else None  # noqa: SIM108
    metadata = {
        "model": const.TORCH_MODEL_NAME,
        "random_seed": config.seed,
        "device": config.device,
        "architecture": {
            "hidden_dims": list(config.hidden_dims),
            "head_hidden_dim": config.head_hidden_dim,
            "activation": config.activation.value,
            "batch_norm": config.batch_norm,
            "dropout": config.dropout,
            "n_features": example.n_features if example is not None else None,
            "n_parameters": count_parameters(example) if example is not None else None,
        },
        "optimization": {
            "optimizer": const.TORCH_OPTIMIZER_NAME,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "loss_weights": {
                TARGET_STABILITY: config.loss_weights[0],
                TARGET_FUNCTION: config.loss_weights[1],
            },
            "early_stopping": {
                "inner_val_fraction": config.inner_val_fraction,
                "patience": config.patience,
                "min_delta": config.min_delta,
            },
        },
        "features": {"tabular": features.tabular, "embedding": features.embedding},
        "fold_models": saved_folds,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    paths["torch_metadata"] = (
        resolve_path(cfg, "models_dir") / const.TORCH_MODELS_SUBDIR / const.TORCH_METADATA_FILE
    )
    paths["torch_metadata"].parent.mkdir(parents=True, exist_ok=True)
    paths["torch_metadata"].write_text(
        json.dumps(metadata, indent=2, default=str), encoding=const.CONFIG_ENCODING
    )
    for key, path in paths.items():
        logger.info("Wrote %s -> %s", key, path)
    return paths


# =============================================================================
# Orchestration
# =============================================================================
def run_torch_training(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Task 4: cross-validate the multitask network and benchmark it.

    Returns:
        ``(metrics_summary, comparison)``.
    """
    config = torch_config_from(cfg)
    df, features = load_training_data(cfg)
    strategy = const.SplitStrategy(cfg["validation"]["primary_strategy"])
    validation = cfg["validation"]
    folds = generate_folds(
        df, strategy, validation["n_splits"], validation["shuffle"], cfg["project"]["random_seed"]
    )
    logger.info(
        "Training %s on %d %s folds | device=%s | hidden=%s | loss weights %.2f/%.2f",
        const.TORCH_MODEL_NAME,
        len(folds),
        strategy.value,
        config.device,
        list(config.hidden_dims),
        *config.loss_weights,
    )

    trainer = MultitaskTrainer(cfg, config, features)
    oof, history, fold_summary, artifacts = trainer.cross_validate(df, folds)
    summary = trainer.summarize(oof)
    comparison = compare_with_baselines(summary, cfg)
    save_artifacts(
        oof, summary, history, fold_summary, comparison, artifacts, features, cfg, config
    )
    figures = make_torch_figures(oof, history, fold_summary, comparison, cfg)
    logger.info(
        "Task 4 complete: %d folds, %d out-of-fold predictions, %d figures",
        len(folds),
        len(oof),
        len(figures),
    )
    return summary, comparison


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_config(args.config)
    setup_logging(config_dict)
    run_torch_training(config_dict)
