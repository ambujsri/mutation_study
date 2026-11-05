"""Fold-local training of the dual-branch network (Task 4, second architecture).

Runs :class:`src.torch_model_v2.ProteinMutationDualBranchNN` over exactly the
folds `src/validation.py` produces, so its numbers sit in the same tables as
the sklearn baselines and the first network without reconciliation.

Run directly (requires Phase 1)::

    python -m src.train_torch_v2 --config config/config.yaml

**Two preprocessors, not one.** The architecture needs its two feature blocks
kept apart, so this module builds a separate fold-local
:class:`~sklearn.compose.ColumnTransformer` per block rather than the single
combined one `src/model.py` assembles for the concatenating models. Each is
still fitted on inner-training rows only, so the leakage guarantee is
unchanged — it is enforced twice instead of once.

The embedding preprocessor normally runs **without PCA**
(``torch_v2.embedding_pca: false``). The learned bottleneck replaces it. PCA
chooses the directions of greatest variance in a protein-level vector, and
under a family-grouped split most of that variance is *between families* — the
one component guaranteed not to transfer to a held-out family. A supervised
bottleneck is free to keep a smaller, different subspace. Setting the flag true
stacks both reductions, which is available but rarely what you want.

**Leakage boundaries are inherited, not re-invented.** Early stopping runs on
an inner split carved out of the training fold by
:func:`src.train_torch.inner_split`, grouped the same way the outer split is::

    outer fold ──┬── training rows ──┬── inner train   (gradient steps)
                 │                   └── inner val     (early stopping)
                 └── validation rows ─── untouched until the final predict

Stopping on the outer validation fold would choose the epoch that scores best
on the very rows used to report performance. Every fitted object here — both
imputers, both scalers, the optional PCA, the target mean/std and the weights —
is fitted on inner-training rows alone.

Reproducibility: seeds for Python, NumPy and Torch are re-set at the start of
every fold from ``project.random_seed``, so a fold's result does not depend on
how many folds ran before it.
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
from sklearn.compose import ColumnTransformer

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table, write_table
from src.evaluate import make_torch_figures
from src.model import FeatureSet, build_preprocessor
from src.torch_model_v2 import (
    DualBranchConfig,
    ProteinMutationDualBranchNN,
    count_branch_parameters,
    dual_branch_config_from,
)
from src.torch_training import FoldInputs, FoldResult, FoldTrainer
from src.train import load_training_data
from src.train_torch import compare_with_baselines as compare_models
from src.validation import generate_folds

logger = logging.getLogger(__name__)

TARGET_STABILITY = const.TARGET_STABILITY
TARGET_FUNCTION = const.TARGET_FUNCTION


# =============================================================================
# The trainer
# =============================================================================
class DualBranchTrainer(FoldTrainer[DualBranchConfig, ProteinMutationDualBranchNN]):
    """Fold-local trainer for the dual-branch network.

    Shares every leakage rule and the whole epoch loop with
    :class:`~src.train_torch.MultitaskTrainer` through :class:`FoldTrainer`.
    What differs is only what this architecture needs: **two** fold-local
    preprocessors instead of one, so the tabular block and the embedding block
    reach the network as separate tensors, and a
    :class:`~src.torch_model_v2.ProteinMutationDualBranchNN` to consume them.
    """

    model_name = const.TORCH_V2_MODEL_NAME

    def branch_preprocessors(self) -> tuple[ColumnTransformer, ColumnTransformer]:
        """Build one unfitted preprocessor per branch.

        Both are assembled by :func:`src.model.build_preprocessor` from a
        single-block :class:`~src.model.FeatureSet`, so the imputation
        strategy and the scaler are the ones every other model uses.

        The embedding block's PCA is governed by ``torch_v2.embedding_pca``:
        when it is false the block's config is overridden locally so the raw
        embedding reaches the learned bottleneck. The override copies the
        reduction sub-dictionary, leaving the caller's config untouched.

        Raises:
            ValueError: If either feature block is empty.
        """
        features, cfg = self.features, self.cfg
        if not features.tabular or not features.embedding:
            raise ValueError(
                "The dual-branch model needs both feature blocks, got "
                f"{len(features.tabular)} tabular and {len(features.embedding)} embedding columns"
            )
        tabular = build_preprocessor(cfg, FeatureSet(tabular=features.tabular, embedding=[]))

        embedding_cfg = dict(cfg)
        if not self.config.embedding_pca:
            reduction = dict(cfg["features"]["embedding_reduction"])
            reduction["enabled"] = False
            embedding_cfg["features"] = {**cfg["features"], "embedding_reduction": reduction}
        embedding = build_preprocessor(
            embedding_cfg, FeatureSet(tabular=[], embedding=features.embedding)
        )
        return tabular, embedding

    def fit_fold_inputs(
        self, df: pd.DataFrame, inner_train_idx: np.ndarray, inner_val_idx: np.ndarray
    ) -> FoldInputs:
        """Fit both preprocessors on the inner-training rows.

        The leakage guarantee is unchanged from the single-block case; it is
        simply enforced twice instead of once.
        """
        tabular_pre, embedding_pre = self.branch_preprocessors()
        tabular_frame = df[self.features.tabular]
        embedding_frame = df[self.features.embedding]
        train = (
            tabular_pre.fit_transform(tabular_frame.iloc[inner_train_idx]),
            embedding_pre.fit_transform(embedding_frame.iloc[inner_train_idx]),
        )
        val = (
            tabular_pre.transform(tabular_frame.iloc[inner_val_idx]),
            embedding_pre.transform(embedding_frame.iloc[inner_val_idx]),
        )
        return FoldInputs(
            train=train,
            val=val,
            transform=lambda frame: (
                tabular_pre.transform(frame[self.features.tabular]),
                embedding_pre.transform(frame[self.features.embedding]),
            ),
            fitted=(tabular_pre, embedding_pre),
        )

    def build_model(self, widths: tuple[int, ...]) -> ProteinMutationDualBranchNN:
        """Construct the dual-branch network for the tabular and embedding widths."""
        return ProteinMutationDualBranchNN(widths[0], widths[1], self.config)


def first_network_metrics(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the v1 network's metrics so v2 is measured against it too.

    Returns an empty frame when the first network has not been run; the
    comparison then falls back to the sklearn baselines alone.
    """
    path = resolve_path(cfg, "metrics_dir", "torch_cv_metrics")
    if not path.is_file():
        logger.info("No %s metrics at %s; comparing against the baselines only",
                    const.TORCH_MODEL_NAME, path)
        return pd.DataFrame()
    return read_table(path)


def to_markdown(comparison: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Render the dual-branch comparison as a Markdown section for the report."""
    selection = cfg["models"]["selection_metric"]
    strategy = comparison[const.CV_COL_STRATEGY].iloc[0] if not comparison.empty else "n/a"
    lines = [
        "# Dual-branch network vs. the other models",
        "",
        f"- Cross-validation: `{strategy}`, identical folds for every model",
        f"- Delta columns are measured against the **best other model for that target** by "
        f"`{const.POOLED_PREFIX}{selection}`",
        "",
        "The architecture routes the 22 tabular columns and the 256-dimensional protein "
        "embedding down separate branches, squeezes the embedding through a learned "
        f"{cfg['torch_v2']['embedding_branch']['bottleneck_dim']}-dimensional bottleneck, and "
        "fuses them through residual blocks. Early stopping runs on a split carved out of the "
        "training fold, so the out-of-fold rows below were never used to pick the stopping epoch.",
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
    artifacts: dict[int, FoldResult[ProteinMutationDualBranchNN]],
    features: FeatureSet,
    cfg: dict[str, Any],
    config: DualBranchConfig,
) -> dict[str, Path]:
    """Write metrics, training history, fold weights and a provenance record.

    Each fold's weights are saved beside *both* preprocessors that produced its
    inputs, because the three are only meaningful together.

    Returns:
        Mapping of artifact name to written path.
    """
    paths: dict[str, Path] = {}
    for key, table in (
        ("torch_v2_oof_predictions", oof),
        ("torch_v2_cv_metrics", summary),
        ("torch_v2_fold_summary", fold_summary),
        ("torch_v2_history", history),
    ):
        paths[key] = resolve_path(cfg, "metrics_dir", key)
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        write_table(table, paths[key])

    if not comparison.empty:
        paths["torch_v2_comparison"] = resolve_path(cfg, "metrics_dir", "torch_v2_comparison")
        write_table(comparison, paths["torch_v2_comparison"])
        paths["torch_v2_comparison_markdown"] = resolve_path(
            cfg, "metrics_dir", "torch_v2_comparison_markdown"
        )
        paths["torch_v2_comparison_markdown"].write_text(
            to_markdown(comparison, cfg), encoding=const.CONFIG_ENCODING
        )

    models_dir = resolve_path(cfg, "models_dir") / const.TORCH_V2_MODELS_SUBDIR
    models_dir.mkdir(parents=True, exist_ok=True)
    saved_folds: list[str] = []
    if config.save_fold_models:
        for index, result in sorted(artifacts.items()):
            model = result.model
            tabular_pre, embedding_pre = result.inputs.fitted
            strategy = cfg["validation"]["primary_strategy"]
            names = {
                "weights": const.TORCH_V2_FOLD_MODEL_TEMPLATE,
                "tabular": const.TORCH_V2_TABULAR_PREPROCESSOR_TEMPLATE,
                "embedding": const.TORCH_V2_EMBEDDING_PREPROCESSOR_TEMPLATE,
            }
            resolved = {
                key: models_dir / template.format(strategy=strategy, fold=index)
                for key, template in names.items()
            }
            torch.save(model.state_dict(), resolved["weights"])
            joblib.dump(tabular_pre, resolved["tabular"])
            joblib.dump(embedding_pre, resolved["embedding"])
            saved_folds.append(resolved["weights"].name)
        paths["torch_v2_fold_models"] = models_dir

    example = next(iter(artifacts.values())).model if artifacts else None
    metadata: dict[str, Any] = {
        "model": const.TORCH_V2_MODEL_NAME,
        "random_seed": config.seed,
        "device": config.device,
        "architecture": {
            "tabular_hidden_dims": list(config.tabular_hidden_dims),
            "embedding_hidden_dims": list(config.embedding_hidden_dims),
            "bottleneck_dim": config.bottleneck_dim,
            "embedding_pca": config.embedding_pca,
            "fusion_hidden_dim": config.fusion_hidden_dim,
            "n_residual_blocks": config.n_residual_blocks,
            "head_hidden_dim": config.head_hidden_dim,
            "activation": config.activation.value,
            "batch_norm": config.batch_norm,
            "dropout": config.dropout,
            "n_tabular_inputs": example.n_tabular if example is not None else None,
            "n_embedding_inputs": example.n_embedding if example is not None else None,
        },
        "parameters_by_branch": count_branch_parameters(example) if example is not None else {},
        "optimizer": {
            "name": const.TORCH_OPTIMIZER_NAME,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        "loss_weights": {"stability": config.loss_weights[0], "function": config.loss_weights[1]},
        "features": {"tabular": features.tabular, "embedding": features.embedding},
        "fold_models": saved_folds,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    paths["torch_v2_metadata"] = models_dir / const.TORCH_V2_METADATA_FILE
    paths["torch_v2_metadata"].write_text(
        json.dumps(metadata, indent=2, default=str), encoding=const.CONFIG_ENCODING
    )
    for key, path in paths.items():
        logger.info("Wrote %s -> %s", key, path)
    return paths


# =============================================================================
# Orchestration
# =============================================================================
def run_dual_branch_training(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate the dual-branch network and benchmark it against every other model.

    Returns:
        ``(metrics_summary, comparison)``.
    """
    config = dual_branch_config_from(cfg)
    df, features = load_training_data(cfg)
    strategy = const.SplitStrategy(cfg["validation"]["primary_strategy"])
    validation = cfg["validation"]
    folds = generate_folds(
        df, strategy, validation["n_splits"], validation["shuffle"], cfg["project"]["random_seed"]
    )
    logger.info(
        "Training %s on %d %s folds | device=%s | tabular=%s | embedding=%s -> %d | "
        "fusion=%d x %d residual | loss weights %.2f/%.2f",
        const.TORCH_V2_MODEL_NAME,
        len(folds),
        strategy.value,
        config.device,
        list(config.tabular_hidden_dims),
        list(config.embedding_hidden_dims),
        config.bottleneck_dim,
        config.fusion_hidden_dim,
        config.n_residual_blocks,
        *config.loss_weights,
    )

    trainer = DualBranchTrainer(cfg, config, features)
    oof, history, fold_summary, artifacts = trainer.cross_validate(df, folds)
    summary = trainer.summarize(oof)
    comparison = compare_models(
        summary,
        cfg,
        model_name=const.TORCH_V2_MODEL_NAME,
        reference=first_network_metrics(cfg),
    )
    save_artifacts(
        oof, summary, history, fold_summary, comparison, artifacts, features, cfg, config
    )
    figures = make_torch_figures(
        oof,
        history,
        fold_summary,
        comparison,
        cfg,
        figure_keys=const.TORCH_V2_FIGURE_FILE_KEYS,
        model_label=const.TORCH_V2_MODEL_NAME,
    )
    logger.info(
        "Dual-branch training complete: %d folds, %d out-of-fold predictions, %d figures",
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
    run_dual_branch_training(config_dict)
