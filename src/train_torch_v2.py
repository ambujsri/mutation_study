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
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table, write_table
from src.evaluate import compute_metrics, configured_metrics, make_torch_figures
from src.model import FeatureSet, build_preprocessor
from src.torch_model import multitask_loss
from src.torch_model_v2 import (
    DualBranchConfig,
    ProteinMutationDualBranchNN,
    count_branch_parameters,
    dual_branch_config_from,
)
from src.train import load_training_data
from src.train_torch import TargetScaler, inner_split, seed_everything, to_tensor
from src.train_torch import compare_with_baselines as compare_models
from src.validation import Fold, generate_folds

logger = logging.getLogger(__name__)

TARGET_STABILITY = const.TARGET_STABILITY
TARGET_FUNCTION = const.TARGET_FUNCTION

# Returned by train_one_fold: the model plus everything needed to reproduce its
# inputs. The two preprocessors and the target scaler are only meaningful
# together with the weights, so they travel as one object.
FoldArtifacts = tuple[ProteinMutationDualBranchNN, ColumnTransformer, ColumnTransformer, TargetScaler]


# =============================================================================
# Fold-local preprocessing, one transformer per branch
# =============================================================================
def build_branch_preprocessors(
    cfg: dict[str, Any], features: FeatureSet, config: DualBranchConfig
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """Build one preprocessor per branch.

    Both are assembled by :func:`src.model.build_preprocessor` from a
    single-block :class:`~src.model.FeatureSet`, so the imputation strategy and
    the scaler are exactly the ones every other model in the project uses.

    The embedding block's PCA is governed by ``torch_v2.embedding_pca``: when
    it is false the block's config is overridden locally so the raw embedding
    reaches the learned bottleneck. The override is a shallow copy of the
    reduction sub-dictionary, so the caller's config is left untouched.

    Returns:
        ``(tabular_preprocessor, embedding_preprocessor)``, both unfitted.

    Raises:
        ValueError: If either feature block is empty.
    """
    if not features.tabular or not features.embedding:
        raise ValueError(
            "The dual-branch model needs both feature blocks, got "
            f"{len(features.tabular)} tabular and {len(features.embedding)} embedding columns"
        )
    tabular = build_preprocessor(cfg, FeatureSet(tabular=features.tabular, embedding=[]))

    embedding_cfg = dict(cfg)
    if not config.embedding_pca:
        reduction = dict(cfg["features"]["embedding_reduction"])
        reduction["enabled"] = False
        embedding_cfg["features"] = {**cfg["features"], "embedding_reduction": reduction}
    embedding = build_preprocessor(
        embedding_cfg, FeatureSet(tabular=[], embedding=features.embedding)
    )
    return tabular, embedding


def make_loader(
    tabular: Tensor,
    embedding: Tensor,
    y_stability: Tensor,
    y_function: Tensor,
    config: DualBranchConfig,
    *,
    shuffle: bool,
) -> DataLoader[tuple[Tensor, ...]]:
    """Wrap the two input blocks and both targets in a seeded, single-process loader.

    ``num_workers=0`` and an explicit generator keep batch order reproducible.

    ``drop_last`` is enabled for shuffled (training) loaders because batch
    normalisation cannot compute a batch variance from a single row: a training
    fold whose size leaves a remainder of one would otherwise raise partway
    through an epoch.
    """
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    dataset = TensorDataset(tabular, embedding, y_stability, y_function)
    drop_last = shuffle and len(dataset) % config.batch_size == 1
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        drop_last=drop_last,
    )


# =============================================================================
# Training one fold
# =============================================================================
def evaluate_loss(
    model: ProteinMutationDualBranchNN,
    loader: DataLoader[tuple[Tensor, ...]],
    config: DualBranchConfig,
) -> float:
    """Mean weighted multitask loss over a loader, with the model in eval mode."""
    model.eval()
    total, n_rows = 0.0, 0
    with torch.no_grad():
        for tabular, embedding, y_stability, y_function in loader:
            loss, _, _ = multitask_loss(
                model(tabular, embedding), (y_stability, y_function), config.loss_weights
            )
            total += loss.item() * tabular.shape[0]
            n_rows += tabular.shape[0]
    return total / max(n_rows, 1)


def train_one_fold(
    df: pd.DataFrame,
    fold: Fold,
    features: FeatureSet,
    cfg: dict[str, Any],
    config: DualBranchConfig,
) -> tuple[FoldArtifacts, pd.DataFrame, dict[str, Any]]:
    """Fit the dual-branch network on one fold, early-stopping on an inner split.

    Returns:
        ``(artifacts, history, summary)`` where ``history`` has one row per
        epoch and ``summary`` records the stopping behaviour for the
        provenance table.
    """
    seed_everything(config.seed)
    inner_train_idx, inner_val_idx = inner_split(df, fold.train_idx, fold.strategy, config)

    tabular_pre, embedding_pre = build_branch_preprocessors(cfg, features, config)
    tabular_frame, embedding_frame = df[features.tabular], df[features.embedding]
    x_tab_train = tabular_pre.fit_transform(tabular_frame.iloc[inner_train_idx])
    x_emb_train = embedding_pre.fit_transform(embedding_frame.iloc[inner_train_idx])
    x_tab_val = tabular_pre.transform(tabular_frame.iloc[inner_val_idx])
    x_emb_val = embedding_pre.transform(embedding_frame.iloc[inner_val_idx])

    scaler = TargetScaler(enabled=config.standardize_stability_target)
    scaler.fit(df[TARGET_STABILITY].to_numpy(dtype=float)[inner_train_idx])

    def targets(idx: np.ndarray) -> tuple[Tensor, Tensor]:
        stability = scaler.transform(df[TARGET_STABILITY].to_numpy(dtype=float)[idx])
        function = df[TARGET_FUNCTION].to_numpy(dtype=float)[idx]
        return to_tensor(stability, config.device), to_tensor(function, config.device)

    train_loader = make_loader(
        to_tensor(x_tab_train, config.device),
        to_tensor(x_emb_train, config.device),
        *targets(inner_train_idx),
        config,
        shuffle=True,
    )
    val_loader = make_loader(
        to_tensor(x_tab_val, config.device),
        to_tensor(x_emb_val, config.device),
        *targets(inner_val_idx),
        config,
        shuffle=False,
    )

    model = ProteinMutationDualBranchNN(
        x_tab_train.shape[1], x_emb_train.shape[1], config
    ).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_loss, best_epoch = float("inf"), 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    records: list[dict[str, Any]] = []
    epochs_without_gain = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running, n_rows = 0.0, 0
        for batch_tab, batch_emb, batch_stability, batch_function in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = multitask_loss(
                model(batch_tab, batch_emb),
                (batch_stability, batch_function),
                config.loss_weights,
            )
            loss.backward()
            optimizer.step()
            running += loss.detach().item() * batch_tab.shape[0]
            n_rows += batch_tab.shape[0]
        train_loss = running / max(n_rows, 1)
        val_loss = evaluate_loss(model, val_loader, config)
        records.append(
            {
                const.CV_COL_STRATEGY: fold.strategy.value,
                const.CV_COL_FOLD: fold.index,
                const.TORCH_COL_EPOCH: epoch,
                const.TORCH_COL_TRAIN_LOSS: train_loss,
                const.TORCH_COL_VAL_LOSS: val_loss,
            }
        )

        if val_loss < best_loss - config.min_delta:
            best_loss, best_epoch, epochs_without_gain = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.patience:
                break

    # Always report the best epoch's network, not the last one's.
    model.load_state_dict(best_state)
    summary = {
        const.CV_COL_STRATEGY: fold.strategy.value,
        const.CV_COL_FOLD: fold.index,
        const.TORCH_COL_BEST_EPOCH: best_epoch,
        const.TORCH_COL_STOPPED_EPOCH: records[-1][const.TORCH_COL_EPOCH],
        const.TORCH_COL_VAL_LOSS: best_loss,
        const.TORCH_COL_N_INNER_TRAIN: int(inner_train_idx.size),
        const.TORCH_COL_N_INNER_VAL: int(inner_val_idx.size),
    }
    logger.info(
        "[%s] fold %d: best epoch %d/%d (val loss %.5f), inner split %d/%d, inputs %d+%d",
        fold.strategy.value,
        fold.index,
        best_epoch,
        summary[const.TORCH_COL_STOPPED_EPOCH],
        best_loss,
        inner_train_idx.size,
        inner_val_idx.size,
        x_tab_train.shape[1],
        x_emb_train.shape[1],
    )
    return (model, tabular_pre, embedding_pre, scaler), pd.DataFrame(records), summary


def predict_rows(
    artifacts: FoldArtifacts,
    frame: pd.DataFrame,
    features: FeatureSet,
    config: DualBranchConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict both targets for ``frame``, returning stability in kcal/mol."""
    model, tabular_pre, embedding_pre, scaler = artifacts
    model.eval()
    tabular = to_tensor(tabular_pre.transform(frame[features.tabular]), config.device)
    embedding = to_tensor(embedding_pre.transform(frame[features.embedding]), config.device)
    with torch.no_grad():
        stability, function = model(tabular, embedding)
    return (
        scaler.inverse_transform(stability.cpu().numpy().astype(float)),
        function.cpu().numpy().astype(float),
    )


# =============================================================================
# Cross-validation
# =============================================================================
def cross_validate_dual_branch(
    df: pd.DataFrame,
    folds: list[Fold],
    features: FeatureSet,
    cfg: dict[str, Any],
    config: DualBranchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, FoldArtifacts]]:
    """Train and score the dual-branch network on every fold.

    Returns:
        ``(oof, history, fold_summary, artifacts)``. ``oof`` uses the same long
        schema as every other model, so the tables concatenate.
    """
    oof_frames, histories, summaries = [], [], []
    artifacts: dict[int, FoldArtifacts] = {}
    for fold in folds:
        fold_artifacts, history, summary = train_one_fold(df, fold, features, cfg, config)
        val_rows = df.iloc[fold.val_idx]
        stability, function = predict_rows(fold_artifacts, val_rows, features, config)

        for target, predicted in ((TARGET_STABILITY, stability), (TARGET_FUNCTION, function)):
            oof_frames.append(
                pd.DataFrame(
                    {
                        const.COL_MUTATION_ID: val_rows[const.COL_MUTATION_ID].to_numpy(),
                        const.COL_PROTEIN_ID: val_rows[const.COL_PROTEIN_ID].to_numpy(),
                        const.COL_PROTEIN_FAMILY: val_rows[const.COL_PROTEIN_FAMILY].to_numpy(),
                        const.CV_COL_STRATEGY: fold.strategy.value,
                        const.COL_MODEL: const.TORCH_V2_MODEL_NAME,
                        const.COL_TARGET: target,
                        const.CV_COL_FOLD: fold.index,
                        const.COL_Y_TRUE: val_rows[target].to_numpy(dtype=float),
                        const.COL_Y_PRED: predicted,
                    }
                )
            )
        histories.append(history)
        summaries.append(summary)
        artifacts[fold.index] = fold_artifacts
    return (
        pd.concat(oof_frames, ignore_index=True),
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(summaries),
        artifacts,
    )


def summarize_metrics(oof: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Pooled out-of-fold metrics per target, plus the per-fold spread."""
    metrics = configured_metrics(cfg)
    rows = []
    for (strategy, target), group in oof.groupby(
        [const.CV_COL_STRATEGY, const.COL_TARGET], sort=False
    ):
        pooled = compute_metrics(
            group[const.COL_Y_TRUE].to_numpy(), group[const.COL_Y_PRED].to_numpy(), metrics
        )
        fold_table = pd.DataFrame(
            [
                compute_metrics(
                    f[const.COL_Y_TRUE].to_numpy(), f[const.COL_Y_PRED].to_numpy(), metrics
                )
                for _, f in group.groupby(const.CV_COL_FOLD, sort=True)
            ]
        )
        rows.append(
            {
                const.CV_COL_STRATEGY: strategy,
                const.COL_MODEL: const.TORCH_V2_MODEL_NAME,
                const.COL_TARGET: target,
                const.COL_N_SAMPLES: len(group),
                **{f"{const.POOLED_PREFIX}{k}": v for k, v in pooled.items()},
                **{
                    f"{const.FOLD_MEAN_PREFIX}{m.value}": float(fold_table[m.value].mean())
                    for m in metrics
                },
                **{
                    f"{const.FOLD_STD_PREFIX}{m.value}": float(fold_table[m.value].std())
                    for m in metrics
                },
            }
        )
        logger.info(
            "[%s|%s|%s] pooled OOF: %s",
            strategy,
            const.TORCH_V2_MODEL_NAME,
            target,
            ", ".join(f"{k}={v:.3f}" for k, v in pooled.items()),
        )
    return pd.DataFrame(rows)


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
    artifacts: dict[int, FoldArtifacts],
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
        for index, (model, tabular_pre, embedding_pre, _) in sorted(artifacts.items()):
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

    example = next(iter(artifacts.values()))[0] if artifacts else None
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

    oof, history, fold_summary, artifacts = cross_validate_dual_branch(
        df, folds, features, cfg, config
    )
    summary = summarize_metrics(oof, cfg)
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
