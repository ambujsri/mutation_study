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
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table, write_table
from src.evaluate import compute_metrics, configured_metrics
from src.model import FeatureSet, build_preprocessor
from src.torch_model import (
    ProteinMutationMultitaskNN,
    TorchConfig,
    count_parameters,
    multitask_loss,
    torch_config_from,
)
from src.train import load_training_data
from src.validation import Fold, generate_folds, group_labels

logger = logging.getLogger(__name__)

TARGET_STABILITY = const.TARGET_STABILITY
TARGET_FUNCTION = const.TARGET_FUNCTION


# =============================================================================
# Reproducibility
# =============================================================================
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch, and ask Torch for deterministic kernels.

    Called once per fold rather than once per run, so that a fold's result does
    not depend on how many folds ran before it. Determinism also requires
    single-process data loading, which :func:`make_loader` enforces.
    """
    random.seed(seed)
    # Legacy global seed on purpose: every splitter here is given an explicit
    # random_state, but a third-party call that reaches for the global NumPy
    # RNG would otherwise be the one unseeded thing in the run.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def make_loader(x: Tensor, y_stability: Tensor, y_function: Tensor, config: TorchConfig, *,
                shuffle: bool) -> DataLoader[tuple[Tensor, ...]]:
    """Wrap tensors in a seeded, single-process ``DataLoader``.

    ``num_workers=0`` and an explicit generator keep batch order reproducible;
    multi-process loading would reintroduce nondeterminism for no gain at this
    data size (840 rows).
    """
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        TensorDataset(x, y_stability, y_function),
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        drop_last=False,
    )


# =============================================================================
# Inner split for early stopping
# =============================================================================
def inner_split(
    df: pd.DataFrame, train_idx: np.ndarray, strategy: const.SplitStrategy, config: TorchConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Split the training fold into inner-train and inner-validation rows.

    The split is grouped by the same column as the outer strategy whenever the
    training fold holds more than one group, so the stopping epoch is chosen
    under the same generalisation regime the outer metric reports. It degrades
    to a plain random split only when grouping is impossible (a single group,
    or a random outer strategy).

    Args:
        df: Full labelled table.
        train_idx: Positional indices of the outer training fold.
        strategy: Outer splitting strategy.
        config: Resolved torch settings.

    Returns:
        ``(inner_train_idx, inner_val_idx)`` as positional indices into ``df``.

    Raises:
        ValueError: If the fraction leaves either side empty.
    """
    groups = group_labels(df, strategy)
    fold_groups = None if groups is None else groups.to_numpy()[train_idx]
    splitter: GroupShuffleSplit | ShuffleSplit
    if fold_groups is not None and pd.unique(fold_groups).size > 1:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=config.inner_val_fraction, random_state=config.seed
        )
    else:
        if fold_groups is not None:
            logger.debug("Training fold has a single %s group; inner split falls back to random",
                         strategy.value)
        fold_groups = None
        splitter = ShuffleSplit(n_splits=1, test_size=config.inner_val_fraction,
                                random_state=config.seed)

    local_train, local_val = next(splitter.split(train_idx, groups=fold_groups))
    if local_train.size == 0 or local_val.size == 0:
        raise ValueError(
            f"torch.early_stopping.inner_val_fraction={config.inner_val_fraction} leaves an empty "
            f"side for a training fold of {train_idx.size} rows"
        )
    return train_idx[local_train], train_idx[local_val]


# =============================================================================
# Target scaling
# =============================================================================
class TargetScaler:
    """Standardises the stability target, fitted on inner-training rows only.

    ``functional_score`` is left untouched: it already lives on 0-1 and is
    produced by a sigmoid head, so scaling it would break that correspondence.
    Stability is a free-energy change whose scale is arbitrary relative to the
    function target, and leaving it unscaled lets it dominate the shared
    backbone's gradients.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.mean = 0.0
        self.std = 1.0

    def fit(self, values: np.ndarray) -> TargetScaler:
        """Learn the mean and standard deviation (no-op when disabled)."""
        if self.enabled:
            self.mean = float(values.mean())
            std = float(values.std(ddof=0))
            # A constant target would otherwise divide by zero; keep it as-is.
            self.std = std if std > 0 else 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the learned standardisation."""
        return (values - self.mean) / self.std if self.enabled else values

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Map predictions back to kcal/mol."""
        return values * self.std + self.mean if self.enabled else values


# =============================================================================
# Training one fold
# =============================================================================
def _to_tensor(values: np.ndarray, device: str) -> Tensor:
    """Float32 tensor on the target device."""
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)


def _evaluate_loss(
    model: ProteinMutationMultitaskNN, loader: DataLoader[tuple[Tensor, ...]], config: TorchConfig
) -> float:
    """Mean weighted multitask loss over a loader, with the model in eval mode."""
    model.eval()
    total, n_rows = 0.0, 0
    with torch.no_grad():
        for features, y_stability, y_function in loader:
            loss, _, _ = multitask_loss(model(features), (y_stability, y_function), config.loss_weights)
            total += float(loss) * features.shape[0]
            n_rows += features.shape[0]
    return total / max(n_rows, 1)


def train_one_fold(
    df: pd.DataFrame,
    fold: Fold,
    features: FeatureSet,
    cfg: dict[str, Any],
    config: TorchConfig,
) -> tuple[ProteinMutationMultitaskNN, ColumnTransformer, TargetScaler, pd.DataFrame, dict[str, Any]]:
    """Fit the network on one fold, early-stopping on an inner split.

    Everything learned here — imputer, scaler, PCA, target mean/std and the
    weights themselves — is fitted on training-fold rows only.

    Returns:
        ``(model, preprocessor, target_scaler, history, summary)`` where
        ``history`` has one row per epoch and ``summary`` records the stopping
        behaviour for the provenance table.
    """
    seed_everything(config.seed)
    inner_train_idx, inner_val_idx = inner_split(df, fold.train_idx, fold.strategy, config)

    # Preprocessing is fitted on the inner-training rows: the inner-validation
    # rows drive early stopping, so they must stay outside every fitted object.
    preprocessor = build_preprocessor(cfg, features)
    x_all = df[features.all]
    x_inner_train = preprocessor.fit_transform(x_all.iloc[inner_train_idx])
    x_inner_val = preprocessor.transform(x_all.iloc[inner_val_idx])

    scaler = TargetScaler(enabled=config.standardize_stability_target)
    scaler.fit(df[TARGET_STABILITY].to_numpy(dtype=float)[inner_train_idx])

    def targets(idx: np.ndarray) -> tuple[Tensor, Tensor]:
        stability = scaler.transform(df[TARGET_STABILITY].to_numpy(dtype=float)[idx])
        function = df[TARGET_FUNCTION].to_numpy(dtype=float)[idx]
        return _to_tensor(stability, config.device), _to_tensor(function, config.device)

    train_loader = make_loader(
        _to_tensor(x_inner_train, config.device), *targets(inner_train_idx), config, shuffle=True
    )
    val_loader = make_loader(
        _to_tensor(x_inner_val, config.device), *targets(inner_val_idx), config, shuffle=False
    )

    model = ProteinMutationMultitaskNN(x_inner_train.shape[1], config).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_loss, best_epoch, best_state = float("inf"), 0, {
        k: v.detach().clone() for k, v in model.state_dict().items()
    }
    records, epochs_without_gain = [], 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running, n_rows = 0.0, 0
        for batch_x, batch_stability, batch_function in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = multitask_loss(
                model(batch_x), (batch_stability, batch_function), config.loss_weights
            )
            loss.backward()
            optimizer.step()
            running += float(loss) * batch_x.shape[0]
            n_rows += batch_x.shape[0]
        train_loss = running / max(n_rows, 1)
        val_loss = _evaluate_loss(model, val_loader, config)
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
        "[%s] fold %d: best epoch %d/%d (val loss %.5f), inner split %d/%d",
        fold.strategy.value,
        fold.index,
        best_epoch,
        summary[const.TORCH_COL_STOPPED_EPOCH],
        best_loss,
        inner_train_idx.size,
        inner_val_idx.size,
    )
    return model, preprocessor, scaler, pd.DataFrame(records), summary


def predict_rows(
    model: ProteinMutationMultitaskNN,
    preprocessor: ColumnTransformer,
    scaler: TargetScaler,
    frame: pd.DataFrame,
    features: FeatureSet,
    config: TorchConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict both targets for ``frame``, returning stability in kcal/mol."""
    model.eval()
    x = _to_tensor(preprocessor.transform(frame[features.all]), config.device)
    with torch.no_grad():
        stability, function = model(x)
    return (
        scaler.inverse_transform(stability.cpu().numpy().astype(float)),
        function.cpu().numpy().astype(float),
    )


# =============================================================================
# Cross-validation
# =============================================================================
def cross_validate_torch(
    df: pd.DataFrame, folds: list[Fold], features: FeatureSet, cfg: dict[str, Any], config: TorchConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, tuple[Any, Any, Any]]]:
    """Train and score the network on every fold.

    Returns:
        ``(oof, history, fold_summary, artifacts)``. ``oof`` uses the same long
        schema as the sklearn baselines, so the two can be concatenated.
        ``artifacts`` maps fold index to ``(model, preprocessor, scaler)``.
    """
    oof_frames, histories, summaries = [], [], []
    artifacts: dict[int, tuple[Any, Any, Any]] = {}
    for fold in folds:
        model, preprocessor, scaler, history, summary = train_one_fold(df, fold, features, cfg, config)
        val_rows = df.iloc[fold.val_idx]
        stability, function = predict_rows(model, preprocessor, scaler, val_rows, features, config)

        for target, predicted in ((TARGET_STABILITY, stability), (TARGET_FUNCTION, function)):
            oof_frames.append(
                pd.DataFrame(
                    {
                        const.COL_MUTATION_ID: val_rows[const.COL_MUTATION_ID].to_numpy(),
                        const.COL_PROTEIN_ID: val_rows[const.COL_PROTEIN_ID].to_numpy(),
                        const.COL_PROTEIN_FAMILY: val_rows[const.COL_PROTEIN_FAMILY].to_numpy(),
                        const.CV_COL_STRATEGY: fold.strategy.value,
                        const.COL_MODEL: const.TORCH_MODEL_NAME,
                        const.COL_TARGET: target,
                        const.CV_COL_FOLD: fold.index,
                        const.COL_Y_TRUE: val_rows[target].to_numpy(dtype=float),
                        const.COL_Y_PRED: predicted,
                    }
                )
            )
        histories.append(history)
        summaries.append(summary)
        artifacts[fold.index] = (model, preprocessor, scaler)
    return (
        pd.concat(oof_frames, ignore_index=True),
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(summaries),
        artifacts,
    )


def summarize_torch_metrics(oof: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Pooled out-of-fold metrics per target, plus the per-fold spread."""
    metrics = configured_metrics(cfg)
    rows = []
    for (strategy, target), group in oof.groupby(
        [const.CV_COL_STRATEGY, const.COL_TARGET], sort=False
    ):
        pooled = compute_metrics(
            group[const.COL_Y_TRUE].to_numpy(), group[const.COL_Y_PRED].to_numpy(), metrics
        )
        per_fold = [
            compute_metrics(f[const.COL_Y_TRUE].to_numpy(), f[const.COL_Y_PRED].to_numpy(), metrics)
            for _, f in group.groupby(const.CV_COL_FOLD, sort=True)
        ]
        fold_table = pd.DataFrame(per_fold)
        rows.append(
            {
                const.CV_COL_STRATEGY: strategy,
                const.COL_MODEL: const.TORCH_MODEL_NAME,
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
            const.TORCH_MODEL_NAME,
            target,
            ", ".join(f"{k}={v:.3f}" for k, v in pooled.items()),
        )
    return pd.DataFrame(rows)


def compare_with_baselines(torch_summary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Place the network beside the sklearn baselines on the same folds.

    Reads the baseline metrics `src/train.py` saved rather than refitting, so
    the comparison cannot disagree with the published baseline numbers.

    Returns:
        One row per target per model under the evaluated strategy, with the
        network's signed difference against the **best** baseline for that
        target. An empty frame if the baselines have not been run.
    """
    path = resolve_path(cfg, "metrics_dir", "cv_metrics_summary")
    if not path.is_file():
        logger.warning("Baseline metrics not found at %s; skipping comparison", path)
        return pd.DataFrame()

    baselines = read_table(path)
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
        baseline_rows = target_rows[target_rows[const.COL_MODEL] != const.TORCH_MODEL_NAME]
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
            const.TORCH_MODEL_NAME,
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
    artifacts: dict[int, tuple[Any, Any, Any]],
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
        for fold_index, (model, preprocessor, scaler) in sorted(artifacts.items()):
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

    example = next(iter(artifacts.values()))[0] if artifacts else None  # noqa: SIM108
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
            "optimizer": "adam",
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

    oof, history, fold_summary, artifacts = cross_validate_torch(df, folds, features, cfg, config)
    summary = summarize_torch_metrics(oof, cfg)
    comparison = compare_with_baselines(summary, cfg)
    save_artifacts(
        oof, summary, history, fold_summary, comparison, artifacts, features, cfg, config
    )
    logger.info("Task 4 complete: %d folds, %d out-of-fold predictions", len(folds), len(oof))
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
