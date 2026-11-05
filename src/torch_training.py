"""Shared fold-local training machinery for both network architectures.

``src/train_torch.py`` and ``src/train_torch_v2.py`` shared 346 identical
lines before this module existed — the same seeding, the same inner split, the
same target scaler, and above all the same epoch loop with the same early
stopping. Two copies of a training loop is two places for the leakage rules to
drift apart, which is exactly the thing this project cannot afford.

:class:`FoldTrainer` owns that loop once. A subclass supplies only what
actually differs between the architectures:

* :meth:`FoldTrainer.fit_fold_inputs` — how the fold's rows become model
  inputs (one preprocessor for v1, two for v2), and
* :meth:`FoldTrainer.build_model` — the network itself.

Everything else is inherited: seeding per fold, the inner split, target
scaling, the loaders, the epoch loop, early stopping with best-weight restore,
out-of-fold prediction and metric aggregation.

**The unification that makes this work** is that a fold's inputs are always a
*tuple* of tensors. v1 yields one element, v2 yields two, and the loop writes
``model(*inputs)`` without caring which. That is the only reason one loop can
serve both, and it is why :class:`FoldInputs` stores tuples rather than named
fields.

The leakage rules documented in ``src/train_torch.py`` are enforced here, in
one place: preprocessing and the target scaler are fitted on inner-training
rows only, and the outer validation fold is untouched until the final predict.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.common import constants as const
from src.evaluate import compute_metrics, configured_metrics
from src.model import FeatureSet
from src.torch_model import multitask_loss
from src.validation import Fold, group_labels

logger = logging.getLogger(__name__)

# The concrete settings class a trainer carries. Making the trainer generic
# over it lets a subclass use its own config's extra fields (hidden_dims,
# bottleneck_dim) without a cast, while the shared loop still sees only the
# TrainingSettings protocol.
ConfigT = TypeVar("ConfigT", bound="TrainingSettings")
# The concrete network a trainer builds, so a subclass's FoldResult keeps the
# architecture's own type and its provenance code can read architecture-specific
# attributes without a cast.
ModelT = TypeVar("ModelT", bound=nn.Module)


# =============================================================================
# Settings both architectures share
# =============================================================================
class TrainingSettings(Protocol):
    """The optimisation settings :class:`FoldTrainer` needs from a config.

    Declared structurally, and with read-only properties, so that both
    ``TorchConfig`` and ``DualBranchConfig`` satisfy it without either
    inheriting from the other or from this. They are frozen dataclasses, whose
    attributes a settable protocol member would reject.
    """

    @property
    def learning_rate(self) -> float:
        """Adam step size."""

    @property
    def weight_decay(self) -> float:
        """Adam L2 penalty."""

    @property
    def batch_size(self) -> int:
        """Minibatch size."""

    @property
    def max_epochs(self) -> int:
        """Upper bound on epochs."""

    @property
    def inner_val_fraction(self) -> float:
        """Share of the training fold held out for early stopping."""

    @property
    def patience(self) -> int:
        """Epochs without improvement before stopping."""

    @property
    def min_delta(self) -> float:
        """Improvement in validation loss that counts as progress."""

    @property
    def loss_weights(self) -> tuple[float, float]:
        """``(stability, function)``, already rescaled to sum to 1."""

    @property
    def standardize_stability_target(self) -> bool:
        """Whether to z-score the stability target."""

    @property
    def device(self) -> str:
        """Torch device string."""

    @property
    def seed(self) -> int:
        """Project random seed."""


# =============================================================================
# Reproducibility and tensors
# =============================================================================
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch, and ask Torch for deterministic kernels.

    Called once per fold rather than once per run, so that a fold's result
    does not depend on how many folds ran before it. Determinism also requires
    single-process data loading, which :meth:`FoldTrainer.make_loader`
    enforces.
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


def to_tensor(values: np.ndarray, device: str) -> Tensor:
    """Float32 tensor on the target device."""
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)


# =============================================================================
# Inner split for early stopping
# =============================================================================
class InnerSplitSettings(Protocol):
    """The two settings :func:`inner_split` needs from a training config."""

    @property
    def inner_val_fraction(self) -> float:
        """Share of the training fold held out for early stopping."""

    @property
    def seed(self) -> int:
        """Project random seed."""


def inner_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    strategy: const.SplitStrategy,
    config: InnerSplitSettings,
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
        config: Resolved training settings.

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
            logger.debug(
                "Training fold has a single %s group; inner split falls back to random",
                strategy.value,
            )
        fold_groups = None
        splitter = ShuffleSplit(
            n_splits=1, test_size=config.inner_val_fraction, random_state=config.seed
        )

    local_train, local_val = next(splitter.split(train_idx, groups=fold_groups))
    if local_train.size == 0 or local_val.size == 0:
        raise ValueError(
            f"inner_val_fraction={config.inner_val_fraction} leaves an empty side for a "
            f"training fold of {train_idx.size} rows"
        )
    return train_idx[local_train], train_idx[local_val]


# =============================================================================
# Target scaling
# =============================================================================
class TargetScaler:
    """Standardises the stability target, fitted on inner-training rows only.

    ``functional_score`` is left untouched: it already lives on 0-1 and is
    produced by a sigmoid head, so scaling it would break that
    correspondence. Stability is a free-energy change whose scale is arbitrary
    relative to the function target, and leaving it unscaled lets it dominate
    the shared gradients.
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
# What one fold produces
# =============================================================================
@dataclass
class FoldInputs:
    """One fold's fitted preprocessing and the arrays it produced.

    ``train`` and ``val`` are tuples so that a one-branch and a two-branch
    architecture share the same loop: the trainer calls ``model(*inputs)``
    without knowing how many blocks there are.

    Attributes:
        train: Inner-training input blocks, in model argument order.
        val: Inner-validation input blocks, same order.
        transform: Maps a raw frame to the same blocks, for prediction.
        fitted: The fitted preprocessing objects, saved beside the weights.
    """

    train: tuple[np.ndarray, ...]
    val: tuple[np.ndarray, ...]
    transform: Any
    fitted: tuple[Any, ...]

    @property
    def widths(self) -> tuple[int, ...]:
        """Column count of each input block, for sizing the network."""
        return tuple(int(block.shape[1]) for block in self.train)


@dataclass
class FoldResult(Generic[ModelT]):
    """Everything one fold's training produced.

    Attributes:
        model: The network, restored to its best epoch's weights.
        inputs: The fitted preprocessing for this fold.
        scaler: The fitted target scaler.
        history: One row per epoch.
        summary: Early-stopping record for the provenance table.
    """

    model: ModelT
    inputs: FoldInputs
    scaler: TargetScaler
    history: pd.DataFrame
    summary: dict[str, Any]


# =============================================================================
# The shared trainer
# =============================================================================
class FoldTrainer(ABC, Generic[ConfigT, ModelT]):
    """Fold-local training with early stopping, shared by both architectures.

    Subclasses implement :meth:`fit_fold_inputs` and :meth:`build_model`, and
    set :attr:`model_name`. The epoch loop, early stopping, best-weight
    restore, out-of-fold prediction and metric aggregation are inherited.

    Attributes:
        model_name: Name used in every metrics table, so this network lines up
            with the sklearn baselines without a special case.
    """

    model_name: str

    def __init__(
        self,
        cfg: dict[str, Any],
        config: ConfigT,
        features: FeatureSet,
    ) -> None:
        self.cfg = cfg
        self.config = config
        self.features = features

    # -- Hooks a subclass must provide ------------------------------------
    @abstractmethod
    def fit_fold_inputs(
        self, df: pd.DataFrame, inner_train_idx: np.ndarray, inner_val_idx: np.ndarray
    ) -> FoldInputs:
        """Fit this fold's preprocessing and produce the model's input blocks.

        Must fit on ``inner_train_idx`` rows only: the inner-validation rows
        drive early stopping, so they are a fitting input and have to stay
        outside every fitted object.
        """

    @abstractmethod
    def build_model(self, widths: tuple[int, ...]) -> ModelT:
        """Construct the network for input blocks of the given widths."""

    # -- Shared machinery --------------------------------------------------
    def make_loader(
        self, inputs: tuple[np.ndarray, ...], targets: tuple[Tensor, Tensor], *, shuffle: bool
    ) -> DataLoader[tuple[Tensor, ...]]:
        """Wrap input blocks and both targets in a seeded, single-process loader.

        ``num_workers=0`` and an explicit generator keep batch order
        reproducible; multi-process loading would reintroduce nondeterminism
        for no gain at this data size.

        ``drop_last`` applies only to a shuffled loader whose final batch would
        hold one row, because batch normalisation cannot compute a variance
        from a single row and would raise partway through an epoch.
        """
        device = self.config.device
        tensors = [to_tensor(block, device) for block in inputs]
        dataset = TensorDataset(*tensors, *targets)
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        drop_last = shuffle and len(dataset) % self.config.batch_size == 1
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
            drop_last=drop_last,
        )

    def split_batch(self, batch: list[Tensor]) -> tuple[list[Tensor], tuple[Tensor, Tensor]]:
        """Separate a batch into input blocks and the two targets.

        The targets are always the last two tensors, whatever the number of
        input blocks before them.
        """
        return list(batch[:-2]), (batch[-2], batch[-1])

    def evaluate_loss(self, model: ModelT, loader: DataLoader[tuple[Tensor, ...]]) -> float:
        """Mean weighted multitask loss over a loader, with the model in eval mode."""
        model.eval()
        total, n_rows = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                inputs, targets = self.split_batch(list(batch))
                loss, _, _ = multitask_loss(model(*inputs), targets, self.config.loss_weights)
                total += loss.item() * inputs[0].shape[0]
                n_rows += inputs[0].shape[0]
        return total / max(n_rows, 1)

    def fold_targets(self, df: pd.DataFrame, idx: np.ndarray, scaler: TargetScaler) -> tuple[Tensor, Tensor]:
        """Both targets for ``idx``, with stability standardised."""
        stability = scaler.transform(df[const.TARGET_STABILITY].to_numpy(dtype=float)[idx])
        function = df[const.TARGET_FUNCTION].to_numpy(dtype=float)[idx]
        device = self.config.device
        return to_tensor(stability, device), to_tensor(function, device)

    def train_one_fold(self, df: pd.DataFrame, fold: Fold) -> FoldResult[ModelT]:
        """Fit the network on one fold, early-stopping on an inner split.

        Everything learned here — the preprocessing, the target mean/std and
        the weights themselves — is fitted on inner-training rows only.
        """
        config = self.config
        seed_everything(config.seed)
        inner_train_idx, inner_val_idx = inner_split(df, fold.train_idx, fold.strategy, config)

        inputs = self.fit_fold_inputs(df, inner_train_idx, inner_val_idx)
        scaler = TargetScaler(enabled=config.standardize_stability_target).fit(
            df[const.TARGET_STABILITY].to_numpy(dtype=float)[inner_train_idx]
        )
        train_loader = self.make_loader(
            inputs.train, self.fold_targets(df, inner_train_idx, scaler), shuffle=True
        )
        val_loader = self.make_loader(
            inputs.val, self.fold_targets(df, inner_val_idx, scaler), shuffle=False
        )

        model = self.build_model(inputs.widths).to(config.device)
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
            for batch in train_loader:
                batch_inputs, batch_targets = self.split_batch(list(batch))
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = multitask_loss(
                    model(*batch_inputs), batch_targets, config.loss_weights
                )
                loss.backward()
                optimizer.step()
                # Detached before the cast: the running total is a log value,
                # and keeping it attached would retain the whole epoch's graph.
                running += loss.detach().item() * batch_inputs[0].shape[0]
                n_rows += batch_inputs[0].shape[0]
            train_loss = running / max(n_rows, 1)
            val_loss = self.evaluate_loss(model, val_loader)
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
            "[%s] fold %d: best epoch %d/%d (val loss %.5f), inner split %d/%d, inputs %s",
            fold.strategy.value,
            fold.index,
            best_epoch,
            summary[const.TORCH_COL_STOPPED_EPOCH],
            best_loss,
            inner_train_idx.size,
            inner_val_idx.size,
            "+".join(str(width) for width in inputs.widths),
        )
        return FoldResult(model, inputs, scaler, pd.DataFrame(records), summary)

    def predict_rows(
        self, result: FoldResult[ModelT], frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict both targets for ``frame``, returning stability in kcal/mol."""
        result.model.eval()
        blocks = [to_tensor(block, self.config.device) for block in result.inputs.transform(frame)]
        with torch.no_grad():
            stability, function = result.model(*blocks)
        return (
            result.scaler.inverse_transform(stability.cpu().numpy().astype(float)),
            function.cpu().numpy().astype(float),
        )

    def cross_validate(
        self, df: pd.DataFrame, folds: list[Fold]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, FoldResult[ModelT]]]:
        """Train and score the network on every fold.

        Returns:
            ``(oof, history, fold_summary, results)``. ``oof`` uses the same
            long schema as every other model, so the tables concatenate.
        """
        oof_frames, histories, summaries = [], [], []
        results: dict[int, FoldResult[ModelT]] = {}
        for fold in folds:
            result = self.train_one_fold(df, fold)
            val_rows = df.iloc[fold.val_idx]
            stability, function = self.predict_rows(result, val_rows)

            for target, predicted in (
                (const.TARGET_STABILITY, stability),
                (const.TARGET_FUNCTION, function),
            ):
                oof_frames.append(
                    pd.DataFrame(
                        {
                            const.COL_MUTATION_ID: val_rows[const.COL_MUTATION_ID].to_numpy(),
                            const.COL_PROTEIN_ID: val_rows[const.COL_PROTEIN_ID].to_numpy(),
                            const.COL_PROTEIN_FAMILY: val_rows[
                                const.COL_PROTEIN_FAMILY
                            ].to_numpy(),
                            const.CV_COL_STRATEGY: fold.strategy.value,
                            const.COL_MODEL: self.model_name,
                            const.COL_TARGET: target,
                            const.CV_COL_FOLD: fold.index,
                            const.COL_Y_TRUE: val_rows[target].to_numpy(dtype=float),
                            const.COL_Y_PRED: predicted,
                        }
                    )
                )
            histories.append(result.history)
            summaries.append(result.summary)
            results[fold.index] = result
        return (
            pd.concat(oof_frames, ignore_index=True),
            pd.concat(histories, ignore_index=True),
            pd.DataFrame(summaries),
            results,
        )

    def summarize(self, oof: pd.DataFrame) -> pd.DataFrame:
        """Pooled out-of-fold metrics per target, plus the per-fold spread."""
        metrics = configured_metrics(self.cfg)
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
                    const.COL_MODEL: self.model_name,
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
                self.model_name,
                target,
                ", ".join(f"{k}={v:.3f}" for k, v in pooled.items()),
            )
        return pd.DataFrame(rows)
