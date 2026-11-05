"""Dual-branch disentangled architecture for joint mutation-effect prediction.

The second Task 4 architecture. Where :mod:`src.torch_model` concatenates the
tabular block and the reduced embedding into one input vector, this one keeps
them apart until after each has been processed on its own terms::

    tabular (22: deltas, conservation, accessibility, structure)
        │
        ├─ tabular branch:  [Linear → BatchNorm → act → Dropout] × n
        │                                                            │
    embedding (256, protein-level)                                   │
        │                                                            │
        ├─ embedding branch: Linear → BatchNorm → act → Dropout      │
        │                    → Linear(bottleneck_dim) → act          │
        │                                                            │
        └──────────────── concat ─────────────────────────────────────┘
                              │
                     fusion trunk: Linear(hidden_dim)
                              │
                     [ResidualBlock] × n_residual_blocks
                              │
            ┌─────────────────┴─────────────────┐
     stability head → scalar          function head → Sigmoid
     (unbounded, kcal/mol)            (bounded to 0-1 by construction)

**Why separate the branches.** The two feature blocks differ in kind, not just
in name. The tabular columns vary from mutation to mutation within a protein —
they are the only thing in the input that distinguishes one substitution of a
given protein from another. The 256-d embedding is a *protein-level* vector:
every mutation of the same protein carries an identical copy. The ablation in
``results/metrics/ablation_summary.md`` measures what that is worth under the
family-grouped split: embeddings on their own score a **negative** rank
correlation (-0.205 on stability, -0.110 on function), and adding all 256 of
them on top of the full tabular set moves Spearman by +0.010 and +0.025. They
are close to inert, and 256 inert columns against 22 informative ones is a
poor ratio to hand a fully-connected first layer, where every embedding
dimension gets its own weights into every hidden unit.

Three architectural consequences follow:

* **The bottleneck is learned, not fixed.** ``torch_v2.embedding_pca: false``
  turns off the fold-local PCA and lets the branch learn its own reduction to
  ``bottleneck_dim``. PCA maximises retained variance, which on a protein-level
  vector is mostly variance *between families* — exactly the component that
  cannot generalise to a held-out family. A supervised bottleneck is free to
  keep a different, smaller subspace.
* **The branches are sized by information content, not by input width.** The
  embedding's 256 columns are squeezed to 32 before fusion while the tabular
  22 are widened to 64, so the fusion trunk sees roughly comparable
  contributions from each.
* **Residual blocks carry the fused signal forward.** With a skip connection
  the trunk starts from the identity, so added depth cannot destroy the
  branch representations it is given; it can only refine them.

Whether any of this beats the baselines is an empirical question that
:mod:`src.train_torch_v2` answers on the same folds, not an assumption.

This module holds the architecture only: no I/O, no optimiser, no fold logic,
mirroring the split between :mod:`src.model` and :mod:`src.train`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.common import constants as const
from src.torch_model import ACTIVATIONS, resolve_device


@dataclass(frozen=True)
class DualBranchConfig:
    """Resolved architecture and optimisation settings for one v2 training run.

    Built by :func:`dual_branch_config_from` so every value is read from
    ``config.yaml`` exactly once.

    Attributes:
        tabular_hidden_dims: Widths of the tabular branch's hidden layers.
        embedding_hidden_dims: Widths of the embedding branch's hidden layers,
            applied before the bottleneck.
        bottleneck_dim: Width the embedding is squeezed to before fusion.
        embedding_pca: Whether to also keep the fold-local PCA. ``False`` sends
            the raw embedding block into the learned bottleneck instead.
        fusion_hidden_dim: Width of the fusion trunk and of every residual block.
        n_residual_blocks: How many residual blocks follow the fusion layer.
        head_hidden_dim: Per-head hidden width, or ``None`` for a single Linear.
        activation: Nonlinearity used throughout.
        batch_norm: Whether to apply batch normalisation.
        dropout: Dropout probability after each activation.
        learning_rate: Adam step size.
        weight_decay: Adam L2 penalty.
        batch_size: Minibatch size.
        max_epochs: Upper bound on epochs; early stopping usually ends sooner.
        inner_val_fraction: Share of the *training fold* held out to stop on.
        patience: Epochs without improvement before stopping.
        min_delta: Improvement in validation loss that counts as progress.
        loss_weights: ``(stability, function)``, rescaled to sum to 1.
        standardize_stability_target: Whether to z-score the stability target.
        save_fold_models: Whether to write each fold's weights to disk.
        device: Torch device string.
        seed: Project random seed.
    """

    tabular_hidden_dims: tuple[int, ...]
    embedding_hidden_dims: tuple[int, ...]
    bottleneck_dim: int
    embedding_pca: bool
    fusion_hidden_dim: int
    n_residual_blocks: int
    head_hidden_dim: int | None
    activation: const.Activation
    batch_norm: bool
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    inner_val_fraction: float
    patience: int
    min_delta: float
    loss_weights: tuple[float, float]
    standardize_stability_target: bool
    save_fold_models: bool
    device: str
    seed: int


def dual_branch_config_from(cfg: dict[str, Any]) -> DualBranchConfig:
    """Read the ``torch_v2`` config section into a typed :class:`DualBranchConfig`.

    Loss weights are rescaled to sum to 1 so that changing their relative size
    does not also change the effective learning rate.

    Raises:
        ValueError: If both loss weights are zero, leaving no objective.
    """
    section = cfg["torch_v2"]
    weights = (section["loss_weights"]["stability"], section["loss_weights"]["function"])
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(
            "torch_v2.loss_weights must not both be zero: there would be nothing to optimise"
        )

    stopping = section["early_stopping"]
    return DualBranchConfig(
        tabular_hidden_dims=tuple(section["tabular_branch"]["hidden_dims"]),
        embedding_hidden_dims=tuple(section["embedding_branch"]["hidden_dims"]),
        bottleneck_dim=int(section["embedding_branch"]["bottleneck_dim"]),
        embedding_pca=bool(section["embedding_pca"]),
        fusion_hidden_dim=int(section["fusion"]["hidden_dim"]),
        n_residual_blocks=int(section["fusion"]["n_residual_blocks"]),
        head_hidden_dim=section["head_hidden_dim"],
        activation=const.Activation(section["activation"]),
        batch_norm=bool(section["batch_norm"]),
        dropout=float(section["dropout"]),
        learning_rate=float(section["optimizer"]["learning_rate"]),
        weight_decay=float(section["optimizer"]["weight_decay"]),
        batch_size=int(section["batch_size"]),
        max_epochs=int(section["max_epochs"]),
        inner_val_fraction=float(stopping["inner_val_fraction"]),
        patience=int(stopping["patience"]),
        min_delta=float(stopping["min_delta"]),
        loss_weights=(weights[0] / total, weights[1] / total),
        standardize_stability_target=bool(section["standardize_stability_target"]),
        save_fold_models=bool(section["save_fold_models"]),
        device=resolve_device(section["device"]),
        seed=int(cfg["project"]["random_seed"]),
    )


def dense_block(
    in_features: int,
    out_features: int,
    activation: type[nn.Module],
    *,
    batch_norm: bool,
    dropout: float,
) -> list[nn.Module]:
    """One ``Linear → (BatchNorm) → activation → (Dropout)`` group.

    Returned as a list rather than a module so callers can splice several into
    a single :class:`torch.nn.Sequential` without nesting.
    """
    layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_features))
    layers.append(activation())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


class ResidualBlock(nn.Module):
    """Two-layer block whose output is added back to its input.

    ``y = act(x + f(x))`` with ``f`` a ``Linear → norm → act → dropout →
    Linear → norm`` stack of constant width. Because ``f`` starts near zero
    under default initialisation, the block starts close to the identity: extra
    depth can refine the fused representation but cannot destroy it, which is
    the property that makes depth safe on a dataset of 840 rows.

    The activation is applied *after* the addition, in the original
    post-activation form, so the skip path itself stays linear.

    Args:
        width: Input and output width; a residual block cannot change shape.
        activation: Nonlinearity class.
        batch_norm: Whether to normalise inside the block.
        dropout: Dropout probability between the two linear layers.
    """

    def __init__(
        self, width: int, activation: type[nn.Module], *, batch_norm: bool, dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(width, width)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(width, width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        self.block = nn.Sequential(*layers)
        self.activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        """Return ``activation(x + block(x))``."""
        return self.activation(x + self.block(x))


class ProteinMutationDualBranchNN(nn.Module):
    """Dual-branch network predicting stability and function together.

    Args:
        n_tabular: Number of columns the tabular preprocessor emits.
        n_embedding: Number of columns the embedding preprocessor emits.
        config: Resolved architecture settings.

    Raises:
        ValueError: If either input width is not positive. Both branches are
            required: with no tabular features there is nothing that varies
            between mutations of one protein, and with no embedding there is
            no second branch to disentangle.
    """

    def __init__(self, n_tabular: int, n_embedding: int, config: DualBranchConfig) -> None:
        super().__init__()
        if n_tabular < 1 or n_embedding < 1:
            raise ValueError(
                "ProteinMutationDualBranchNN needs both blocks, got "
                f"n_tabular={n_tabular}, n_embedding={n_embedding}"
            )
        self.n_tabular = n_tabular
        self.n_embedding = n_embedding
        self.config = config
        activation = ACTIVATIONS[config.activation]

        def block(in_features: int, out_features: int) -> list[nn.Module]:
            return dense_block(
                in_features,
                out_features,
                activation,
                batch_norm=config.batch_norm,
                dropout=config.dropout,
            )

        tabular_layers: list[nn.Module] = []
        width = n_tabular
        for hidden in config.tabular_hidden_dims:
            tabular_layers += block(width, hidden)
            width = hidden
        self.tabular_branch = nn.Sequential(*tabular_layers)
        self.tabular_out_dim = width

        # The embedding branch ends in a plain Linear → activation rather than
        # another dense_block: the bottleneck is the representation handed to
        # fusion, and dropping units from a 32-d summary of 256 inputs would
        # discard a large share of it on every step.
        embedding_layers: list[nn.Module] = []
        width = n_embedding
        for hidden in config.embedding_hidden_dims:
            embedding_layers += block(width, hidden)
            width = hidden
        embedding_layers += [nn.Linear(width, config.bottleneck_dim), activation()]
        self.embedding_branch = nn.Sequential(*embedding_layers)
        self.embedding_out_dim = config.bottleneck_dim

        fused_dim = self.tabular_out_dim + self.embedding_out_dim
        self.fusion = nn.Sequential(*block(fused_dim, config.fusion_hidden_dim))
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    config.fusion_hidden_dim,
                    activation,
                    batch_norm=config.batch_norm,
                    dropout=config.dropout,
                )
                for _ in range(config.n_residual_blocks)
            ]
        )
        self.representation_dim = config.fusion_hidden_dim

        self.stability_head = self._build_head(activation, bounded=False)
        self.function_head = self._build_head(activation, bounded=True)

    def _build_head(self, activation: type[nn.Module], *, bounded: bool) -> nn.Sequential:
        """Build one task head.

        ``bounded`` appends a sigmoid, correct only for a target that genuinely
        lives on 0-1 (``functional_score``), never for ``stability_ddG``.
        """
        layers: list[nn.Module] = []
        width = self.representation_dim
        if self.config.head_hidden_dim is not None:
            layers += [nn.Linear(width, self.config.head_hidden_dim), activation()]
            width = self.config.head_hidden_dim
        layers.append(nn.Linear(width, 1))
        if bounded:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def encode(self, tabular: Tensor, embedding: Tensor) -> Tensor:
        """Return the fused representation both heads read.

        Exposed separately from :meth:`forward` so that the representation can
        be inspected — for instance to check whether held-out families land
        somewhere the training families never occupied.
        """
        branches = [self.tabular_branch(tabular), self.embedding_branch(embedding)]
        return self.residual_blocks(self.fusion(torch.cat(branches, dim=1)))

    def forward(self, tabular: Tensor, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Predict both targets from one batch.

        Args:
            tabular: Float tensor of shape ``(batch, n_tabular)``.
            embedding: Float tensor of shape ``(batch, n_embedding)``.

        Returns:
            ``(stability, function)``, each of shape ``(batch,)``. Stability is
            unbounded; function is in the open interval (0, 1).
        """
        shared = self.encode(tabular, embedding)
        return self.stability_head(shared).squeeze(-1), self.function_head(shared).squeeze(-1)


def count_branch_parameters(model: ProteinMutationDualBranchNN) -> dict[str, int]:
    """Trainable parameters per branch, for the provenance record.

    The split is the point: it shows how much capacity is being spent on the
    protein-level embedding relative to the mutation-level tabular features,
    which is the trade-off this architecture exists to control.
    """

    def total(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    return {
        const.TORCH_V2_BRANCH_TABULAR: total(model.tabular_branch),
        const.TORCH_V2_BRANCH_EMBEDDING: total(model.embedding_branch),
        const.TORCH_V2_BRANCH_FUSION: total(model.fusion) + total(model.residual_blocks),
        const.TORCH_V2_BRANCH_HEADS: total(model.stability_head) + total(model.function_head),
    }
