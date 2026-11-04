"""Multitask neural architecture for joint mutation-effect prediction.

Assignment Task 4. One shared backbone learns a representation of the mutation
from the concatenated tabular features and PCA-reduced protein embedding; two
small heads then read that representation for the two targets::

    tabular (22) + PCA embedding (16)
        │
        ▼
    shared backbone:  [Linear → BatchNorm → activation → Dropout] × len(hidden_dims)
        │
        ├── stability head  → Linear → scalar        (unbounded, kcal/mol)
        └── function head   → Linear → Sigmoid       (bounded to 0-1 by construction)

The motivation for sharing is that both targets are driven by the same physical
question — how much does this substitution perturb its local environment — so
the gradients from the better-determined target (`functional_score`, pooled
R² 0.63) regularise the representation used by the harder one (`stability_ddG`,
R² 0.34). Whether that actually helps on this dataset is an empirical question,
and `src/train_torch.py` answers it against the sklearn baselines rather than
assuming it.

Two architectural choices are worth stating explicitly:

* **The function head ends in a sigmoid.** `functional_score` is a normalised
  retention score in 0-1, so bounding it by construction removes a class of
  nonsense prediction rather than hoping the loss discourages it. The cost is
  that the head can never output exactly 0 or 1; two labelled rows sit exactly
  at 1.0, so a small irreducible error on those is accepted deliberately.
* **The stability head is unbounded.** `stability_ddG` is a free-energy change
  with no natural bound, and clamping it would distort the tail that matters
  most for ranking.

This module holds the architecture only — no I/O, no optimiser, no fold logic.
That separation keeps it unit-testable without touching disk, and mirrors the
split between `src/model.py` and `src/train.py` on the sklearn side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.common import constants as const

# Activation name -> factory. Extending the enum is the only change needed to
# add another nonlinearity.
ACTIVATIONS: dict[const.Activation, type[nn.Module]] = {
    const.Activation.RELU: nn.ReLU,
    const.Activation.GELU: nn.GELU,
}


@dataclass(frozen=True)
class TorchConfig:
    """Resolved architecture and optimisation settings for one training run.

    Built by :func:`torch_config_from` so that every value is read from
    ``config.yaml`` exactly once, and the rest of the code can depend on a
    typed object rather than on nested dictionary lookups.

    Attributes:
        hidden_dims: Width of each shared-backbone hidden layer, in order.
        head_hidden_dim: Width of each task head's hidden layer, or ``None``
            for a single linear projection.
        activation: Nonlinearity between hidden layers.
        batch_norm: Whether to apply batch normalisation in the backbone.
        dropout: Dropout probability applied after each backbone activation.
        learning_rate: Adam step size.
        weight_decay: Adam L2 penalty.
        batch_size: Minibatch size.
        max_epochs: Upper bound on epochs; early stopping usually ends sooner.
        inner_val_fraction: Share of the *training fold* held out to early-stop on.
        patience: Epochs without improvement before stopping.
        min_delta: Improvement in validation loss that counts as progress.
        loss_weights: ``(stability, function)``, rescaled to sum to 1.
        standardize_stability_target: Whether to z-score the stability target.
        device: Torch device string.
        seed: Project random seed.
    """

    hidden_dims: tuple[int, ...]
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
    device: str
    seed: int


def resolve_device(requested: str) -> str:
    """Resolve the configured device, falling back when an accelerator is absent.

    ``auto`` deliberately resolves to CPU rather than to the fastest available
    backend: MPS and CUDA kernels are not bit-reproducible, and this project
    treats reproducibility as the stronger requirement. An explicitly requested
    accelerator is honoured when present, and warned about and downgraded when
    it is not, so a config written on one machine still runs on another.
    """
    device = const.TorchDevice(requested)
    if device is const.TorchDevice.AUTO or device is const.TorchDevice.CPU:
        return const.TorchDevice.CPU.value
    available = {
        const.TorchDevice.CUDA: torch.cuda.is_available(),
        const.TorchDevice.MPS: torch.backends.mps.is_available(),
    }
    return device.value if available[device] else const.TorchDevice.CPU.value


def torch_config_from(cfg: dict[str, Any]) -> TorchConfig:
    """Read the ``torch`` config section into a typed :class:`TorchConfig`.

    Loss weights are rescaled to sum to 1 so that changing their relative size
    does not also change the effective learning rate.

    Raises:
        ValueError: If both loss weights are zero, which would leave no objective.
    """
    section = cfg["torch"]
    weights = (section["loss_weights"]["stability"], section["loss_weights"]["function"])
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("torch.loss_weights must not both be zero: there would be nothing to optimise")

    stopping = section["early_stopping"]
    return TorchConfig(
        hidden_dims=tuple(section["hidden_dims"]),
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
        device=resolve_device(section["device"]),
        seed=int(cfg["project"]["random_seed"]),
    )


class ProteinMutationMultitaskNN(nn.Module):
    """Shared-backbone network predicting stability and function together.

    Args:
        n_features: Width of the input vector, i.e. the number of columns the
            fold-local preprocessor emits (tabular + PCA components).
        config: Resolved architecture settings.

    Raises:
        ValueError: If ``n_features`` is not positive.
    """

    def __init__(self, n_features: int, config: TorchConfig) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError(f"n_features must be positive, got {n_features}")
        self.n_features = n_features
        self.config = config

        activation = ACTIVATIONS[config.activation]
        layers: list[nn.Module] = []
        width = n_features
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(width, hidden))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(activation())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            width = hidden
        self.backbone = nn.Sequential(*layers)
        self.representation_dim = width

        self.stability_head = self._build_head(width, activation, bounded=False)
        self.function_head = self._build_head(width, activation, bounded=True)

    def _build_head(self, width: int, activation: type[nn.Module], *, bounded: bool) -> nn.Sequential:
        """Build one task head.

        ``bounded`` appends a sigmoid, which is correct only for a target that
        genuinely lives in 0-1 (``functional_score``), never for ``stability_ddG``.
        """
        layers: list[nn.Module] = []
        if self.config.head_hidden_dim is not None:
            layers += [nn.Linear(width, self.config.head_hidden_dim), activation()]
            width = self.config.head_hidden_dim
        layers.append(nn.Linear(width, 1))
        if bounded:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict both targets from one batch of features.

        Args:
            x: Float tensor of shape ``(batch, n_features)``.

        Returns:
            ``(stability, function)``, each of shape ``(batch,)``. Stability is
            unbounded; function is in the open interval (0, 1).
        """
        shared = self.backbone(x)
        return self.stability_head(shared).squeeze(-1), self.function_head(shared).squeeze(-1)


def multitask_loss(
    predictions: tuple[Tensor, Tensor],
    targets: tuple[Tensor, Tensor],
    weights: tuple[float, float],
) -> tuple[Tensor, Tensor, Tensor]:
    """Weighted sum of the two per-task mean squared errors.

    ``loss = w_stability * MSE(stability) + w_function * MSE(function)``, with
    the weights already rescaled to sum to 1 by :func:`torch_config_from`.

    Both per-task terms are returned as well as the total, because their ratio
    is what shows whether one task is dominating the shared backbone — the
    usual failure mode of naive multitask training, and the reason the two
    targets are scaled onto comparable ranges before this is called.

    Returns:
        ``(total, stability_mse, function_mse)``.
    """
    stability_mse = nn.functional.mse_loss(predictions[0], targets[0])
    function_mse = nn.functional.mse_loss(predictions[1], targets[1])
    total = weights[0] * stability_mse + weights[1] * function_mse
    return total, stability_mse, function_mse


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters, for the provenance record."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
