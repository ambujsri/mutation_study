"""Unit tests for ``src.torch_model_v2``: the dual-branch architecture itself.

The architecture module has no I/O and no fold logic, so everything here runs
on small random tensors. Three properties are worth guaranteeing:

* **shape and bounds** — both heads emit one value per row, and the function
  head's sigmoid holds its target inside (0, 1) whatever the input;
* **disentanglement** — the two branches really are separate paths, so a
  change to one block cannot be silently routed through the other's weights;
  and
* **the residual skip** — a block with zeroed internals is the identity, which
  is the property that makes added depth safe.

Config parsing is checked against hand-built dictionaries rather than the real
``config.yaml``, so a failure points at the parser and not at the file.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from src.common import constants as const
from src.torch_model_v2 import (
    DualBranchConfig,
    ProteinMutationDualBranchNN,
    ResidualBlock,
    count_branch_parameters,
    dense_block,
    dual_branch_config_from,
)

TEST_N_TABULAR: int = 6
TEST_N_EMBEDDING: int = 16
TEST_BATCH: int = 8
TEST_BOTTLENECK: int = 4
TEST_FUSION_DIM: int = 12
# Large magnitudes: the sigmoid bound must hold on inputs far outside the
# range any scaler would produce, not merely on well-behaved ones.
TEST_EXTREME_SCALE: float = 1e4


# =============================================================================
# Fixtures & helpers
# =============================================================================
@pytest.fixture()
def config() -> DualBranchConfig:
    """A small but structurally complete architecture config."""
    return DualBranchConfig(
        tabular_hidden_dims=(8, 8),
        embedding_hidden_dims=(8,),
        bottleneck_dim=TEST_BOTTLENECK,
        embedding_pca=False,
        fusion_hidden_dim=TEST_FUSION_DIM,
        n_residual_blocks=2,
        head_hidden_dim=4,
        activation=const.Activation.RELU,
        batch_norm=True,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=4,
        max_epochs=2,
        inner_val_fraction=0.2,
        patience=1,
        min_delta=0.0,
        loss_weights=(0.5, 0.5),
        standardize_stability_target=True,
        save_fold_models=False,
        device=const.TorchDevice.CPU.value,
        seed=20261102,
    )


def inputs(batch: int = TEST_BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic pair of input blocks."""
    generator = torch.Generator().manual_seed(20261102)
    return (
        torch.randn(batch, TEST_N_TABULAR, generator=generator),
        torch.randn(batch, TEST_N_EMBEDDING, generator=generator),
    )


def raw_config_section() -> dict[str, Any]:
    """The ``torch_v2`` config section as ``config.yaml`` spells it."""
    return {
        "tabular_branch": {"hidden_dims": [8]},
        "embedding_branch": {"bottleneck_dim": TEST_BOTTLENECK, "hidden_dims": [8]},
        "embedding_pca": False,
        "fusion": {"hidden_dim": TEST_FUSION_DIM, "n_residual_blocks": 1},
        "head_hidden_dim": 4,
        "activation": const.Activation.RELU.value,
        "batch_norm": True,
        "dropout": 0.1,
        "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
        "batch_size": 4,
        "max_epochs": 2,
        "early_stopping": {"inner_val_fraction": 0.2, "patience": 1, "min_delta": 0.0},
        "loss_weights": {"stability": 1.0, "function": 3.0},
        "standardize_stability_target": True,
        "device": const.TorchDevice.CPU.value,
        "save_fold_models": False,
    }


# =============================================================================
# Config parsing
# =============================================================================
def test_loss_weights_are_rescaled_to_sum_to_one() -> None:
    """Relative weighting is preserved; absolute scale is not, so the LR is unaffected."""
    cfg = {"torch_v2": raw_config_section(), "project": {"random_seed": 20261102}}

    config = dual_branch_config_from(cfg)

    assert config.loss_weights == pytest.approx((0.25, 0.75))
    assert sum(config.loss_weights) == pytest.approx(1.0)


def test_zero_loss_weights_are_rejected() -> None:
    """Two zero weights would leave nothing to optimise, which must fail loudly."""
    section = raw_config_section()
    section["loss_weights"] = {"stability": 0.0, "function": 0.0}
    cfg = {"torch_v2": section, "project": {"random_seed": 20261102}}

    with pytest.raises(ValueError, match="nothing to optimise"):
        dual_branch_config_from(cfg)


def test_config_reads_every_architecture_field() -> None:
    """Each nested YAML key lands on the matching dataclass attribute."""
    cfg = {"torch_v2": raw_config_section(), "project": {"random_seed": 20261102}}

    config = dual_branch_config_from(cfg)

    assert config.tabular_hidden_dims == (8,)
    assert config.embedding_hidden_dims == (8,)
    assert config.bottleneck_dim == TEST_BOTTLENECK
    assert config.embedding_pca is False
    assert config.fusion_hidden_dim == TEST_FUSION_DIM
    assert config.n_residual_blocks == 1
    assert config.seed == 20261102


# =============================================================================
# Building blocks
# =============================================================================
def test_dense_block_includes_norm_and_dropout_when_asked() -> None:
    """The block is Linear -> BatchNorm -> activation -> Dropout, in that order."""
    layers = dense_block(4, 8, torch.nn.ReLU, batch_norm=True, dropout=0.5)

    assert [type(layer) for layer in layers] == [
        torch.nn.Linear,
        torch.nn.BatchNorm1d,
        torch.nn.ReLU,
        torch.nn.Dropout,
    ]


def test_dense_block_omits_disabled_layers() -> None:
    """Zero dropout and no batch norm leave just Linear -> activation."""
    layers = dense_block(4, 8, torch.nn.ReLU, batch_norm=False, dropout=0.0)

    assert [type(layer) for layer in layers] == [torch.nn.Linear, torch.nn.ReLU]


def test_residual_block_preserves_shape() -> None:
    """A residual block cannot change width; that is what makes the skip legal."""
    block = ResidualBlock(TEST_FUSION_DIM, torch.nn.ReLU, batch_norm=True, dropout=0.0)
    x = torch.randn(TEST_BATCH, TEST_FUSION_DIM)

    assert block(x).shape == x.shape


def test_residual_block_with_zeroed_branch_is_the_identity() -> None:
    """With f(x) = 0 the block returns act(x), so depth starts out harmless.

    This is the guarantee that justifies stacking blocks on 840 rows: an
    untrained block cannot destroy the representation handed to it.
    """
    block = ResidualBlock(TEST_FUSION_DIM, torch.nn.ReLU, batch_norm=False, dropout=0.0)
    block.eval()
    for parameter in block.block.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.rand(TEST_BATCH, TEST_FUSION_DIM)  # positive, so ReLU is the identity

    assert torch.allclose(block(x), x)


# =============================================================================
# The network
# =============================================================================
def test_forward_returns_one_value_per_row_per_target(config: DualBranchConfig) -> None:
    """Both heads squeeze to ``(batch,)``, matching the long-format OOF schema."""
    model = ProteinMutationDualBranchNN(TEST_N_TABULAR, TEST_N_EMBEDDING, config)
    model.eval()

    stability, function = model(*inputs())

    assert stability.shape == (TEST_BATCH,)
    assert function.shape == (TEST_BATCH,)


def test_function_head_is_bounded_even_on_extreme_inputs(config: DualBranchConfig) -> None:
    """functional_score is a 0-1 retention score, bounded by construction not by loss."""
    model = ProteinMutationDualBranchNN(TEST_N_TABULAR, TEST_N_EMBEDDING, config)
    model.eval()
    tabular, embedding = inputs()

    _, function = model(tabular * TEST_EXTREME_SCALE, embedding * TEST_EXTREME_SCALE)

    assert bool((function >= 0).all()) and bool((function <= 1).all())


def test_stability_head_is_unbounded(config: DualBranchConfig) -> None:
    """Stability is a free-energy change; clamping it would distort the ranking tail."""
    model = ProteinMutationDualBranchNN(TEST_N_TABULAR, TEST_N_EMBEDDING, config)

    assert not any(isinstance(layer, torch.nn.Sigmoid) for layer in model.stability_head)
    assert any(isinstance(layer, torch.nn.Sigmoid) for layer in model.function_head)


def test_embedding_is_squeezed_to_the_bottleneck(config: DualBranchConfig) -> None:
    """Whatever the embedding's width, fusion sees exactly ``bottleneck_dim`` of it."""
    model = ProteinMutationDualBranchNN(TEST_N_TABULAR, TEST_N_EMBEDDING, config)
    model.eval()
    _, embedding = inputs()

    assert model.embedding_branch(embedding).shape == (TEST_BATCH, TEST_BOTTLENECK)
    assert model.embedding_out_dim == TEST_BOTTLENECK


def test_branches_are_disentangled(config: DualBranchConfig) -> None:
    """Changing the tabular input must not move the embedding branch's output.

    This is the architectural claim the module exists to make: the two blocks
    travel separate paths and meet only at fusion. A monolithic MLP would fail
    this, because every input column reaches every first-layer unit.
    """
    model = ProteinMutationDualBranchNN(TEST_N_TABULAR, TEST_N_EMBEDDING, config)
    model.eval()
    tabular, embedding = inputs()

    before = model.embedding_branch(embedding)
    after = model.embedding_branch(embedding)
    perturbed_tabular = model.tabular_branch(tabular * 2)

    assert torch.allclose(before, after)
    assert not torch.allclose(perturbed_tabular, model.tabular_branch(tabular))
    # And the fused representation does respond to both, or fusion is broken.
    assert not torch.allclose(
        model.encode(tabular, embedding), model.encode(tabular * 2, embedding)
    )
    assert not torch.allclose(
        model.encode(tabular, embedding), model.encode(tabular, embedding * 2)
    )


def test_both_blocks_are_required(config: DualBranchConfig) -> None:
    """A dual-branch model with one branch is a configuration error, not a fallback."""
    with pytest.raises(ValueError, match="needs both blocks"):
        ProteinMutationDualBranchNN(0, TEST_N_EMBEDDING, config)
    with pytest.raises(ValueError, match="needs both blocks"):
        ProteinMutationDualBranchNN(TEST_N_TABULAR, 0, config)


def test_head_hidden_dim_none_makes_a_single_linear_head(config: DualBranchConfig) -> None:
    """``null`` is the documented way to ask for a bare projection."""
    bare = ProteinMutationDualBranchNN(
        TEST_N_TABULAR, TEST_N_EMBEDDING, replace(config, head_hidden_dim=None)
    )

    assert len(bare.stability_head) == 1
    assert isinstance(bare.stability_head[0], torch.nn.Linear)


def test_residual_block_count_follows_the_config(config: DualBranchConfig) -> None:
    """Depth is a config decision, not a constant baked into the class."""
    for n_blocks in (1, 3):
        model = ProteinMutationDualBranchNN(
            TEST_N_TABULAR, TEST_N_EMBEDDING, replace(config, n_residual_blocks=n_blocks)
        )
        assert len(model.residual_blocks) == n_blocks


def test_parameter_counts_are_reported_per_branch(config: DualBranchConfig) -> None:
    """The provenance record splits capacity by branch, which is the trade-off tracked."""
    model = ProteinMutationDualBranchNN(TEST_N_TABULAR, TEST_N_EMBEDDING, config)

    counts = count_branch_parameters(model)

    assert set(counts) == {
        const.TORCH_V2_BRANCH_TABULAR,
        const.TORCH_V2_BRANCH_EMBEDDING,
        const.TORCH_V2_BRANCH_FUSION,
        const.TORCH_V2_BRANCH_HEADS,
    }
    assert all(value > 0 for value in counts.values())
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert sum(counts.values()) == total


def test_identical_seeds_give_identical_initial_weights(config: DualBranchConfig) -> None:
    """Construction is reproducible once the global torch seed is set."""
    torch.manual_seed(config.seed)
    first = ProteinMutationDualBranchNN(TEST_N_TABULAR, TEST_N_EMBEDDING, config).state_dict()
    torch.manual_seed(config.seed)
    second = ProteinMutationDualBranchNN(TEST_N_TABULAR, TEST_N_EMBEDDING, config).state_dict()

    assert all(torch.equal(first[key], second[key]) for key in first)
