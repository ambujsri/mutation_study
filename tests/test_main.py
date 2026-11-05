"""Unit and integration tests for the ``main.py`` pipeline orchestrator.

Two things are worth guaranteeing about an orchestrator, and nothing else:

* **stage selection** — the stages actually run are the ones asked for, always
  in dependency order, and an unknown name fails loudly rather than being
  silently skipped; and
* **the seam** — every stage really does run end to end on the synthetic data,
  consuming the previous stage's artifacts.

Ordering is tested against an injected fake registry so it cannot depend on how
long a real stage takes, while the end-to-end test runs the genuine pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import main
from src import data_processing as pp
from src.common import constants as const
from src.common.config import validate_config

# The synthetic data has a single labeled protein in one family, so a family-grouped
# training fold can hold one embedding vector; PCA then has zero variance and numpy
# warns while computing explained-variance ratios. Expected for this fixture only.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

TEST_N_SPLITS: int = 2  # synthetic data has only 2 families
TEST_N_COMPONENTS: int = 2
TEST_TOP_K: int = 4
TEST_N_CANDIDATES: int = 2  # candidates written up by the interpretation stage
TEST_MAX_EPOCHS: int = 3    # the end-to-end test checks wiring, not convergence
TEST_PATIENCE: int = 2
TEST_BOTTLENECK_DIM: int = 4
TEST_FUSION_DIM: int = 8
TEST_ATTRIBUTION_REPEATS: int = 1


# =============================================================================
# Fixtures & helpers
# =============================================================================
@pytest.fixture()
def pipeline_cfg(cfg: dict[str, Any], raw_files: dict[str, Path]) -> dict[str, Any]:
    """Config small enough to run every stage end to end inside one test."""
    cfg["validation"]["n_splits"] = TEST_N_SPLITS
    cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS
    cfg["models"]["candidates"] = {const.ModelName.RIDGE.value: {"alpha": 1.0}}
    cfg["ablation"]["model"] = const.ModelName.RIDGE.value
    cfg["ablation"]["subsets"] = {
        "tabular_only": [const.GROUP_PHYSICOCHEMICAL, const.GROUP_SEQUENCE],
        "full": [
            const.GROUP_PHYSICOCHEMICAL,
            const.GROUP_SEQUENCE,
            const.GROUP_STRUCTURAL,
            const.GROUP_DELTAS,
            const.GROUP_EMBEDDINGS,
        ],
    }
    cfg["ablation"]["baseline"] = "full"
    cfg["prioritization"]["top_k"] = TEST_TOP_K
    cfg["prioritization"]["min_fold_models"] = TEST_N_SPLITS
    cfg["interpretation"]["n_candidates"] = TEST_N_CANDIDATES
    # The neural network is trained for real in the end-to-end test, so keep it
    # tiny: the test asserts that the stage runs and writes its artifacts, not
    # that a 24-row synthetic fit is any good.
    cfg["torch"]["hidden_dims"] = [8]
    cfg["torch"]["head_hidden_dim"] = None
    cfg["torch"]["batch_size"] = 8
    cfg["torch"]["max_epochs"] = TEST_MAX_EPOCHS
    cfg["torch"]["early_stopping"]["patience"] = TEST_PATIENCE
    cfg["torch_v2"]["tabular_branch"]["hidden_dims"] = [8]
    cfg["torch_v2"]["embedding_branch"]["hidden_dims"] = [8]
    cfg["torch_v2"]["embedding_branch"]["bottleneck_dim"] = TEST_BOTTLENECK_DIM
    cfg["torch_v2"]["fusion"]["hidden_dim"] = TEST_FUSION_DIM
    cfg["torch_v2"]["fusion"]["n_residual_blocks"] = 1
    cfg["torch_v2"]["head_hidden_dim"] = None
    cfg["torch_v2"]["batch_size"] = 8
    cfg["torch_v2"]["max_epochs"] = TEST_MAX_EPOCHS
    cfg["torch_v2"]["early_stopping"]["patience"] = TEST_PATIENCE
    # Permutation importance costs one predict per feature per repeat per fold,
    # so the end-to-end run uses the smallest setting that still produces a table.
    cfg["attribution"]["n_repeats"] = TEST_ATTRIBUTION_REPEATS
    validate_config(cfg)
    return cfg


def recording_registry(names: list[str], log: list[str]) -> dict[str, Any]:
    """Build a fake stage registry whose stages only record that they ran."""
    return {name: (lambda _cfg, name=name: log.append(name)) for name in names}


# =============================================================================
# Stage selection
# =============================================================================
def test_stage_registry_is_in_dependency_order() -> None:
    """Each stage consumes the one before it, so the declared order is the real order."""
    assert list(main.STAGES) == [
        "data_processing",
        "validation",
        "train",
        "train_torch",
        "train_torch_v2",
        "evaluate",
        "predict",
        "ablation",
        "attribution",
        "interpret",
    ]


def test_no_selection_runs_every_stage() -> None:
    """The default is the whole pipeline."""
    assert main.resolve_stages(None, None) == list(main.STAGES)


def test_requested_stages_are_reordered_canonically() -> None:
    """Stages run in dependency order whatever order they were typed in."""
    assert main.resolve_stages(["predict", "train"], None) == ["train", "predict"]


def test_from_runs_the_stage_and_everything_after() -> None:
    """``--from`` resumes a pipeline without redoing the stages that have not changed."""
    assert main.resolve_stages(None, "evaluate") == [
        "evaluate", "predict", "ablation", "attribution", "interpret",
    ]


def test_from_the_first_stage_is_the_whole_pipeline() -> None:
    """Resuming from the first stage is equivalent to running everything."""
    assert main.resolve_stages(None, "data_processing") == list(main.STAGES)


def test_unknown_requested_stage_is_rejected() -> None:
    """A typo must fail rather than quietly running a shorter pipeline."""
    with pytest.raises(ValueError, match="Unknown stage"):
        main.resolve_stages(["trian"], None)


def test_unknown_from_stage_is_rejected() -> None:
    """The same guarantee applies to ``--from``."""
    with pytest.raises(ValueError, match="Unknown stage"):
        main.resolve_stages(None, "evaluat")


def test_stage_selectors_are_mutually_exclusive() -> None:
    """``--stages`` and ``--from`` mean different things; combining them is ambiguous."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        main.resolve_stages(["train"], "evaluate")


# =============================================================================
# Execution order
# =============================================================================
def test_run_pipeline_executes_stages_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stages are called once each, in the order given, and every one is timed."""
    log: list[str] = []
    names = ["first", "second", "third"]
    monkeypatch.setattr(main, "STAGES", recording_registry(names, log))

    timings = main.run_pipeline({}, names)

    assert log == names
    assert list(timings) == names
    assert all(seconds >= 0 for seconds in timings.values())


def test_run_pipeline_propagates_stage_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing stage stops the pipeline instead of letting later stages read stale artifacts."""
    log: list[str] = []

    def boom(_cfg: dict[str, Any]) -> None:
        raise RuntimeError("stage failed")

    monkeypatch.setattr(main, "STAGES", {"ok": lambda _cfg: log.append("ok"), "bad": boom,
                                         "never": lambda _cfg: log.append("never")})

    with pytest.raises(RuntimeError, match="stage failed"):
        main.run_pipeline({}, ["ok", "bad", "never"])

    assert log == ["ok"]


# =============================================================================
# End-to-end
# =============================================================================
def test_full_pipeline_produces_every_deliverable(pipeline_cfg: dict[str, Any]) -> None:
    """Running all stages in order yields the processed data, metrics, figures and rankings."""
    timings = main.run_pipeline(pipeline_cfg, main.resolve_stages(None, None))

    assert list(timings) == list(main.STAGES)
    expected = {
        "processed_dir": ["processed_train", "processed_candidates", "feature_manifest"],
        "metrics_dir": [
            "cv_fold_summary",
            "oof_predictions",
            "cv_fold_metrics",
            "cv_metrics_summary",
            "model_selection",
            "evaluation_markdown",
            "evaluation_json",
            "model_comparison",
            "ablation_summary",
            "ablation_summary_markdown",
            "feature_importances",
            "feature_group_importances",
            "attribution_markdown",
            "torch_cv_metrics",
            "torch_history",
            "torch_fold_summary",
            "torch_comparison",
            "torch_comparison_markdown",
            "torch_v2_cv_metrics",
            "torch_v2_history",
            "torch_v2_fold_summary",
            "torch_v2_comparison",
            "torch_v2_comparison_markdown",
        ],
        "models_dir": ["training_metadata"],
        "figures_dir": [
            *const.FIGURE_FILE_KEYS,
            *const.TORCH_FIGURE_FILE_KEYS,
            *const.TORCH_V2_FIGURE_FILE_KEYS,
            *const.ABLATION_FIGURE_FILE_KEYS,
            *const.ATTR_FIGURE_FILE_KEYS,
        ],
        "rankings_dir": [
            "all_predictions",
            "top_candidates",
            "top_candidates_markdown",
            "interpretation_markdown",
        ],
    }
    for dir_key, file_keys in expected.items():
        for file_key in file_keys:
            path = Path(pipeline_cfg["paths"][dir_key], pipeline_cfg["files"][file_key])
            assert path.is_file(), f"{dir_key}/{file_key}"
            assert path.stat().st_size > 0, f"{dir_key}/{file_key} is empty"


def test_resuming_from_a_later_stage_reuses_existing_artifacts(pipeline_cfg: dict[str, Any]) -> None:
    """``--from evaluate`` refreshes downstream output without re-running Phase 1."""
    main.run_pipeline(pipeline_cfg, main.resolve_stages(None, None))
    processed = Path(pipeline_cfg["paths"]["processed_dir"], pipeline_cfg["files"]["processed_train"])
    before = processed.stat().st_mtime_ns

    main.run_pipeline(pipeline_cfg, main.resolve_stages(None, "evaluate"))

    assert processed.stat().st_mtime_ns == before


def test_single_stage_runs_without_the_rest(pipeline_cfg: dict[str, Any]) -> None:
    """A stage can be run on its own, which is how the per-phase commands in the README work."""
    main.run_pipeline(pipeline_cfg, main.resolve_stages(["data_processing"], None))

    assert Path(pipeline_cfg["paths"]["processed_dir"], pipeline_cfg["files"]["processed_train"]).is_file()
    assert not Path(pipeline_cfg["paths"]["metrics_dir"], pipeline_cfg["files"]["model_selection"]).exists()


def test_stage_registry_matches_the_modules_it_orchestrates(pipeline_cfg: dict[str, Any]) -> None:
    """Every stage is a callable taking only the config, so main adds no behaviour of its own."""
    assert all(callable(stage) for stage in main.STAGES.values())
    assert main.STAGES["data_processing"] is pp.run_preprocessing
