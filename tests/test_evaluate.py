"""Unit tests for ``src.evaluate``: metrics, aggregation, selection and figures.

Three concerns are covered here:

* the metric calculators themselves, against hand-computed values;
* model selection, which must respect each metric's direction, break ties
  deterministically and refuse to choose on NaN scores; and
* the standalone evaluation stage, which re-reports a finished training run and
  writes the figures without refitting anything.

Selection is checked against hand-built summary tables rather than a real
cross-validation run, so a failure points at the selection rule and not at the
model. ``eval_cfg``-style fixtures follow the convention of the other test
modules: each one is self-contained and runs on the synthetic data from
``conftest.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import data_processing as pp
from src import evaluate as ev
from src import train as tr
from src.common import constants as const
from src.common.config import validate_config
from src.common.io import read_table

# The synthetic data has a single labeled protein in one family, so a family-grouped
# training fold can hold one embedding vector; PCA then has zero variance and numpy
# warns while computing explained-variance ratios. Expected for this fixture only.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

TEST_N_SPLITS: int = 2  # synthetic data has only 2 families
TEST_N_COMPONENTS: int = 3  # must not exceed the smallest training fold


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture()
def eval_cfg(cfg: dict[str, Any], raw_files: dict[str, Path]) -> dict[str, Any]:
    """Config with Phase 1 run and a cheap single-model candidate set."""
    cfg["validation"]["n_splits"] = TEST_N_SPLITS
    cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS
    cfg["models"]["candidates"] = {const.ModelName.RIDGE.value: {"alpha": 1.0}}
    validate_config(cfg)
    pp.run_preprocessing(cfg)
    return cfg


@pytest.fixture()
def trained_cfg(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    """Config whose training artifacts already exist on disk, ready for evaluation."""
    tr.run_training(eval_cfg)
    return eval_cfg


# =============================================================================
# Metrics
# =============================================================================
def test_compute_metrics_known_values() -> None:
    """Hand-computed example: one error of 1 on four points."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])

    scores = ev.compute_metrics(y_true, y_pred, list(const.Metric))

    assert scores[const.Metric.RMSE.value] == pytest.approx(0.5)
    assert scores[const.Metric.MAE.value] == pytest.approx(0.25)
    assert scores[const.Metric.R2.value] == pytest.approx(1 - 1 / 5)
    assert scores[const.Metric.SPEARMAN.value] == pytest.approx(1.0)


def test_compute_metrics_constant_prediction_gives_nan_spearman() -> None:
    """Rank correlation is undefined for constant predictions and is reported as NaN, not 0."""
    scores = ev.compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]), [const.Metric.SPEARMAN])

    assert np.isnan(scores[const.Metric.SPEARMAN.value])


def test_compute_metrics_returns_only_requested() -> None:
    """Only configured metrics are computed."""
    scores = ev.compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.5]), [const.Metric.MAE])

    assert list(scores) == [const.Metric.MAE.value]


def make_summary(values: dict[tuple[str, str], float], metric: const.Metric) -> pd.DataFrame:
    """Build a minimal metrics summary from ``{(strategy, model): value}`` for both targets."""
    rows = [
        {
            const.CV_COL_STRATEGY: strategy,
            const.COL_MODEL: model,
            const.COL_TARGET: target,
            f"{const.POOLED_PREFIX}{metric.value}": value,
        }
        for target in const.TARGETS.values()
        for (strategy, model), value in values.items()
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("metric", "values", "expected"),
    [
        (
            const.Metric.SPEARMAN,
            {"ridge": 0.4, "random_forest": 0.7, "gradient_boosting": 0.6},
            "random_forest",
        ),
        (const.Metric.R2, {"ridge": 0.4, "random_forest": 0.1, "gradient_boosting": 0.3}, "ridge"),
        (
            const.Metric.RMSE,
            {"ridge": 0.4, "random_forest": 0.5, "gradient_boosting": 0.3},
            "gradient_boosting",
        ),
        (const.Metric.MAE, {"ridge": 0.2, "random_forest": 0.5, "gradient_boosting": 0.3}, "ridge"),
        (
            const.Metric.SPEARMAN,
            {"ridge": 0.6, "random_forest": 0.6, "gradient_boosting": 0.5},
            "ridge",
        ),  # tie
        (
            const.Metric.SPEARMAN,
            {"ridge": float("nan"), "random_forest": 0.1, "gradient_boosting": 0.0},
            "random_forest",
        ),
    ],
)
def test_select_best_models_respects_metric_direction(
    eval_cfg: dict[str, Any], metric: const.Metric, values: dict[str, float], expected: str
) -> None:
    """Higher wins for r2/spearman, lower for rmse/mae; ties go to config order; NaN is skipped."""
    primary = eval_cfg["validation"]["primary_strategy"]
    eval_cfg["models"]["selection_metric"] = metric.value
    summary = make_summary({(primary, m): v for m, v in values.items()}, metric)

    selection = ev.select_best_models(summary, eval_cfg)

    for target in const.TARGETS.values():
        assert selection[target][const.COL_MODEL] == expected
        assert selection[target][const.CV_COL_STRATEGY] == primary


def test_select_best_models_ignores_non_primary_strategies(eval_cfg: dict[str, Any]) -> None:
    """A model that only looks best under the (optimistic) random split is not selected."""
    primary = eval_cfg["validation"]["primary_strategy"]
    eval_cfg["models"]["selection_metric"] = const.Metric.SPEARMAN.value
    summary = make_summary(
        {
            (const.SplitStrategy.RANDOM.value, "ridge"): 0.99,
            (primary, "ridge"): 0.2,
            (primary, "random_forest"): 0.5,
        },
        const.Metric.SPEARMAN,
    )

    selection = ev.select_best_models(summary, eval_cfg)

    assert selection[const.TARGET_STABILITY][const.COL_MODEL] == "random_forest"


def test_select_best_models_raises_without_finite_scores(eval_cfg: dict[str, Any]) -> None:
    """If every candidate scored NaN there is nothing defensible to select."""
    primary = eval_cfg["validation"]["primary_strategy"]
    eval_cfg["models"]["selection_metric"] = const.Metric.SPEARMAN.value
    summary = make_summary({(primary, "ridge"): float("nan")}, const.Metric.SPEARMAN)

    with pytest.raises(ValueError, match="No model has a finite"):
        ev.select_best_models(summary, eval_cfg)



# =============================================================================
# Optimism gap
# =============================================================================
def test_optimism_gap_reports_random_minus_primary(eval_cfg: dict[str, Any]) -> None:
    """The gap is the random split's advantage over the primary one, signed and per target."""
    primary = eval_cfg["validation"]["primary_strategy"]
    eval_cfg["models"]["selection_metric"] = const.Metric.SPEARMAN.value
    summary = make_summary(
        {(const.SplitStrategy.RANDOM.value, "ridge"): 0.9, (primary, "ridge"): 0.5},
        const.Metric.SPEARMAN,
    )

    gaps = ev.log_optimism_gap(summary, eval_cfg)

    assert len(gaps) == len(const.TARGETS)
    assert gaps[const.EVAL_COL_OPTIMISM_GAP].to_numpy() == pytest.approx(0.4)


def test_optimism_gap_is_empty_without_a_random_split(eval_cfg: dict[str, Any]) -> None:
    """With no random split there is nothing to compare against, so no gap is claimed."""
    primary = eval_cfg["validation"]["primary_strategy"]
    eval_cfg["models"]["selection_metric"] = const.Metric.SPEARMAN.value
    summary = make_summary({(primary, "ridge"): 0.5}, const.Metric.SPEARMAN)

    assert ev.log_optimism_gap(summary, eval_cfg).empty


# =============================================================================
# Standalone evaluation of saved artifacts
# =============================================================================
def test_load_evaluation_inputs_requires_training(eval_cfg: dict[str, Any]) -> None:
    """Evaluating before training points the user at the stage they skipped."""
    with pytest.raises(FileNotFoundError, match="src.train"):
        ev.load_evaluation_inputs(eval_cfg)


def test_selected_oof_keeps_only_the_chosen_model_and_strategy(trained_cfg: dict[str, Any]) -> None:
    """Rows from other strategies or unselected models are excluded from the headline."""
    oof, _, selection = ev.load_evaluation_inputs(trained_cfg)
    strategy = ev.evaluation_strategy(trained_cfg)

    focus = ev.selected_oof(oof, selection, strategy)

    assert set(focus[const.CV_COL_STRATEGY]) == {strategy}
    assert set(focus[const.COL_TARGET]) == set(const.TARGETS.values())
    for target, choice in selection.items():
        rows = focus[focus[const.COL_TARGET] == target]
        assert set(rows[const.COL_MODEL]) == {choice[const.COL_MODEL]}


def test_evaluation_strategy_defaults_to_primary(trained_cfg: dict[str, Any]) -> None:
    """A null ``evaluation.strategy`` means "whatever the pipeline selects models under"."""
    trained_cfg["evaluation"]["strategy"] = None

    assert ev.evaluation_strategy(trained_cfg) == trained_cfg["validation"]["primary_strategy"]

    trained_cfg["evaluation"]["strategy"] = const.SplitStrategy.RANDOM.value
    assert ev.evaluation_strategy(trained_cfg) == const.SplitStrategy.RANDOM.value


def test_headline_table_matches_pooled_metrics(trained_cfg: dict[str, Any]) -> None:
    """The headline row for a target is the pooled metric over that target's OOF rows."""
    oof, _, selection = ev.load_evaluation_inputs(trained_cfg)
    focus = ev.selected_oof(oof, selection, ev.evaluation_strategy(trained_cfg))

    headline = ev.headline_table(focus, trained_cfg)

    assert len(headline) == len(const.TARGETS)
    for _, row in headline.iterrows():
        rows = focus[focus[const.COL_TARGET] == row[const.COL_TARGET]]
        expected = ev.pooled_metrics(rows, ev.configured_metrics(trained_cfg))
        assert row[const.COL_N_SAMPLES] == len(rows)
        for name, value in expected.items():
            assert row[f"{const.POOLED_PREFIX}{name}"] == pytest.approx(value, nan_ok=True)


def test_run_evaluation_writes_figures_and_summary(trained_cfg: dict[str, Any]) -> None:
    """The stage writes every configured figure plus the Markdown summary."""
    headline = ev.run_evaluation(trained_cfg)

    figures = Path(trained_cfg["paths"]["figures_dir"])
    for key in const.FIGURE_FILE_KEYS:
        path = figures / trained_cfg["files"][key]
        assert path.is_file(), key
        assert path.stat().st_size > 0
    summary_path = Path(trained_cfg["paths"]["metrics_dir"], trained_cfg["files"]["evaluation_markdown"])
    text = summary_path.read_text(encoding=const.CONFIG_ENCODING)
    assert "# Evaluation summary" in text
    for target in const.TARGETS.values():
        assert target in text
    assert len(headline) == len(const.TARGETS)


def test_run_evaluation_can_skip_figures(trained_cfg: dict[str, Any]) -> None:
    """``evaluation.figures: false`` writes the summary but no image files."""
    trained_cfg["evaluation"]["figures"] = False

    ev.run_evaluation(trained_cfg)

    figures = Path(trained_cfg["paths"]["figures_dir"])
    assert not any((figures / trained_cfg["files"][key]).exists() for key in const.FIGURE_FILE_KEYS)
    assert Path(trained_cfg["paths"]["metrics_dir"], trained_cfg["files"]["evaluation_markdown"]).is_file()


def test_run_evaluation_does_not_refit_or_touch_models(trained_cfg: dict[str, Any]) -> None:
    """Evaluation is a read-only stage: the saved model artifacts are left untouched."""
    models = Path(trained_cfg["paths"]["models_dir"])
    before = {p: p.stat().st_mtime_ns for p in sorted(models.rglob("*")) if p.is_file()}

    ev.run_evaluation(trained_cfg)

    after = {p: p.stat().st_mtime_ns for p in sorted(models.rglob("*")) if p.is_file()}
    assert before == after


def test_run_evaluation_rejects_a_strategy_that_was_never_run(trained_cfg: dict[str, Any]) -> None:
    """Asking for a strategy with no out-of-fold rows fails rather than reporting nothing."""
    trained_cfg["evaluation"]["strategy"] = const.SplitStrategy.RANDOM.value
    oof_path = Path(trained_cfg["paths"]["metrics_dir"], trained_cfg["files"]["oof_predictions"])
    oof = read_table(oof_path)
    kept = oof[oof[const.CV_COL_STRATEGY] != const.SplitStrategy.RANDOM.value]
    kept.to_parquet(oof_path, index=False)

    with pytest.raises(ValueError, match="No out-of-fold predictions"):
        ev.run_evaluation(trained_cfg)


def test_evaluation_summary_reports_the_same_numbers_as_training(trained_cfg: dict[str, Any]) -> None:
    """Re-reporting a finished run reproduces the selection metric Phase 3 recorded."""
    selection_path = Path(trained_cfg["paths"]["metrics_dir"], trained_cfg["files"]["model_selection"])
    selection = json.loads(selection_path.read_text(encoding=const.CONFIG_ENCODING))

    headline = ev.run_evaluation(trained_cfg)

    column = f"{const.POOLED_PREFIX}{trained_cfg['models']['selection_metric']}"
    for target, choice in selection.items():
        row = headline[headline[const.COL_TARGET] == target].iloc[0]
        assert row[column] == pytest.approx(choice["value"], nan_ok=True)
