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
from src.common.io import read_table, write_table

# The synthetic data has a single labeled protein in one family, so a family-grouped
# training fold can hold one embedding vector; PCA then has zero variance and numpy
# warns while computing explained-variance ratios. Expected for this fixture only.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

TEST_N_SPLITS: int = 2  # synthetic data has only 2 families
TEST_N_COMPONENTS: int = 3  # must not exceed the smallest training fold
# Shifts the stand-in network predictions off the baseline's so the overlay
# figure has two visibly distinct series to draw.
TEST_PREDICTION_OFFSET: float = 0.1


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
    # Pearson is below 1 where Spearman is exactly 1: the ordering is perfect
    # but the relationship is not linear. This is why both are reported.
    assert scores[const.Metric.PEARSON.value] == pytest.approx(0.9827076, abs=1e-6)
    assert scores[const.Metric.PEARSON.value] < scores[const.Metric.SPEARMAN.value]


def test_compute_metrics_covers_every_declared_metric() -> None:
    """Every ``Metric`` member has a calculator, so adding one cannot silently no-op."""
    scores = ev.compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]), list(const.Metric))

    assert set(scores) == {metric.value for metric in const.Metric}
    assert all(np.isfinite(value) for value in scores.values())


def test_pearson_and_spearman_differ_on_a_monotone_nonlinear_fit() -> None:
    """A perfectly ranked but curved fit scores 1.0 on Spearman and less on Pearson."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true**3  # monotone, strongly nonlinear

    scores = ev.compute_metrics(y_true, y_pred, [const.Metric.PEARSON, const.Metric.SPEARMAN])

    assert scores[const.Metric.SPEARMAN.value] == pytest.approx(1.0)
    assert scores[const.Metric.PEARSON.value] < 0.95


@pytest.mark.parametrize("metric", [const.Metric.PEARSON, const.Metric.SPEARMAN])
def test_correlation_is_nan_for_a_constant_prediction(metric: const.Metric) -> None:
    """Correlation is undefined for constant predictions and is reported as NaN, not 0."""
    scores = ev.compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]), [metric])

    assert np.isnan(scores[metric.value])


def test_metric_directions_are_declared_for_every_metric() -> None:
    """Model selection needs to know which way each metric points."""
    assert set(const.METRIC_GREATER_IS_BETTER) == set(const.Metric)
    assert const.METRIC_GREATER_IS_BETTER[const.Metric.PEARSON] is True
    assert const.METRIC_GREATER_IS_BETTER[const.Metric.RMSE] is False


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


# =============================================================================
# Baseline benchmark (assignment Task 3)
# =============================================================================
@pytest.fixture()
def multi_model_cfg(cfg: dict[str, Any], raw_files: dict[str, Path]) -> dict[str, Any]:
    """Config carrying all three baseline families, kept cheap enough to fit in a test."""
    cfg["validation"]["n_splits"] = TEST_N_SPLITS
    cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS
    cfg["models"]["candidates"] = {
        const.ModelName.RIDGE.value: {"alpha": 1.0},
        const.ModelName.RANDOM_FOREST.value: {
            "n_estimators": 5, "max_depth": 3, "min_samples_leaf": 1,
            "max_features": 0.5, "n_jobs": 1,
        },
        const.ModelName.GRADIENT_BOOSTING.value: {
            "n_estimators": 5, "learning_rate": 0.1, "max_depth": 2, "subsample": 1.0,
        },
    }
    validate_config(cfg)
    pp.run_preprocessing(cfg)
    tr.run_training(cfg)
    return cfg


def test_benchmark_covers_every_model_target_and_strategy(multi_model_cfg: dict[str, Any]) -> None:
    """The Task 3 table is the full grid: one row per model x target x strategy."""
    _, summary, _ = ev.load_evaluation_inputs(multi_model_cfg)

    table = ev.benchmark_table(summary, multi_model_cfg)

    n_models = len(multi_model_cfg["models"]["candidates"])
    n_strategies = len(multi_model_cfg["validation"]["strategies"])
    assert len(table) == n_models * n_strategies * len(const.TARGETS)
    assert set(table[const.COL_MODEL]) == set(multi_model_cfg["models"]["candidates"])
    assert set(table[const.CV_COL_STRATEGY]) == set(multi_model_cfg["validation"]["strategies"])
    assert set(table[const.COL_TARGET]) == set(const.TARGETS.values())


def test_benchmark_reports_every_configured_metric(multi_model_cfg: dict[str, Any]) -> None:
    """All five regression metrics appear, Pearson and Spearman among them."""
    _, summary, _ = ev.load_evaluation_inputs(multi_model_cfg)

    table = ev.benchmark_table(summary, multi_model_cfg)

    for metric in ev.configured_metrics(multi_model_cfg):
        assert f"{const.POOLED_PREFIX}{metric.value}" in table.columns
    for required in (const.Metric.MAE, const.Metric.RMSE, const.Metric.R2,
                     const.Metric.PEARSON, const.Metric.SPEARMAN):
        assert f"{const.POOLED_PREFIX}{required.value}" in table.columns


def test_benchmark_values_match_the_saved_summary(multi_model_cfg: dict[str, Any]) -> None:
    """The table re-presents the saved metrics; it never recomputes or refits."""
    _, summary, _ = ev.load_evaluation_inputs(multi_model_cfg)

    table = ev.benchmark_table(summary, multi_model_cfg)

    keys = [const.COL_TARGET, const.COL_MODEL, const.CV_COL_STRATEGY]
    expected = summary.set_index(keys)
    for _, row in table.iterrows():
        source = expected.loc[tuple(row[k] for k in keys)]
        for metric in ev.configured_metrics(multi_model_cfg):
            column = f"{const.POOLED_PREFIX}{metric.value}"
            assert row[column] == pytest.approx(source[column], nan_ok=True)


def test_benchmark_optimism_gap_is_random_minus_row(multi_model_cfg: dict[str, Any]) -> None:
    """The gap column quantifies what the optimistic split would have added."""
    _, summary, _ = ev.load_evaluation_inputs(multi_model_cfg)
    column = f"{const.POOLED_PREFIX}{multi_model_cfg['models']['selection_metric']}"
    random_ = const.SplitStrategy.RANDOM.value

    table = ev.benchmark_table(summary, multi_model_cfg)

    indexed = table.set_index([const.COL_TARGET, const.COL_MODEL, const.CV_COL_STRATEGY])
    for (target, model, strategy), row in indexed.iterrows():
        if strategy == random_:
            assert np.isnan(row[const.EVAL_COL_OPTIMISM_GAP])  # nothing to compare against itself
        else:
            reference = indexed.loc[(target, model, random_), column]
            assert row[const.EVAL_COL_OPTIMISM_GAP] == pytest.approx(reference - row[column])


def test_benchmark_rows_follow_config_order(multi_model_cfg: dict[str, Any]) -> None:
    """Rows read in the order the config declares, so the table is stable across runs."""
    _, summary, _ = ev.load_evaluation_inputs(multi_model_cfg)

    table = ev.benchmark_table(summary, multi_model_cfg)

    first_target = next(iter(const.TARGETS.values()))
    models = list(dict.fromkeys(table[table[const.COL_TARGET] == first_target][const.COL_MODEL]))
    assert models == list(multi_model_cfg["models"]["candidates"])


def test_benchmark_markdown_names_both_targets_and_all_models(multi_model_cfg: dict[str, Any]) -> None:
    """The report section is self-describing: models, targets and the gap's definition."""
    _, summary, _ = ev.load_evaluation_inputs(multi_model_cfg)
    table = ev.benchmark_table(summary, multi_model_cfg)

    markdown = ev.benchmark_to_markdown(table, multi_model_cfg)

    assert "# Baseline model benchmark" in markdown
    for target in const.TARGETS.values():
        assert f"## {target}" in markdown
    for model in multi_model_cfg["models"]["candidates"]:
        assert model in markdown
    assert const.EVAL_COL_OPTIMISM_GAP in markdown


def test_run_evaluation_writes_the_benchmark_artifacts(trained_cfg: dict[str, Any]) -> None:
    """The evaluation stage emits the Task 3 table as both CSV and Markdown."""
    ev.run_evaluation(trained_cfg)

    metrics_dir = Path(trained_cfg["paths"]["metrics_dir"])
    csv_path = metrics_dir / trained_cfg["files"]["baseline_benchmark"]
    md_path = metrics_dir / trained_cfg["files"]["baseline_benchmark_markdown"]
    assert csv_path.is_file() and md_path.is_file()
    written = read_table(csv_path)
    assert not written.empty
    assert f"{const.POOLED_PREFIX}{const.Metric.PEARSON.value}" in written.columns


# =============================================================================
# Neural-network and comparison figures (assignment Task 4)
# =============================================================================
@pytest.fixture()
def torch_frames(trained_cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Minimal stand-ins for the artifacts ``src.train_torch`` writes.

    Built by hand rather than by training a network: these tests are about the
    plotting code, and a real fit would make them slow and non-deterministic
    for no extra coverage.
    """
    strategy = ev.evaluation_strategy(trained_cfg)
    oof, _, selection = ev.load_evaluation_inputs(trained_cfg)
    baseline = ev.selected_oof(oof, selection, strategy)
    network = baseline.copy()
    network[const.COL_MODEL] = const.TORCH_MODEL_NAME
    network[const.COL_Y_PRED] = network[const.COL_Y_PRED] + TEST_PREDICTION_OFFSET

    history = pd.DataFrame(
        {
            const.CV_COL_STRATEGY: strategy,
            const.CV_COL_FOLD: [0, 0, 0, 1, 1, 1],
            const.TORCH_COL_EPOCH: [1, 2, 3, 1, 2, 3],
            const.TORCH_COL_TRAIN_LOSS: [0.9, 0.6, 0.4, 0.8, 0.5, 0.3],
            const.TORCH_COL_VAL_LOSS: [1.0, 0.7, 0.8, 0.9, 0.6, 0.7],
        }
    )
    fold_summary = pd.DataFrame(
        {
            const.CV_COL_STRATEGY: [strategy, strategy],
            const.CV_COL_FOLD: [0, 1],
            const.TORCH_COL_BEST_EPOCH: [2, 2],
            const.TORCH_COL_STOPPED_EPOCH: [3, 3],
        }
    )
    metrics = ev.configured_metrics(trained_cfg)
    comparison = pd.DataFrame(
        [
            {
                const.COL_TARGET: target,
                const.COL_MODEL: model,
                const.CV_COL_STRATEGY: strategy,
                **{f"{const.POOLED_PREFIX}{m.value}": 0.5 for m in metrics},
            }
            for target in const.TARGETS.values()
            for model in (const.ModelName.RIDGE.value, const.TORCH_MODEL_NAME)
        ]
    )
    return {
        "oof": network,
        "history": history,
        "fold_summary": fold_summary,
        "comparison": comparison,
    }


def test_make_torch_figures_writes_every_configured_figure(
    trained_cfg: dict[str, Any], torch_frames: dict[str, pd.DataFrame]
) -> None:
    """All three Task 4 figures land in ``figures_dir`` under their configured names."""
    written = ev.make_torch_figures(
        torch_frames["oof"],
        torch_frames["history"],
        torch_frames["fold_summary"],
        torch_frames["comparison"],
        trained_cfg,
    )

    assert set(written) == set(const.TORCH_FIGURE_FILE_KEYS)
    for key, path in written.items():
        assert path == Path(trained_cfg["paths"]["figures_dir"], trained_cfg["files"][key])
        assert path.is_file() and path.stat().st_size > 0


def test_torch_figures_honour_the_figures_switch(
    trained_cfg: dict[str, Any], torch_frames: dict[str, pd.DataFrame]
) -> None:
    """``evaluation.figures: false`` suppresses the Task 4 figures too, not just Task 3's."""
    trained_cfg["evaluation"]["figures"] = False

    written = ev.make_torch_figures(
        torch_frames["oof"],
        torch_frames["history"],
        torch_frames["fold_summary"],
        torch_frames["comparison"],
        trained_cfg,
    )

    assert written == {}
    figures = Path(trained_cfg["paths"]["figures_dir"])
    assert not any((figures / trained_cfg["files"][k]).exists() for k in const.TORCH_FIGURE_FILE_KEYS)


def test_torch_figures_skip_the_overlay_without_baselines(
    trained_cfg: dict[str, Any], torch_frames: dict[str, pd.DataFrame]
) -> None:
    """A Task 4 run before Task 3 still draws what it can instead of failing.

    The comparison figure needs the baseline metrics and the overlay needs the
    baseline predictions; neither is invented when they are absent.
    """
    Path(trained_cfg["paths"]["metrics_dir"], trained_cfg["files"]["model_selection"]).unlink()

    written = ev.make_torch_figures(
        torch_frames["oof"],
        torch_frames["history"],
        torch_frames["fold_summary"],
        pd.DataFrame(),
        trained_cfg,
    )

    assert set(written) == {"fig_torch_learning_curves"}


def test_learning_curves_draw_one_panel_per_fold(
    trained_cfg: dict[str, Any], torch_frames: dict[str, pd.DataFrame]
) -> None:
    """Every fold in the history gets a panel, and the kept epoch is annotated."""
    path = Path(trained_cfg["paths"]["figures_dir"], trained_cfg["files"]["fig_torch_learning_curves"])

    written = ev.plot_learning_curves(torch_frames["history"], torch_frames["fold_summary"], path, 60)

    assert written == path and path.is_file()


def test_learning_curves_tolerate_a_missing_fold_summary(
    trained_cfg: dict[str, Any], torch_frames: dict[str, pd.DataFrame]
) -> None:
    """Without the early-stopping record the curves still plot, just unannotated."""
    path = Path(trained_cfg["paths"]["figures_dir"], trained_cfg["files"]["fig_torch_learning_curves"])

    written = ev.plot_learning_curves(torch_frames["history"], pd.DataFrame(), path, 60)

    assert written.is_file() and written.stat().st_size > 0


# =============================================================================
# Comparative analysis across families and strategies (assignment Task 5)
# =============================================================================
def write_torch_metrics(cfg: dict[str, Any], strategy: str) -> pd.DataFrame:
    """Write a stand-in ``torch_cv_metrics.csv`` so the network joins the comparison."""
    metrics = ev.configured_metrics(cfg)
    network = pd.DataFrame(
        [
            {
                const.CV_COL_STRATEGY: strategy,
                const.COL_MODEL: const.TORCH_MODEL_NAME,
                const.COL_TARGET: target,
                const.COL_N_SAMPLES: 10,
                **{f"{const.POOLED_PREFIX}{m.value}": 0.4 for m in metrics},
                **{f"{const.FOLD_STD_PREFIX}{m.value}": 0.01 for m in metrics},
            }
            for target in const.TARGETS.values()
        ]
    )
    write_table(network, Path(cfg["paths"]["metrics_dir"], cfg["files"]["torch_cv_metrics"]))
    return network


def test_unified_summary_without_the_network_is_the_baselines(trained_cfg: dict[str, Any]) -> None:
    """Task 5 runs before Task 4 has ever been executed without inventing rows."""
    combined = ev.unified_summary(trained_cfg)

    baselines = read_table(
        Path(trained_cfg["paths"]["metrics_dir"], trained_cfg["files"]["cv_metrics_summary"])
    )
    assert len(combined) == len(baselines)
    assert const.TORCH_MODEL_NAME not in set(combined[const.COL_MODEL])


def test_unified_summary_folds_in_the_network(trained_cfg: dict[str, Any]) -> None:
    """When the network's metrics exist they are stacked onto the baselines."""
    write_torch_metrics(trained_cfg, ev.evaluation_strategy(trained_cfg))

    combined = ev.unified_summary(trained_cfg)

    assert const.TORCH_MODEL_NAME in set(combined[const.COL_MODEL])


def test_model_comparison_covers_both_axes(multi_model_cfg: dict[str, Any]) -> None:
    """One row per model x target x strategy, each tagged with its estimator family."""
    strategy = ev.evaluation_strategy(multi_model_cfg)
    write_torch_metrics(multi_model_cfg, strategy)
    combined = ev.unified_summary(multi_model_cfg)

    comparison = ev.model_comparison_table(combined, multi_model_cfg)

    strategies = set(comparison[const.CV_COL_STRATEGY])
    assert len(strategies) > 1, "the strategy axis needs more than one strategy to compare"
    assert set(comparison[const.COL_MODEL]) == {
        *multi_model_cfg["models"]["candidates"],
        const.TORCH_MODEL_NAME,
    }
    # Families, not names: the three baselines must not collapse into one label.
    assert len(set(comparison[const.EVAL_COL_MODEL_FAMILY])) == 4
    for metric in ev.configured_metrics(multi_model_cfg):
        assert f"{const.POOLED_PREFIX}{metric.value}" in comparison.columns
        assert f"{const.EVAL_OPTIMISM_GAP_PREFIX}{metric.value}" in comparison.columns


def test_optimism_gap_is_random_minus_this_row_for_every_metric(
    multi_model_cfg: dict[str, Any],
) -> None:
    """The per-metric gap is computed against that model's own random-split score."""
    combined = ev.unified_summary(multi_model_cfg)
    comparison = ev.model_comparison_table(combined, multi_model_cfg)
    random_ = const.SplitStrategy.RANDOM.value
    indexed = comparison.set_index([const.COL_TARGET, const.COL_MODEL, const.CV_COL_STRATEGY])

    for (target, model, strategy), row in indexed.iterrows():
        for metric in ev.configured_metrics(multi_model_cfg):
            pooled = f"{const.POOLED_PREFIX}{metric.value}"
            gap = row[f"{const.EVAL_OPTIMISM_GAP_PREFIX}{metric.value}"]
            if strategy == random_:
                assert np.isnan(gap), "a random-split row has nothing to be optimistic against"
            else:
                expected = indexed.loc[(target, model, random_), pooled] - row[pooled]
                assert gap == pytest.approx(expected)


def test_family_leaderboard_ranks_within_a_target(multi_model_cfg: dict[str, Any]) -> None:
    """Ranking is per target, best first, and only on the reported strategy."""
    combined = ev.unified_summary(multi_model_cfg)
    comparison = ev.model_comparison_table(combined, multi_model_cfg)

    leaderboard = ev.family_leaderboard(comparison, multi_model_cfg)

    strategy = ev.evaluation_strategy(multi_model_cfg)
    column = f"{const.POOLED_PREFIX}{multi_model_cfg['models']['selection_metric']}"
    assert set(leaderboard[const.CV_COL_STRATEGY]) == {strategy}
    for target in const.TARGETS.values():
        rows = leaderboard[leaderboard[const.COL_TARGET] == target]
        assert list(rows["rank"]) == list(range(1, len(rows) + 1))
        scores = rows[column].to_numpy(dtype=float)
        assert np.all(np.diff(scores) <= 0), "spearman is higher-is-better, so rank 1 scores most"


def test_interpretation_notes_quote_the_measured_numbers(multi_model_cfg: dict[str, Any]) -> None:
    """The commentary is generated from this run, not fixed prose."""
    combined = ev.unified_summary(multi_model_cfg)
    comparison = ev.model_comparison_table(combined, multi_model_cfg)

    notes = ev.interpretation_notes(comparison, multi_model_cfg)

    text = "\n".join(notes)
    assert "optimism" in text.lower() or "random split" in text
    assert "Pearson" in text and "Spearman" in text
    strategy = ev.evaluation_strategy(multi_model_cfg)
    reported = comparison[comparison[const.CV_COL_STRATEGY] == strategy]
    gap_column = f"{const.EVAL_OPTIMISM_GAP_PREFIX}{multi_model_cfg['models']['selection_metric']}"
    worst = reported.loc[reported[gap_column].idxmax()]
    assert f"{float(worst[gap_column]):+.3f}" in text
    assert str(worst[const.COL_MODEL]) in text


def test_interpretation_notes_survive_a_single_strategy(trained_cfg: dict[str, Any]) -> None:
    """With no random split to compare against, the gap note is omitted, not faked."""
    trained_cfg["validation"]["strategies"] = [trained_cfg["validation"]["primary_strategy"]]
    combined = ev.unified_summary(trained_cfg)
    comparison = ev.model_comparison_table(combined, trained_cfg)

    notes = ev.interpretation_notes(comparison, trained_cfg)

    assert not any("optimism" in note.lower() for note in notes)
    assert any("Spearman" in note for note in notes)


def test_run_evaluation_writes_the_comparison_artifacts(trained_cfg: dict[str, Any]) -> None:
    """Task 5 emits the comparison table as CSV and the summary as JSON."""
    write_torch_metrics(trained_cfg, ev.evaluation_strategy(trained_cfg))

    ev.run_evaluation(trained_cfg)

    metrics_dir = Path(trained_cfg["paths"]["metrics_dir"])
    comparison = read_table(metrics_dir / trained_cfg["files"]["model_comparison"])
    assert const.EVAL_COL_MODEL_FAMILY in comparison.columns
    assert const.TORCH_MODEL_NAME in set(comparison[const.COL_MODEL])

    record = json.loads(
        (metrics_dir / trained_cfg["files"]["evaluation_json"]).read_text(
            encoding=const.CONFIG_ENCODING
        )
    )
    assert record["random_seed"] == trained_cfg["project"]["random_seed"]
    assert record["reported_strategy"] == ev.evaluation_strategy(trained_cfg)
    assert record["targets"] == list(const.TARGETS.values())
    assert len(record["model_comparison"]) == len(comparison)
    assert record["interpretation"]
    assert record["model_families"][const.TORCH_MODEL_NAME] == const.MODEL_FAMILY_NEURAL


def test_evaluation_markdown_carries_the_comparative_section(trained_cfg: dict[str, Any]) -> None:
    """The human-readable report gains the Task 5 analysis without losing Task 3's."""
    ev.run_evaluation(trained_cfg)

    text = Path(
        trained_cfg["paths"]["metrics_dir"], trained_cfg["files"]["evaluation_markdown"]
    ).read_text(encoding=const.CONFIG_ENCODING)

    assert "# Evaluation summary" in text
    assert "# Comparative evaluation" in text
    assert "## Interpretation" in text
    assert const.EVAL_OPTIMISM_GAP_PREFIX in text
