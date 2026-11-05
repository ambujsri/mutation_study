"""Unit tests for ``src.attribution``: feature attribution for the selected model.

An attribution module is unusually easy to get quietly wrong — a shuffled
column that never reaches the estimator, an off-by-one in the PCA feature
names, or an aggregation that averages over the wrong axis all produce
plausible-looking tables. So the tests check the mechanism, not just the
shape of the output:

* a feature the model genuinely uses must score above one it ignores;
* the permutation must reach the fitted pipeline, scored on *validation* rows;
* the sign convention must be normalised so larger always means more
  important, whichever metric is configured; and
* every output column must be present and free of NaN where a number is
  promised.

Runs use the synthetic data from ``conftest.py`` with a cheap model and few
repeats: these assert correctness of the machinery, not the biology.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src import attribution as at
from src import data_processing as pp
from src import evaluate as ev
from src import train as tr
from src.common import constants as const
from src.common.config import validate_config
from src.common.io import read_table
from src.model import FeatureSet
from src.validation import generate_folds

# The synthetic data has a single labeled protein in one family, so a family-grouped
# training fold can hold one embedding vector; PCA then has zero variance and numpy
# warns while computing explained-variance ratios. Expected for this fixture only.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

TEST_N_SPLITS: int = 2  # synthetic data has only 2 families
TEST_N_COMPONENTS: int = 2
TEST_N_REPEATS: int = 2
TEST_TOP_K: int = 5


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture()
def attribution_cfg(cfg: dict[str, Any], raw_files: dict[str, Path]) -> dict[str, Any]:
    """Config with Phase 1 and Phase 3 run, and a cheap attribution setting."""
    cfg["validation"]["n_splits"] = TEST_N_SPLITS
    cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS
    cfg["models"]["candidates"] = {
        const.ModelName.GRADIENT_BOOSTING.value: {
            "n_estimators": 5, "learning_rate": 0.1, "max_depth": 2, "subsample": 1.0,
        }
    }
    cfg["models"]["save_fold_pipelines"] = True
    cfg["attribution"]["n_repeats"] = TEST_N_REPEATS
    cfg["attribution"]["top_k"] = TEST_TOP_K
    cfg["evaluation"]["figures"] = False
    validate_config(cfg)
    pp.run_preprocessing(cfg)
    tr.run_training(cfg)
    return cfg


@pytest.fixture()
def attribution_tables(attribution_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The computed attribution, shared by the reporting tests."""
    return at.compute_attribution(attribution_cfg)


@pytest.fixture()
def first_fold(attribution_cfg: dict[str, Any]) -> tuple[pd.DataFrame, FeatureSet, Any, Any]:
    """The data, features, first fold and its fitted pipelines.

    Rebuilt with the same call the module makes, so a test exercises the same
    folds the attribution does rather than a lookalike.
    """
    df, features = at.load_training_data(attribution_cfg)
    folds = generate_folds(
        df,
        const.SplitStrategy(attribution_cfg["validation"]["primary_strategy"]),
        attribution_cfg["validation"]["n_splits"],
        attribution_cfg["validation"]["shuffle"],
        attribution_cfg["project"]["random_seed"],
    )
    return df, features, folds[0], at.load_fold_pipelines(attribution_cfg)


# =============================================================================
# Loading what Phase 3 saved
# =============================================================================
def test_fold_pipelines_are_keyed_by_the_fold_in_their_filename(
    attribution_cfg: dict[str, Any],
) -> None:
    """Indices are parsed, not inferred from listing order, so a gap cannot shift them."""
    pipelines = at.load_fold_pipelines(attribution_cfg)

    assert set(pipelines) == set(const.TARGETS.values())
    for folds in pipelines.values():
        assert sorted(folds) == list(range(TEST_N_SPLITS))


def test_missing_training_metadata_is_reported(cfg: dict[str, Any]) -> None:
    """Running attribution before Phase 3 fails with an actionable message."""
    with pytest.raises(FileNotFoundError, match="src.train"):
        at.load_fold_pipelines(cfg)


def test_absent_fold_pipelines_are_reported(attribution_cfg: dict[str, Any]) -> None:
    """``save_fold_pipelines: false`` is a configuration error here, named as one."""
    metadata_path = Path(
        attribution_cfg["paths"]["models_dir"], attribution_cfg["files"]["training_metadata"]
    )
    metadata = json.loads(metadata_path.read_text(encoding=const.CONFIG_ENCODING))
    metadata["fold_pipelines"] = []
    metadata_path.write_text(json.dumps(metadata), encoding=const.CONFIG_ENCODING)

    with pytest.raises(FileNotFoundError, match="save_fold_pipelines"):
        at.load_fold_pipelines(attribution_cfg)


# =============================================================================
# Permutation units and grouping
# =============================================================================
def test_embeddings_form_one_permutation_unit_by_default(
    attribution_cfg: dict[str, Any],
) -> None:
    """All embedding columns are shuffled together; each tabular column is its own unit."""
    _, features = at.load_training_data(attribution_cfg)

    units = at.permutation_units(features, attribution_cfg)

    assert units[const.GROUP_EMBEDDINGS] == features.embedding
    assert len(units) == len(features.tabular) + 1
    assert all(units[column] == [column] for column in features.tabular)


def test_embeddings_can_be_permuted_individually(attribution_cfg: dict[str, Any]) -> None:
    """The block flag is real: false gives one unit per embedding column."""
    _, features = at.load_training_data(attribution_cfg)
    attribution_cfg["attribution"]["group_embeddings_as_block"] = False

    units = at.permutation_units(features, attribution_cfg)

    assert const.GROUP_EMBEDDINGS not in units
    assert len(units) == len(features.tabular) + len(features.embedding)


def test_every_unit_is_assigned_a_known_feature_group(attribution_cfg: dict[str, Any]) -> None:
    """No tabular column falls through to 'unassigned', which would hide it in the figure."""
    _, features = at.load_training_data(attribution_cfg)
    units = at.permutation_units(features, attribution_cfg)

    groups = {unit: at.feature_group_of(unit, units) for unit in units}

    assert groups[const.GROUP_EMBEDDINGS] == const.GROUP_EMBEDDINGS
    assert const.ATTR_GROUP_UNKNOWN not in set(groups.values())
    assert set(groups.values()) <= set(const.FEATURE_GROUPS)


def test_an_unrecognised_column_is_flagged_not_guessed() -> None:
    """A column outside every declared group is reported as unassigned."""
    assert at.feature_group_of("not_a_real_feature", {}) == const.ATTR_GROUP_UNKNOWN


# =============================================================================
# Permutation importance mechanics
# =============================================================================
def test_a_used_feature_outscores_an_ignored_one(
    attribution_cfg: dict[str, Any],
    first_fold: tuple[pd.DataFrame, FeatureSet, Any, Any],
) -> None:
    """The headline guarantee: shuffling a feature the model relies on costs more.

    A constant column the model never saw is added to the frame and declared
    as a unit. Permuting it cannot reach a prediction, so its importance must
    be exactly zero, while the real columns between them must move the score:
    a run where *nothing* registered would mean the shuffle never reached the
    fitted pipeline at all, which is the failure this guards against.

    Which individual column matters is deliberately not asserted. On 24
    synthetic rows a five-tree stump may simply never split on a given
    feature, and a test that depended on it would be testing the fixture.
    """
    df, features, fold, pipelines = first_fold
    target = const.TARGET_STABILITY
    rows = df.iloc[fold.val_idx][features.all].copy()
    rows["unused_constant"] = 1.0
    units = {column: [column] for column in features.tabular}
    units["unused_constant"] = ["unused_constant"]

    scores = at.permutation_importance_fold(
        pipelines[target][fold.index],
        rows,
        df.iloc[fold.val_idx][target].to_numpy(dtype=float),
        units,
        const.Metric.SPEARMAN,
        TEST_N_REPEATS,
        attribution_cfg["project"]["random_seed"],
    )

    assert scores["unused_constant"] == pytest.approx(0.0)
    used = {name: value for name, value in scores.items() if name != "unused_constant"}
    assert any(value != pytest.approx(0.0) for value in used.values())
    assert max(used.values()) > scores["unused_constant"]


def test_importance_sign_is_normalised_across_metric_polarity(
    attribution_cfg: dict[str, Any],
    first_fold: tuple[pd.DataFrame, FeatureSet, Any, Any],
) -> None:
    """A real feature scores positive under both a higher-is-better and a lower-is-better metric.

    Spearman improves when a feature is intact; RMSE worsens. Both must come
    back positive, or the figure's "larger is more important" claim is false
    for half the configurations.
    """
    df, features, fold, pipelines = first_fold
    target = const.TARGET_STABILITY
    rows = df.iloc[fold.val_idx][features.all]
    y_true = df.iloc[fold.val_idx][target].to_numpy(dtype=float)
    units = {column: [column] for column in features.tabular}

    scored = {
        metric: at.permutation_importance_fold(
            pipelines[target][fold.index], rows, y_true, units, metric, TEST_N_REPEATS,
            attribution_cfg["project"]["random_seed"],
        )
        for metric in (const.Metric.SPEARMAN, const.Metric.RMSE)
    }

    # The single most important feature by each metric must be positive.
    for metric, scores in scored.items():
        assert max(scores.values()) > 0, metric.value


def test_impurity_names_pca_components_separately(
    first_fold: tuple[pd.DataFrame, FeatureSet, Any, Any],
) -> None:
    """Post-PCA inputs are labelled as components, never as embedding column names."""
    _, features, _, pipelines = first_fold
    pipeline = pipelines[const.TARGET_STABILITY][0]

    scores = at.impurity_importance_fold(pipeline, features)

    assert set(features.tabular) <= set(scores)
    components = [name for name in scores if name.startswith("embedding_pc")]
    assert len(components) == TEST_N_COMPONENTS
    assert not set(features.embedding) & set(scores)


def test_impurity_is_skipped_for_an_estimator_without_it() -> None:
    """Ridge has no feature_importances_; coefficients are not substituted for them."""
    pipeline = Pipeline([("model", Ridge())])

    assert at.impurity_importance_fold(pipeline, FeatureSet(["a"], [])) == {}


# =============================================================================
# Aggregation
# =============================================================================
def test_aggregation_averages_over_folds_and_reports_spread() -> None:
    """Mean and standard deviation are taken across folds, per feature."""
    per_fold = {0: {"a": 1.0, "b": 0.0}, 1: {"a": 3.0, "b": 0.0}}

    table = at.aggregate_folds(per_fold, const.TARGET_STABILITY, const.ImportanceKind.PERMUTATION,
                               {"a": ["a"], "b": ["b"]})

    row_a = table[table[const.ATTR_COL_FEATURE] == "a"].iloc[0]
    assert row_a[const.ATTR_COL_MEAN] == pytest.approx(2.0)
    assert row_a[const.ATTR_COL_STD] == pytest.approx(1.0)
    assert row_a[const.ATTR_COL_N_FOLDS] == 2
    assert row_a[const.ATTR_COL_RANK] == 1


def test_shares_sum_to_one_over_the_positive_importances() -> None:
    """``importance_share`` is a share of attributable signal, so it normalises to 1.

    Negative means are noise around zero and are excluded from the
    denominator; including them would inflate every other feature's share.
    """
    per_fold = {0: {"a": 3.0, "b": 1.0, "c": -0.5}, 1: {"a": 3.0, "b": 1.0, "c": -0.5}}

    table = at.aggregate_folds(per_fold, const.TARGET_STABILITY, const.ImportanceKind.PERMUTATION,
                               {})

    assert table[const.ATTR_COL_SHARE].sum() == pytest.approx(1.0)
    negative = table[table[const.ATTR_COL_FEATURE] == "c"].iloc[0]
    assert negative[const.ATTR_COL_SHARE] == pytest.approx(0.0)


def test_aggregation_of_nothing_is_empty_not_a_crash() -> None:
    """A target with no usable fold yields an empty frame, handled upstream."""
    assert at.aggregate_folds({}, const.TARGET_STABILITY,
                              const.ImportanceKind.PERMUTATION, {}).empty


def test_group_importances_sum_within_each_group(
    attribution_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The group table is the per-feature table summed by group, per target and kind."""
    per_feature, per_group = attribution_tables

    for _, row in per_group.iterrows():
        members = per_feature[
            (per_feature[const.COL_TARGET] == row[const.COL_TARGET])
            & (per_feature[const.ATTR_COL_KIND] == row[const.ATTR_COL_KIND])
            & (per_feature[const.ATTR_COL_GROUP] == row[const.ATTR_COL_GROUP])
        ]
        assert row[const.ATTR_COL_MEAN] == pytest.approx(members[const.ATTR_COL_MEAN].sum())
        assert row["n_features"] == len(members)


# =============================================================================
# Output contract
# =============================================================================
def test_output_table_has_every_column_and_no_missing_numbers(
    attribution_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Every promised column is present, and the numeric ones are finite."""
    per_feature, per_group = attribution_tables

    expected = {
        const.COL_TARGET,
        const.ATTR_COL_KIND,
        const.ATTR_COL_FEATURE,
        const.ATTR_COL_GROUP,
        const.ATTR_COL_MEAN,
        const.ATTR_COL_STD,
        const.ATTR_COL_SHARE,
        const.ATTR_COL_N_FOLDS,
        const.ATTR_COL_RANK,
    }
    assert expected <= set(per_feature.columns)
    for column in (const.ATTR_COL_MEAN, const.ATTR_COL_STD, const.ATTR_COL_SHARE):
        assert np.isfinite(per_feature[column].to_numpy(dtype=float)).all(), column
    assert not per_group.empty
    # A feature name repeats across targets and kinds, but must appear once
    # within a (target, kind) block or the ranking is double-counting it.
    for _, block in per_feature.groupby([const.COL_TARGET, const.ATTR_COL_KIND]):
        assert block[const.ATTR_COL_FEATURE].is_unique


def test_both_targets_and_both_kinds_are_covered(
    attribution_tables: tuple[pd.DataFrame, pd.DataFrame], attribution_cfg: dict[str, Any]
) -> None:
    """Attribution is reported for every target under every configured kind."""
    per_feature, _ = attribution_tables

    assert set(per_feature[const.COL_TARGET]) == set(const.TARGETS.values())
    assert set(per_feature[const.ATTR_COL_KIND]) == set(attribution_cfg["attribution"]["kinds"])


def test_ranks_restart_at_one_for_each_target_and_kind(
    attribution_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Rank 1 is the most important feature within its own block, not across blocks."""
    per_feature, _ = attribution_tables

    for (_, _), block in per_feature.groupby([const.COL_TARGET, const.ATTR_COL_KIND]):
        assert sorted(block[const.ATTR_COL_RANK]) == list(range(1, len(block) + 1))
        ordered = block.sort_values(const.ATTR_COL_RANK)
        assert ordered[const.ATTR_COL_MEAN].is_monotonic_decreasing


def test_run_writes_tables_markdown_and_figures(attribution_cfg: dict[str, Any]) -> None:
    """The stage emits all Task 7 artifacts, figures included."""
    attribution_cfg["evaluation"]["figures"] = True

    per_feature, _ = at.run_attribution(attribution_cfg)

    metrics_dir = Path(attribution_cfg["paths"]["metrics_dir"])
    written = read_table(metrics_dir / attribution_cfg["files"]["feature_importances"])
    assert len(written) == len(per_feature)
    assert (
        metrics_dir / attribution_cfg["files"]["feature_group_importances"]
    ).is_file()
    markdown = (metrics_dir / attribution_cfg["files"]["attribution_markdown"]).read_text(
        encoding=const.CONFIG_ENCODING
    )
    assert markdown.startswith("# Feature attribution")
    assert "## By feature group" in markdown

    figures_dir = Path(attribution_cfg["paths"]["figures_dir"])
    for key in const.ATTR_FIGURE_FILE_KEYS:
        path = figures_dir / attribution_cfg["files"][key]
        assert path.is_file() and path.stat().st_size > 0, key


def test_figures_honour_the_figures_switch(
    attribution_tables: tuple[pd.DataFrame, pd.DataFrame], attribution_cfg: dict[str, Any]
) -> None:
    """``evaluation.figures: false`` suppresses them, as for every other figure."""
    per_feature, per_group = attribution_tables
    attribution_cfg["evaluation"]["figures"] = False

    assert ev.make_attribution_figures(per_feature, per_group, attribution_cfg) == {}


def test_markdown_states_the_method_not_just_the_numbers(
    attribution_tables: tuple[pd.DataFrame, pd.DataFrame], attribution_cfg: dict[str, Any]
) -> None:
    """A reader must be able to tell what was measured and on which rows."""
    per_feature, per_group = attribution_tables

    markdown = at.to_markdown(per_feature, per_group, attribution_cfg)

    assert "validation" in markdown
    assert attribution_cfg["attribution"]["scoring_metric"] in markdown
    assert str(attribution_cfg["attribution"]["n_repeats"]) in markdown


def test_attribution_is_reproducible(attribution_cfg: dict[str, Any]) -> None:
    """Seed 20261102 fixes the permutation order, so two runs agree exactly."""
    first, _ = at.compute_attribution(attribution_cfg)
    second, _ = at.compute_attribution(attribution_cfg)

    pd.testing.assert_frame_equal(first, second)
