"""Unit and integration tests for ``src.ablation`` (Task 6).

Subsets are resolved and cross-validated on the synthetic fixtures, with a
cheap model so the suite stays fast. The important guarantees tested here are
that subsets really do restrict the columns the model sees, that single-block
subsets work at all, and that baseline deltas are computed per target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src import ablation as ab
from src import data_processing as pp
from src import evaluate as ev
from src import model as md
from src import train as tr
from src.common import constants as const
from src.common.config import validate_config
from src.common.io import read_table

TEST_N_SPLITS: int = 2
TEST_N_COMPONENTS: int = 2

pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")


@pytest.fixture()
def ablation_cfg(cfg: dict[str, Any], raw_files: dict[str, Path]) -> dict[str, Any]:
    """Config with Phase 1 run, a cheap model, and a small set of ablation subsets."""
    cfg["validation"]["n_splits"] = TEST_N_SPLITS
    cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS
    cfg["models"]["candidates"] = {const.ModelName.RIDGE.value: {"alpha": 1.0}}
    cfg["ablation"]["model"] = const.ModelName.RIDGE.value
    cfg["ablation"]["subsets"] = {
        "tabular_only": [const.GROUP_PHYSICOCHEMICAL, const.GROUP_SEQUENCE],
        "embeddings_only": [const.GROUP_EMBEDDINGS],
        "full": [
            const.GROUP_PHYSICOCHEMICAL,
            const.GROUP_SEQUENCE,
            const.GROUP_STRUCTURAL,
            const.GROUP_DELTAS,
            const.GROUP_EMBEDDINGS,
        ],
    }
    cfg["ablation"]["baseline"] = "full"
    validate_config(cfg)
    pp.run_preprocessing(cfg)
    return cfg


@pytest.fixture()
def manifest_features(ablation_cfg: dict[str, Any]) -> md.FeatureSet:
    """Feature lists from the synthetic Phase 1 run."""
    return tr.load_training_data(ablation_cfg)[1]


# =============================================================================
# Subset resolution
# =============================================================================
def test_feature_groups_cover_manifest_tabular(manifest_features: md.FeatureSet) -> None:
    """Every tabular feature Phase 1 produces belongs to exactly one declared group."""
    grouped = [
        column
        for group, columns in const.FEATURE_GROUPS.items()
        if group != const.GROUP_EMBEDDINGS
        for column in columns
    ]
    assert len(grouped) == len(set(grouped)), "groups must not overlap"
    assert set(manifest_features.tabular) <= set(grouped), "a manifest feature is in no group"


def test_resolve_subset_splits_tabular_and_embedding(manifest_features: md.FeatureSet) -> None:
    """Embedding columns land in the embedding block; everything else is tabular."""
    subset = ab.resolve_subset(
        "mixed", [const.GROUP_PHYSICOCHEMICAL, const.GROUP_EMBEDDINGS], manifest_features
    )

    assert subset.features.embedding == list(const.EMBEDDING_COLUMNS)
    assert set(subset.features.tabular) == set(const.FEATURE_GROUPS[const.GROUP_PHYSICOCHEMICAL])
    assert not [c for c in subset.features.tabular if c.startswith(const.EMBEDDING_PREFIX)]


def test_resolve_subset_preserves_manifest_order(manifest_features: md.FeatureSet) -> None:
    """Column order follows the manifest, so runs are deterministic."""
    subset = ab.resolve_subset("all_tabular", [g for g in const.FEATURE_GROUPS], manifest_features)

    assert subset.features.tabular == manifest_features.tabular


def test_resolve_subset_skips_columns_absent_from_manifest(manifest_features: md.FeatureSet) -> None:
    """A group listing optional columns (e.g. the ordinal SS encoding) resolves to what exists."""
    subset = ab.resolve_subset("structural", [const.GROUP_STRUCTURAL], manifest_features)

    assert const.SS_ORDINAL_COLUMN in const.FEATURE_GROUPS[const.GROUP_STRUCTURAL]
    assert const.SS_ORDINAL_COLUMN not in subset.features.tabular  # one-hot encoding was used
    assert const.COL_CONSERVATION in subset.features.tabular


def test_resolve_subset_empty_raises(manifest_features: md.FeatureSet) -> None:
    """A subset that resolves to nothing is an error, not an empty silent run."""
    empty = md.FeatureSet([], [])

    with pytest.raises(ValueError, match="resolves to no available columns"):
        ab.resolve_subset("nothing", [const.GROUP_PHYSICOCHEMICAL], empty)


def test_build_subsets_returns_all_configured(
    ablation_cfg: dict[str, Any], manifest_features: md.FeatureSet
) -> None:
    """Every configured subset is resolved, in config order."""
    subsets = ab.build_subsets(ablation_cfg, manifest_features)

    assert [s.name for s in subsets] == list(ablation_cfg["ablation"]["subsets"])


# =============================================================================
# Model resolution
# =============================================================================
def test_resolve_models_honours_explicit_choice(ablation_cfg: dict[str, Any]) -> None:
    """A pinned ``ablation.model`` is used for both targets."""
    models = ab.resolve_models(ablation_cfg)

    assert set(models) == set(const.TARGETS.values())
    assert all(model is const.ModelName.RIDGE for model in models.values())


def test_resolve_models_requires_phase3_when_null(ablation_cfg: dict[str, Any]) -> None:
    """With ``ablation.model: null`` and no Phase 3 metadata, the error says what to run."""
    ablation_cfg["ablation"]["model"] = None

    with pytest.raises(FileNotFoundError, match="src.train"):
        ab.resolve_models(ablation_cfg)


def test_resolve_models_reads_phase3_selection(ablation_cfg: dict[str, Any]) -> None:
    """With ``null``, each target inherits the model Phase 3 selected for it."""
    tr.run_training(ablation_cfg)
    ablation_cfg["ablation"]["model"] = None

    models = ab.resolve_models(ablation_cfg)

    metadata = json.loads(
        Path(ablation_cfg["paths"]["models_dir"], ablation_cfg["files"]["training_metadata"]).read_text()
    )
    for target, model in models.items():
        assert model.value == metadata["selection"][target][const.COL_MODEL]


# =============================================================================
# Leakage guarantee for single-block subsets
# =============================================================================
def test_embeddings_only_subset_builds_single_block_pipeline(
    ablation_cfg: dict[str, Any], manifest_features: md.FeatureSet
) -> None:
    """A subset with no tabular columns omits that block instead of fitting on zero features."""
    subset = ab.resolve_subset("emb", [const.GROUP_EMBEDDINGS], manifest_features)

    preprocessor = md.build_preprocessor(ablation_cfg, subset.features)

    assert [name for name, _, _ in preprocessor.transformers] == [const.BLOCK_EMBEDDING]


def test_tabular_only_subset_omits_embedding_block(
    ablation_cfg: dict[str, Any], manifest_features: md.FeatureSet
) -> None:
    """A subset without embeddings carries no PCA step at all."""
    subset = ab.resolve_subset("tab", [const.GROUP_PHYSICOCHEMICAL], manifest_features)

    preprocessor = md.build_preprocessor(ablation_cfg, subset.features)

    assert [name for name, _, _ in preprocessor.transformers] == [const.BLOCK_TABULAR]


def test_subset_model_sees_only_its_columns(
    ablation_cfg: dict[str, Any], manifest_features: md.FeatureSet
) -> None:
    """The fitted estimator's input width matches the subset, proving restriction is real."""
    df = tr.load_training_data(ablation_cfg)[0]
    subset = ab.resolve_subset("phys", [const.GROUP_PHYSICOCHEMICAL], manifest_features)

    pipeline = md.build_pipeline(ablation_cfg, const.ModelName.RIDGE, subset.features)
    pipeline.fit(df[subset.features.all], df[const.TARGET_STABILITY])

    assert pipeline.named_steps[const.STEP_MODEL].n_features_in_ == len(subset.features.tabular)


# =============================================================================
# Ablation run and reporting
# =============================================================================
@pytest.fixture()
def ablation_table(ablation_cfg: dict[str, Any]) -> pd.DataFrame:
    """Result of a full ablation run on synthetic data."""
    return ab.run_ablation(ablation_cfg)


def test_ablation_covers_subsets_and_targets(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """One row per subset x target, each with pooled metrics over every labeled row."""
    n_subsets = len(ablation_cfg["ablation"]["subsets"])

    assert len(ablation_table) == n_subsets * len(const.TARGETS)
    assert set(ablation_table[const.COL_TARGET]) == set(const.TARGETS.values())
    for metric in ablation_cfg["validation"]["metrics"]:
        assert f"{const.POOLED_PREFIX}{metric}" in ablation_table.columns
        assert f"{const.ABL_DELTA_PREFIX}{metric}" in ablation_table.columns


def test_ablation_records_feature_counts(ablation_table: pd.DataFrame) -> None:
    """Reported feature counts distinguish the subsets, including the single-block ones."""
    rows = ablation_table.set_index([const.ABL_COL_SUBSET, const.COL_TARGET])
    emb = rows.loc[("embeddings_only", const.TARGET_STABILITY)]
    tab = rows.loc[("tabular_only", const.TARGET_STABILITY)]

    assert emb[const.ABL_COL_N_TABULAR] == 0
    assert emb[const.ABL_COL_N_EMBEDDING] == const.EMBEDDING_DIM
    assert tab[const.ABL_COL_N_EMBEDDING] == 0
    assert tab[const.ABL_COL_N_TABULAR] > 0


def test_baseline_delta_is_zero_for_baseline_subset(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """The baseline subset's own delta is exactly zero for every metric and target."""
    baseline = ablation_cfg["ablation"]["baseline"]
    rows = ablation_table[ablation_table[const.ABL_COL_SUBSET] == baseline]

    assert not rows.empty
    for metric in ablation_cfg["validation"]["metrics"]:
        assert (rows[f"{const.ABL_DELTA_PREFIX}{metric}"].abs() < 1e-12).all()


def test_baseline_delta_matches_manual_difference(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """Deltas are the subset's metric minus the baseline's, computed per target."""
    baseline = ablation_cfg["ablation"]["baseline"]
    metric = f"{const.POOLED_PREFIX}{const.Metric.SPEARMAN.value}"
    delta = f"{const.ABL_DELTA_PREFIX}{const.Metric.SPEARMAN.value}"

    for target in const.TARGETS.values():
        rows = ablation_table[ablation_table[const.COL_TARGET] == target]
        reference = rows.loc[rows[const.ABL_COL_SUBSET] == baseline, metric].iloc[0]
        for _, row in rows.iterrows():
            assert row[delta] == pytest.approx(row[metric] - reference)


def test_ablation_markdown_lists_every_subset(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """The Markdown report names each subset and both targets, with the baseline stated."""
    markdown = ab.to_markdown(ablation_table, ablation_cfg)

    for subset in ablation_cfg["ablation"]["subsets"]:
        assert subset in markdown
    for target in const.TARGETS.values():
        assert f"## {target}" in markdown
    assert ablation_cfg["ablation"]["baseline"] in markdown


def test_export_writes_csv_and_markdown(ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]) -> None:
    """Both artifacts are written and the CSV round-trips."""
    paths = ab.export_artifacts(ablation_table, ablation_cfg)

    pd.testing.assert_frame_equal(read_table(paths["ablation_summary"]), ablation_table, check_dtype=False)
    assert paths["ablation_summary_markdown"].read_text().startswith("# Feature-subset ablation")


def test_ablation_uses_configured_strategy(ablation_cfg: dict[str, Any]) -> None:
    """The recorded strategy is the configured one, not silently the primary default."""
    ablation_cfg["ablation"]["strategy"] = const.SplitStrategy.RANDOM.value

    table = ab.run_ablation(ablation_cfg)

    assert set(table[const.CV_COL_STRATEGY]) == {const.SplitStrategy.RANDOM.value}


def test_ablation_defaults_to_primary_strategy(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """With ``ablation.strategy: null`` the primary CV strategy is used."""
    assert ablation_cfg["ablation"]["strategy"] is None
    assert set(ablation_table[const.CV_COL_STRATEGY]) == {ablation_cfg["validation"]["primary_strategy"]}


def test_ablation_is_deterministic(ablation_cfg: dict[str, Any]) -> None:
    """Two runs give identical metrics (same folds, seed and models)."""
    first = ab.run_ablation(ablation_cfg)
    second = ab.run_ablation(ablation_cfg)

    pd.testing.assert_frame_equal(first, second)


# =============================================================================
# Attribution reporting and the figure (assignment Task 6)
# =============================================================================
def test_markdown_reports_a_delta_for_both_headline_metrics(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """Spearman and R2 each get a level and a delta.

    A feature group can matter for the ordering and not for the magnitudes, or
    the reverse, so reporting only the selection metric would hide half of the
    attribution.
    """
    markdown = ab.to_markdown(ablation_table, ablation_cfg)

    for metric in const.ABL_HEADLINE_METRICS:
        assert f"{const.POOLED_PREFIX}{metric.value}" in markdown
        assert f"{const.ABL_DELTA_PREFIX}{metric.value}" in markdown
    assert f"{const.POOLED_PREFIX}{const.Metric.PEARSON.value}" in markdown


def test_markdown_survives_a_reduced_metric_set(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """Columns the configured metric list does not produce are omitted, not crashed on."""
    reduced = ablation_table.drop(
        columns=[c for c in ablation_table.columns if const.Metric.PEARSON.value in c]
    )

    markdown = ab.to_markdown(reduced, ablation_cfg)

    assert f"{const.POOLED_PREFIX}{const.Metric.PEARSON.value}" not in markdown
    assert f"{const.POOLED_PREFIX}{const.Metric.SPEARMAN.value}" in markdown


def test_ablation_figure_is_written(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """The attribution figure lands under its configured name."""
    written = ev.make_ablation_figures(ablation_table, ablation_cfg)

    assert set(written) == set(const.ABLATION_FIGURE_FILE_KEYS)
    path = written["fig_ablation_drop"]
    assert path == Path(ablation_cfg["paths"]["figures_dir"], ablation_cfg["files"]["fig_ablation_drop"])
    assert path.is_file() and path.stat().st_size > 0


def test_ablation_figure_honours_the_figures_switch(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """``evaluation.figures: false`` suppresses it, as for every other figure."""
    ablation_cfg["evaluation"]["figures"] = False

    assert ev.make_ablation_figures(ablation_table, ablation_cfg) == {}


def test_ablation_figure_skipped_on_an_empty_table(ablation_cfg: dict[str, Any]) -> None:
    """Nothing to attribute means no figure, rather than an empty one."""
    assert ev.make_ablation_figures(pd.DataFrame(), ablation_cfg) == {}


def test_export_also_writes_the_figure(
    ablation_table: pd.DataFrame, ablation_cfg: dict[str, Any]
) -> None:
    """The stage emits all three Task 6 artifacts in one call."""
    paths = ab.export_artifacts(ablation_table, ablation_cfg)

    assert {"ablation_summary", "ablation_summary_markdown", "fig_ablation_drop"} <= set(paths)
    assert all(path.exists() for path in paths.values())
