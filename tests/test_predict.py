"""Unit and integration tests for ``src.predict`` (Phase 4).

Artifacts are produced by running the real Phase 1 and Phase 3 on the synthetic
raw tables from ``conftest.py``, so these tests exercise the actual handoff
between phases rather than a mocked one.

The scoring tests use hand-built frames with known values, so the arithmetic of
the multi-objective score, the sign convention on ``stability_ddG`` and the
uncertainty penalty are each verified against values computed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import data_processing as pp
from src import predict as pr
from src import train as tr
from src.common import constants as const
from src.common.config import validate_config
from src.common.io import read_table

TEST_N_SPLITS: int = 2
TEST_N_COMPONENTS: int = 3
TEST_TOP_K: int = 4

# The synthetic fixture has one labeled protein in one family, so a family-grouped
# training fold can hold a single embedding vector and PCA then sees zero variance.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")


# =============================================================================
# Fixtures & helpers
# =============================================================================
@pytest.fixture()
def phase4_cfg(cfg: dict[str, Any], raw_files: dict[str, Path]) -> dict[str, Any]:
    """Config for fast end-to-end Phase 1 + 3 on synthetic data, ready for Phase 4."""
    cfg["validation"]["n_splits"] = TEST_N_SPLITS
    cfg["features"]["embedding_reduction"]["n_components"] = TEST_N_COMPONENTS
    cfg["models"]["candidates"] = {const.ModelName.RIDGE.value: {"alpha": 1.0}}
    cfg["prioritization"]["top_k"] = TEST_TOP_K
    cfg["prioritization"]["min_fold_models"] = TEST_N_SPLITS
    validate_config(cfg)
    pp.run_preprocessing(cfg)
    tr.run_training(cfg)
    return cfg


@pytest.fixture()
def artifacts(phase4_cfg: dict[str, Any]) -> pr.TrainedArtifacts:
    """Artifacts written by the synthetic Phase 3 run."""
    return pr.load_artifacts(phase4_cfg)


@pytest.fixture()
def candidate_table(phase4_cfg: dict[str, Any], artifacts: pr.TrainedArtifacts) -> pd.DataFrame:
    """Processed candidate table for the synthetic run."""
    return pr.load_candidates(phase4_cfg, artifacts.features)


@pytest.fixture()
def train_table(phase4_cfg: dict[str, Any]) -> pd.DataFrame:
    """Processed training table, used as the reference for distribution-shift flags."""
    from src.common.config import resolve_path

    return read_table(resolve_path(phase4_cfg, "processed_dir", "processed_train"))


def make_scored(
    ddg: list[float],
    func: list[float],
    std_ddg: list[float] | None = None,
    std_func: list[float] | None = None,
    flags: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal prediction frame with known values for scoring tests."""
    n = len(ddg)
    return pd.DataFrame(
        {
            const.COL_MUTATION_ID: [f"CAND{i:04d}" for i in range(1, n + 1)],
            const.COL_PROTEIN_ID: [f"P{i}" for i in range(n)],
            const.COL_PROTEIN_FAMILY: ["F0"] * n,
            const.COL_WT_AA: ["A"] * n,
            const.COL_POSITION: range(1, n + 1),
            const.COL_MUTANT_AA: ["V"] * n,
            const.COL_SECONDARY_STRUCTURE: [const.SS_CATEGORIES[0]] * n,
            const.COL_CONSERVATION: [0.5] * n,
            const.COL_SOLVENT_ACCESSIBILITY: [0.5] * n,
            const.COL_DIST_ACTIVE_SITE: [10.0] * n,
            f"{const.PRED_PREFIX}{const.TARGET_STABILITY}": ddg,
            f"{const.PRED_PREFIX}{const.TARGET_FUNCTION}": func,
            f"{const.PRED_STD_PREFIX}{const.TARGET_STABILITY}": std_ddg or [0.0] * n,
            f"{const.PRED_STD_PREFIX}{const.TARGET_FUNCTION}": std_func or [0.0] * n,
            const.COL_OOD_FLAG: flags or [const.OOD_IN_DISTRIBUTION] * n,
        }
    )


# =============================================================================
# 1. Artifact loading
# =============================================================================
def test_load_artifacts_groups_fold_pipelines_by_target(artifacts: pr.TrainedArtifacts) -> None:
    """Every target gets its own fold ensemble, plus the final refit model."""
    assert set(artifacts.fold_pipelines) == set(const.TARGETS.values())
    for pipelines in artifacts.fold_pipelines.values():
        assert len(pipelines) == TEST_N_SPLITS
    assert set(artifacts.final_pipelines) == set(const.TARGETS.values())
    assert artifacts.features.embedding == list(const.EMBEDDING_COLUMNS)


def test_load_artifacts_requires_phase3(phase4_cfg: dict[str, Any]) -> None:
    """A missing metadata file points at the Phase 3 run that should have produced it."""
    Path(phase4_cfg["paths"]["models_dir"], phase4_cfg["files"]["training_metadata"]).unlink()

    with pytest.raises(FileNotFoundError, match="run Phase 3"):
        pr.load_artifacts(phase4_cfg)


def test_load_artifacts_detects_deleted_pipeline(phase4_cfg: dict[str, Any]) -> None:
    """A pipeline listed in metadata but absent on disk fails loudly, not silently."""
    models_dir = Path(phase4_cfg["paths"]["models_dir"])
    metadata = json.loads((models_dir / phase4_cfg["files"]["training_metadata"]).read_text())
    (models_dir / metadata["fold_pipelines"][0]).unlink()

    with pytest.raises(FileNotFoundError, match="listed in metadata is missing"):
        pr.load_artifacts(phase4_cfg)


def test_load_artifacts_enforces_minimum_ensemble_size(phase4_cfg: dict[str, Any]) -> None:
    """Too few fold models to measure a spread is an error, not a silent zero uncertainty."""
    phase4_cfg["prioritization"]["min_fold_models"] = TEST_N_SPLITS + 1

    with pytest.raises(ValueError, match="need at least"):
        pr.load_artifacts(phase4_cfg)


def test_load_candidates_detects_missing_features(
    phase4_cfg: dict[str, Any], artifacts: pr.TrainedArtifacts
) -> None:
    """A candidate table lacking a training feature is rejected before inference."""
    path = Path(phase4_cfg["paths"]["processed_dir"], phase4_cfg["files"]["processed_candidates"])
    read_table(path).drop(columns=const.COL_CONSERVATION).to_parquet(path, index=False)

    with pytest.raises(KeyError, match="lacks 1 training features"):
        pr.load_candidates(phase4_cfg, artifacts.features)


# =============================================================================
# 2. Ensemble inference & uncertainty
# =============================================================================
def test_predict_ensemble_adds_columns_for_both_targets(
    candidate_table: pd.DataFrame, artifacts: pr.TrainedArtifacts, phase4_cfg: dict[str, Any]
) -> None:
    """Each target gets a point estimate, a fold spread, a final-model prediction and a count."""
    out = pr.predict_ensemble(candidate_table, artifacts, phase4_cfg)

    assert len(out) == len(candidate_table)
    for target in const.TARGETS.values():
        assert np.isfinite(out[f"{const.PRED_PREFIX}{target}"]).all()
        assert (out[f"{const.PRED_STD_PREFIX}{target}"] >= 0).all()
        assert np.isfinite(out[f"{const.PRED_FINAL_PREFIX}{target}"]).all()
        assert (out[f"{const.COL_N_FOLD_MODELS}_{target}"] == TEST_N_SPLITS).all()


def test_ensemble_mean_equals_mean_of_fold_predictions(
    candidate_table: pd.DataFrame, artifacts: pr.TrainedArtifacts, phase4_cfg: dict[str, Any]
) -> None:
    """The point estimate and spread are exactly the mean and sample sd of the fold models."""
    target = const.TARGET_STABILITY
    inputs = candidate_table[artifacts.features.all]
    expected = np.vstack([p.predict(inputs) for p in artifacts.fold_pipelines[target]])

    out = pr.predict_ensemble(candidate_table, artifacts, phase4_cfg)

    np.testing.assert_allclose(out[f"{const.PRED_PREFIX}{target}"], expected.mean(axis=0))
    np.testing.assert_allclose(out[f"{const.PRED_STD_PREFIX}{target}"], expected.std(axis=0, ddof=1))


def test_point_estimate_final_model_uses_refit_model(
    candidate_table: pd.DataFrame, artifacts: pr.TrainedArtifacts, phase4_cfg: dict[str, Any]
) -> None:
    """With ``point_estimate: final_model`` the reported prediction is the refit model's."""
    phase4_cfg["prioritization"]["point_estimate"] = const.PointEstimate.FINAL_MODEL.value

    out = pr.predict_ensemble(candidate_table, artifacts, phase4_cfg)

    for target in const.TARGETS.values():
        pd.testing.assert_series_equal(
            out[f"{const.PRED_PREFIX}{target}"],
            out[f"{const.PRED_FINAL_PREFIX}{target}"],
            check_names=False,
        )


def test_point_estimate_final_model_requires_refit(
    candidate_table: pd.DataFrame, artifacts: pr.TrainedArtifacts, phase4_cfg: dict[str, Any]
) -> None:
    """Asking for the final model when Phase 3 skipped the refit is an explicit error."""
    phase4_cfg["prioritization"]["point_estimate"] = const.PointEstimate.FINAL_MODEL.value
    stripped = pr.TrainedArtifacts(
        artifacts.features, artifacts.fold_pipelines, {}, artifacts.selection, artifacts.metadata
    )

    with pytest.raises(ValueError, match="refit_on_full_data"):
        pr.predict_ensemble(candidate_table, stripped, phase4_cfg)


def test_uncertainty_is_zero_for_identical_fold_models(
    candidate_table: pd.DataFrame, artifacts: pr.TrainedArtifacts, phase4_cfg: dict[str, Any]
) -> None:
    """Agreeing fold models give zero spread, so the metric reflects disagreement only."""
    target = const.TARGET_STABILITY
    cloned = dict(artifacts.fold_pipelines)
    cloned[target] = [artifacts.fold_pipelines[target][0]] * TEST_N_SPLITS

    out = pr.predict_ensemble(
        candidate_table,
        pr.TrainedArtifacts(
            artifacts.features, cloned, artifacts.final_pipelines, artifacts.selection, artifacts.metadata
        ),
        phase4_cfg,
    )

    np.testing.assert_allclose(out[f"{const.PRED_STD_PREFIX}{target}"], 0.0)


# =============================================================================
# 3. Distribution-shift flagging
# =============================================================================
def test_flags_identify_unseen_proteins_and_families(
    candidate_table: pd.DataFrame, train_table: pd.DataFrame, phase4_cfg: dict[str, Any]
) -> None:
    """Flags match an independent set-difference against the training table."""
    out = pr.flag_distribution_shift(candidate_table, train_table, phase4_cfg)

    unseen_proteins = set(candidate_table[const.COL_PROTEIN_ID]) - set(train_table[const.COL_PROTEIN_ID])
    expected = out[const.COL_PROTEIN_ID].isin(unseen_proteins)
    pd.testing.assert_series_equal(out[const.COL_UNSEEN_PROTEIN], expected, check_names=False)
    assert unseen_proteins, "fixture must contain an unseen protein for this test to have teeth"


def test_unseen_family_takes_precedence_over_unseen_protein(phase4_cfg: dict[str, Any]) -> None:
    """A candidate from an unseen family carries the stronger of the two warnings."""
    train = pd.DataFrame({const.COL_PROTEIN_ID: ["P1"], const.COL_PROTEIN_FAMILY: ["F1"]})
    cands = pd.DataFrame(
        {
            const.COL_PROTEIN_ID: ["P1", "P2", "P9"],
            const.COL_PROTEIN_FAMILY: ["F1", "F1", "F9"],
        }
    )

    out = pr.flag_distribution_shift(cands, train, phase4_cfg)

    assert out[const.COL_OOD_FLAG].tolist() == [
        const.OOD_IN_DISTRIBUTION,
        const.OOD_UNSEEN_PROTEIN,
        const.OOD_UNSEEN_FAMILY,
    ]


def test_family_flagging_can_be_disabled(phase4_cfg: dict[str, Any]) -> None:
    """With family flagging off, an unseen-family candidate falls back to the protein flag."""
    phase4_cfg["prioritization"]["ood_flag_unseen_family"] = False
    train = pd.DataFrame({const.COL_PROTEIN_ID: ["P1"], const.COL_PROTEIN_FAMILY: ["F1"]})
    cands = pd.DataFrame({const.COL_PROTEIN_ID: ["P9"], const.COL_PROTEIN_FAMILY: ["F9"]})

    out = pr.flag_distribution_shift(cands, train, phase4_cfg)

    assert out[const.COL_OOD_FLAG].tolist() == [const.OOD_UNSEEN_PROTEIN]
    assert bool(out[const.COL_UNSEEN_FAMILY].iloc[0]) is True  # still recorded


# =============================================================================
# 4. Normalization & scoring
# =============================================================================
@pytest.mark.parametrize("method", list(const.ScoreNormalization))
def test_normalize_objective_is_monotonic_in_zero_one(method: const.ScoreNormalization) -> None:
    """Normalization preserves order and lands inside [0, 1] for both methods."""
    values = pd.Series([3.0, -1.0, 10.0, 0.5])

    normalized = pr.normalize_objective(values, method)

    assert normalized.between(0, 1).all()
    assert normalized.rank().tolist() == values.rank().tolist()


@pytest.mark.parametrize("method", list(const.ScoreNormalization))
def test_normalize_objective_constant_input_is_neutral(method: const.ScoreNormalization) -> None:
    """A degenerate objective contributes 0.5 to every candidate and cannot drive the ranking."""
    normalized = pr.normalize_objective(pd.Series([2.0, 2.0, 2.0]), method)

    assert normalized.tolist() == [0.5, 0.5, 0.5]


def test_minmax_normalization_keeps_spacing() -> None:
    """Min-max preserves relative distances, unlike rank normalization."""
    normalized = pr.normalize_objective(pd.Series([0.0, 1.0, 10.0]), const.ScoreNormalization.MINMAX)

    np.testing.assert_allclose(normalized, [0.0, 0.1, 1.0])


def test_stability_objective_rewards_negative_ddg(phase4_cfg: dict[str, Any]) -> None:
    """Stabilizing (negative ddG) scores higher than destabilizing, per the dataset convention."""
    phase4_cfg["prioritization"]["weights"] = {"stability": 1.0, "function": 0.0}
    phase4_cfg["prioritization"]["uncertainty_penalty"] = 0.0
    df = make_scored(ddg=[2.0, -2.0, 0.0], func=[0.5, 0.5, 0.5])

    scored = pr.score_candidates(df, phase4_cfg)

    assert scored[const.COL_MUTATION_ID].tolist() == ["CAND0002", "CAND0003", "CAND0001"]
    assert scored[const.COL_PRIORITY_RANK].tolist() == [1, 2, 3]


def test_function_objective_rewards_high_score(phase4_cfg: dict[str, Any]) -> None:
    """Higher functional retention scores higher."""
    phase4_cfg["prioritization"]["weights"] = {"stability": 0.0, "function": 1.0}
    phase4_cfg["prioritization"]["uncertainty_penalty"] = 0.0
    df = make_scored(ddg=[0.0, 0.0, 0.0], func=[0.1, 0.9, 0.5])

    scored = pr.score_candidates(df, phase4_cfg)

    assert scored[const.COL_MUTATION_ID].tolist() == ["CAND0002", "CAND0003", "CAND0001"]


def test_score_arithmetic_matches_hand_computation(phase4_cfg: dict[str, Any]) -> None:
    """Known inputs reproduce the documented formula exactly, including weight rescaling."""
    phase4_cfg["prioritization"]["weights"] = {"stability": 1.0, "function": 3.0}  # -> 0.25 / 0.75
    phase4_cfg["prioritization"]["uncertainty_penalty"] = 0.0
    phase4_cfg["prioritization"]["score_normalization"] = const.ScoreNormalization.MINMAX.value
    # -ddG = [1, -1] -> minmax [1, 0]; func [0.2, 1.0] -> minmax [0, 1]
    df = make_scored(ddg=[-1.0, 1.0], func=[0.2, 1.0])

    scored = pr.score_candidates(df, phase4_cfg)

    by_id = scored.set_index(const.COL_MUTATION_ID)[const.COL_PRIORITY_SCORE]
    assert by_id["CAND0001"] == pytest.approx(0.25 * 1 + 0.75 * 0)
    assert by_id["CAND0002"] == pytest.approx(0.25 * 0 + 0.75 * 1)


def test_uncertainty_penalty_demotes_uncertain_candidates(phase4_cfg: dict[str, Any]) -> None:
    """With identical objectives, the candidate whose fold models disagree most ranks last."""
    phase4_cfg["prioritization"]["uncertainty_penalty"] = 0.5
    phase4_cfg["prioritization"]["score_normalization"] = const.ScoreNormalization.MINMAX.value
    df = make_scored(ddg=[0.0, 0.0], func=[0.5, 0.5], std_ddg=[0.5, 0.0])

    scored = pr.score_candidates(df, phase4_cfg)

    assert scored[const.COL_MUTATION_ID].tolist() == ["CAND0002", "CAND0001"]
    assert scored[const.COL_PRIORITY_SCORE].iloc[0] > scored[const.COL_PRIORITY_SCORE].iloc[1]


def test_zero_penalty_ignores_uncertainty(phase4_cfg: dict[str, Any]) -> None:
    """With the penalty at zero, uncertainty does not affect the score at all."""
    phase4_cfg["prioritization"]["uncertainty_penalty"] = 0.0
    certain = make_scored(ddg=[0.0, 1.0], func=[0.5, 0.9], std_ddg=[0.0, 0.0])
    uncertain = make_scored(ddg=[0.0, 1.0], func=[0.5, 0.9], std_ddg=[0.9, 0.3])

    np.testing.assert_allclose(
        pr.score_candidates(certain, phase4_cfg)[const.COL_PRIORITY_SCORE],
        pr.score_candidates(uncertain, phase4_cfg)[const.COL_PRIORITY_SCORE],
    )


def test_total_uncertainty_combines_targets_in_quadrature(phase4_cfg: dict[str, Any]) -> None:
    """Per-target spreads are combined as independent sources of error."""
    df = make_scored(ddg=[0.0], func=[0.5], std_ddg=[0.3], std_func=[0.4])

    scored = pr.score_candidates(df, phase4_cfg)

    assert scored[const.COL_TOTAL_UNCERTAINTY].iloc[0] == pytest.approx(0.5)


def test_ranks_are_dense_and_ordered(phase4_cfg: dict[str, Any]) -> None:
    """Ranks run 1..n in descending score order with no gaps."""
    df = make_scored(ddg=[0.5, -0.5, 0.0, 1.0], func=[0.4, 0.8, 0.6, 0.2])

    scored = pr.score_candidates(df, phase4_cfg)

    assert scored[const.COL_PRIORITY_RANK].tolist() == [1, 2, 3, 4]
    assert scored[const.COL_PRIORITY_SCORE].is_monotonic_decreasing


def test_tied_scores_broken_deterministically(phase4_cfg: dict[str, Any]) -> None:
    """Identical candidates are ordered by mutation id, so the ranking is reproducible."""
    df = make_scored(ddg=[0.0, 0.0], func=[0.5, 0.5])

    first = pr.score_candidates(df, phase4_cfg)
    second = pr.score_candidates(df.iloc[::-1].reset_index(drop=True), phase4_cfg)

    assert first[const.COL_MUTATION_ID].tolist() == ["CAND0001", "CAND0002"]
    assert second[const.COL_MUTATION_ID].tolist() == first[const.COL_MUTATION_ID].tolist()


@pytest.mark.parametrize(
    "weights", [{"stability": 0.0, "function": 0.0}, {"stability": -1.0, "function": 2.0}]
)
def test_invalid_weights_raise(phase4_cfg: dict[str, Any], weights: dict[str, float]) -> None:
    """All-zero or negative weights are rejected rather than producing a meaningless ranking."""
    phase4_cfg["prioritization"]["weights"] = weights
    df = make_scored(ddg=[0.0, 1.0], func=[0.5, 0.9])

    with pytest.raises(ValueError, match="non-negative with a positive sum"):
        pr.score_candidates(df, phase4_cfg)


# =============================================================================
# 5. Top-K report
# =============================================================================
def test_build_top_k_shape_and_columns(phase4_cfg: dict[str, Any]) -> None:
    """The report holds exactly ``top_k`` rows and the documented columns, best first."""
    df = make_scored(ddg=[0.0] * 6, func=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    scored = pr.score_candidates(df, phase4_cfg)

    report = pr.build_top_k(scored, phase4_cfg)

    assert len(report) == TEST_TOP_K
    assert list(report.columns) == list(const.TOP_K_REPORT_COLUMNS)
    assert report[const.COL_PRIORITY_RANK].tolist() == [1, 2, 3, 4]


def test_build_top_k_warns_when_fewer_candidates(
    phase4_cfg: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Asking for more candidates than exist reports all of them, with a warning."""
    scored = pr.score_candidates(make_scored(ddg=[0.0, 1.0], func=[0.5, 0.9]), phase4_cfg)

    with caplog.at_level("WARNING", logger=pr.__name__):
        report = pr.build_top_k(scored, phase4_cfg)

    assert len(report) == 2
    assert any("exceeds" in r.getMessage() for r in caplog.records)


def test_justification_reports_direction_and_ood_caveat(phase4_cfg: dict[str, Any]) -> None:
    """Each justification states the predicted direction and any distribution-shift caveat."""
    df = make_scored(
        ddg=[-0.5, 0.5],
        func=[0.9, 0.3],
        flags=[const.OOD_UNSEEN_FAMILY, const.OOD_IN_DISTRIBUTION],
    )
    scored = pr.score_candidates(df, phase4_cfg)

    report = pr.build_top_k(scored, phase4_cfg)
    texts = dict(zip(report[const.COL_MUTATION_ID], report[const.COL_JUSTIFICATION], strict=True))

    assert "stabilizing" in texts["CAND0001"] and "exploratory" in texts["CAND0001"]
    assert "destabilizing" in texts["CAND0002"]
    assert "represented in training data" in texts["CAND0002"]


# =============================================================================
# End-to-end
# =============================================================================
def test_run_prediction_writes_all_artifacts(phase4_cfg: dict[str, Any]) -> None:
    """All candidates, the Top-K CSV and the Markdown report are written and consistent."""
    scored, report = pr.run_prediction(phase4_cfg)

    reports_dir = Path(phase4_cfg["paths"]["rankings_dir"])
    all_path = reports_dir / phase4_cfg["files"]["all_predictions"]
    top_path = reports_dir / phase4_cfg["files"]["top_candidates"]
    markdown_path = reports_dir / phase4_cfg["files"]["top_candidates_markdown"]

    assert len(report) == TEST_TOP_K
    pd.testing.assert_frame_equal(read_table(top_path), report, check_dtype=False)
    written = read_table(all_path)
    assert len(written) == len(scored)
    # Embeddings are reproducible from Phase 1 and must not bloat the report.
    assert not [c for c in written.columns if c.startswith(const.EMBEDDING_PREFIX)]

    markdown = markdown_path.read_text()
    assert f"Top {TEST_TOP_K} prioritized candidate mutations" in markdown
    assert str(phase4_cfg["project"]["random_seed"]) in markdown
    assert "relative to this candidate pool" in markdown
    for mutation_id in report[const.COL_MUTATION_ID]:
        assert mutation_id in markdown


def test_run_prediction_scores_every_candidate_once(phase4_cfg: dict[str, Any]) -> None:
    """Every candidate is scored exactly once, with finite predictions and a flag."""
    scored, _ = pr.run_prediction(phase4_cfg)
    candidate_table = read_table(
        Path(phase4_cfg["paths"]["processed_dir"]) / phase4_cfg["files"]["processed_candidates"]
    )

    assert sorted(scored[const.COL_MUTATION_ID]) == sorted(candidate_table[const.COL_MUTATION_ID])
    for target in const.TARGETS.values():
        assert np.isfinite(scored[f"{const.PRED_PREFIX}{target}"]).all()
    assert (
        scored[const.COL_OOD_FLAG]
        .isin([const.OOD_IN_DISTRIBUTION, const.OOD_UNSEEN_PROTEIN, const.OOD_UNSEEN_FAMILY])
        .all()
    )


def test_run_prediction_is_deterministic(phase4_cfg: dict[str, Any]) -> None:
    """Two runs on the same artifacts produce identical scores and the same shortlist."""
    scored_a, report_a = pr.run_prediction(phase4_cfg)
    scored_b, report_b = pr.run_prediction(phase4_cfg)

    pd.testing.assert_frame_equal(scored_a, scored_b)
    pd.testing.assert_frame_equal(report_a, report_b)


def test_run_prediction_requires_processed_candidates(phase4_cfg: dict[str, Any]) -> None:
    """Running Phase 4 without Phase 1 output fails with a pointer to the upstream stage."""
    Path(phase4_cfg["paths"]["processed_dir"], phase4_cfg["files"]["processed_candidates"]).unlink()

    with pytest.raises(FileNotFoundError, match="previous pipeline stage"):
        pr.run_prediction(phase4_cfg)


# =============================================================================
# Multi-objective viability gate (assignment Task 8)
# =============================================================================
def gate_cfg(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Apply viability settings on top of a config, leaving the rest alone."""
    cfg["prioritization"]["viability"].update(overrides)
    return cfg


def test_gate_passes_only_candidates_clearing_both_objectives(phase4_cfg: dict[str, Any]) -> None:
    """Viability is a conjunction: stabilizing AND functional, never either alone."""
    gate_cfg(phase4_cfg, enabled=True, max_stability_ddG=0.0, min_functional_score=0.7)
    #                     stabilizing? functional?
    df = make_scored(ddg=[-0.5, -0.5, 0.5, 0.5], func=[0.9, 0.1, 0.9, 0.1])

    gated = pr.apply_viability_gate(df, phase4_cfg)

    assert gated[const.COL_IS_VIABLE].tolist() == [True, False, False, False]
    assert gated[const.COL_VIABILITY].tolist() == [
        const.VIABILITY_PASS,
        const.VIABILITY_FAIL_FUNCTION,
        const.VIABILITY_FAIL_STABILITY,
        const.VIABILITY_FAIL_BOTH,
    ]


def test_gate_boundaries_are_inclusive(phase4_cfg: dict[str, Any]) -> None:
    """A candidate exactly on a threshold passes; the comparison is <= and >=.

    Pinned because flipping either to a strict inequality would silently drop
    boundary candidates from the shortlist.
    """
    gate_cfg(phase4_cfg, enabled=True, max_stability_ddG=0.0, min_functional_score=0.7)
    df = make_scored(ddg=[0.0, 0.001], func=[0.7, 0.7])

    gated = pr.apply_viability_gate(df, phase4_cfg)

    assert gated[const.COL_IS_VIABLE].tolist() == [True, False]


def test_disabled_gate_marks_everything_viable(phase4_cfg: dict[str, Any]) -> None:
    """Turning the gate off must not reorder anything or drop a candidate."""
    gate_cfg(phase4_cfg, enabled=False)
    df = make_scored(ddg=[5.0, -5.0], func=[0.0, 1.0])

    gated = pr.apply_viability_gate(df, phase4_cfg)

    assert gated[const.COL_IS_VIABLE].all()
    assert set(gated[const.COL_VIABILITY]) == {const.VIABILITY_PASS}


def test_viable_candidates_outrank_higher_scoring_non_viable_ones(
    phase4_cfg: dict[str, Any],
) -> None:
    """The gate is a hard constraint, so it sorts before the soft priority score.

    CAND0002 scores higher on the weighted objective but fails the stability
    gate; CAND0001 clears both gates and must still rank first.

    Function is weighted up so the soft score genuinely prefers CAND0002:
    with equal weights the two tie, and a tie would not test anything.
    """
    gate_cfg(phase4_cfg, enabled=True, max_stability_ddG=0.0, min_functional_score=0.7)
    phase4_cfg["prioritization"]["uncertainty_penalty"] = 0.0
    phase4_cfg["prioritization"]["weights"] = {"stability": 0.2, "function": 0.8}
    phase4_cfg["prioritization"]["score_normalization"] = const.ScoreNormalization.MINMAX.value
    df = make_scored(ddg=[-0.1, 0.5], func=[0.75, 1.0])

    scored = pr.score_candidates(df, phase4_cfg)

    assert scored[const.COL_MUTATION_ID].tolist() == ["CAND0001", "CAND0002"]
    assert scored[const.COL_PRIORITY_RANK].tolist() == [1, 2]
    # The soft score really does favour the other one; only the gate reorders.
    by_id = scored.set_index(const.COL_MUTATION_ID)[const.COL_PRIORITY_SCORE]
    assert by_id["CAND0002"] > by_id["CAND0001"]


def test_excluding_non_viable_shortens_the_pool(phase4_cfg: dict[str, Any]) -> None:
    """``exclude_non_viable: true`` drops them entirely rather than demoting them."""
    gate_cfg(
        phase4_cfg, enabled=True, max_stability_ddG=0.0, min_functional_score=0.7,
        exclude_non_viable=True,
    )
    df = make_scored(ddg=[-0.1, 0.5], func=[0.75, 1.0])

    scored = pr.score_candidates(df, phase4_cfg)

    assert scored[const.COL_MUTATION_ID].tolist() == ["CAND0001"]
    assert scored[const.COL_IS_VIABLE].all()


def test_ranking_is_contiguous_after_exclusion(phase4_cfg: dict[str, Any]) -> None:
    """Ranks restart at 1 with no gaps once candidates have been removed."""
    gate_cfg(
        phase4_cfg, enabled=True, max_stability_ddG=0.0, min_functional_score=0.5,
        exclude_non_viable=True,
    )
    df = make_scored(ddg=[-0.1, 5.0, -0.2, 5.0], func=[0.9, 0.9, 0.9, 0.1])

    scored = pr.score_candidates(df, phase4_cfg)

    assert scored[const.COL_PRIORITY_RANK].tolist() == list(range(1, len(scored) + 1))
    assert len(scored) == 2


def test_gate_status_reaches_the_shortlist_and_its_justification(
    phase4_cfg: dict[str, Any],
) -> None:
    """The reported table carries the gate verdict, and the prose names it."""
    gate_cfg(phase4_cfg, enabled=True, max_stability_ddG=0.0, min_functional_score=0.7)
    phase4_cfg["prioritization"]["top_k"] = 2
    df = make_scored(ddg=[-0.1, 0.5], func=[0.75, 1.0])
    scored = pr.score_candidates(df, phase4_cfg)

    report = pr.build_top_k(scored, phase4_cfg)

    assert const.COL_VIABILITY in report.columns
    assert report[const.COL_VIABILITY].tolist() == [
        const.VIABILITY_PASS, const.VIABILITY_FAIL_STABILITY
    ]
    assert "clears both viability gates" in report[const.COL_JUSTIFICATION].iloc[0]
    assert "stability gate" in report[const.COL_JUSTIFICATION].iloc[1]


def test_markdown_states_how_many_candidates_are_viable(phase4_cfg: dict[str, Any]) -> None:
    """A ten-row shortlist from a pool with few viable candidates must say so."""
    gate_cfg(phase4_cfg, enabled=True, max_stability_ddG=0.0, min_functional_score=0.7)
    df = make_scored(ddg=[-0.1, 0.5], func=[0.75, 1.0])
    scored = pr.score_candidates(df, phase4_cfg)

    lines = pr._viability_lines(scored, phase4_cfg)

    assert lines and "1 of 2" in lines[0]
    assert "not wet-lab ready" in lines[0]


def test_shortlist_has_no_missing_values(phase4_cfg: dict[str, Any]) -> None:
    """Every reported column is populated: a blank cell in a wet-lab handoff is a defect."""
    scored, report = pr.run_prediction(phase4_cfg)

    assert not report.isna().to_numpy().any()
    for column in (
        f"{const.PRED_PREFIX}{const.TARGET_STABILITY}",
        f"{const.PRED_STD_PREFIX}{const.TARGET_STABILITY}",
        f"{const.PRED_PREFIX}{const.TARGET_FUNCTION}",
        f"{const.PRED_STD_PREFIX}{const.TARGET_FUNCTION}",
        const.COL_TOTAL_UNCERTAINTY,
        const.COL_PRIORITY_SCORE,
    ):
        values = report[column].to_numpy(dtype=float)
        assert np.isfinite(values).all(), column
    assert not scored.isna().to_numpy().any()


def test_prediction_is_deterministic(phase4_cfg: dict[str, Any]) -> None:
    """Two runs over the same artifacts give the same ranking, row for row."""
    first_scored, first_report = pr.run_prediction(phase4_cfg)
    second_scored, second_report = pr.run_prediction(phase4_cfg)

    pd.testing.assert_frame_equal(first_report, second_report)
    pd.testing.assert_frame_equal(first_scored, second_scored)
