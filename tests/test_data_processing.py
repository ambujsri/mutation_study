"""Unit and integration tests for ``src.data_processing``.

Covers every public step of Phase 1 against small synthetic tables built in
``conftest.py``: schema validation, the embedding merge, anomaly detection and
handling, secondary-structure encoding, delta features, export, and an
end-to-end run of ``run_preprocessing``. Column names always come from
``src.common.constants``; the only literals are synthetic test values.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import data_processing as pp
from src.common import constants as const

ONEHOT_COLUMNS: list[str] = [f"{const.SS_ONEHOT_PREFIX}_{code}" for code in const.SS_CATEGORIES]


# =============================================================================
# Helpers
# =============================================================================
def merged_pair(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge both mutation tables with embeddings, as ``run_preprocessing`` does."""
    emb = frames[const.TABLE_EMBEDDINGS]
    train = pp.merge_embeddings(frames[const.TABLE_LABELED], emb, cfg, const.TABLE_LABELED)
    cands = pp.merge_embeddings(frames[const.TABLE_CANDIDATES], emb, cfg, const.TABLE_CANDIDATES)
    return train, cands


def with_encoding(cfg: dict[str, Any], encoding: const.Encoding) -> dict[str, Any]:
    """Return ``cfg`` with the secondary-structure encoding set to ``encoding``."""
    cfg["features"]["secondary_structure_encoding"] = encoding.value
    return cfg


# =============================================================================
# 1. Ingestion
# =============================================================================
def test_load_raw_data_reads_all_three_tables(
    cfg: dict[str, Any], raw_files: dict[str, Path], frames: dict[str, pd.DataFrame]
) -> None:
    """``load_raw_data`` returns every configured table with its original shape and columns."""
    loaded = pp.load_raw_data(cfg)

    assert set(loaded) == {const.TABLE_LABELED, const.TABLE_CANDIDATES, const.TABLE_EMBEDDINGS}
    for key, df in frames.items():
        assert loaded[key].shape == df.shape
        assert list(loaded[key].columns) == list(df.columns)


def test_load_raw_data_missing_file_raises(cfg: dict[str, Any]) -> None:
    """A missing raw file fails immediately instead of producing an empty table."""
    with pytest.raises(FileNotFoundError):
        pp.load_raw_data(cfg)


# =============================================================================
# 2. Schema validation
# =============================================================================
def test_expected_columns_cover_schema_constants() -> None:
    """Required columns include identifiers, all numeric features, targets and all embeddings."""
    expected = pp.expected_columns()

    for table in (const.TABLE_LABELED, const.TABLE_CANDIDATES):
        cols = expected[table]
        assert len(cols) == len(set(cols)), "required columns must be unique"
        assert {const.COL_MUTATION_ID, const.JOIN_KEY, const.COL_SECONDARY_STRUCTURE} <= set(cols)
        assert set(const.NUMERIC_FEATURE_COLUMNS) <= set(cols)
    assert set(const.TARGETS.values()) <= set(expected[const.TABLE_LABELED])
    assert not set(const.TARGETS.values()) & set(expected[const.TABLE_CANDIDATES])
    assert expected[const.TABLE_EMBEDDINGS][-const.EMBEDDING_DIM :] == list(const.EMBEDDING_COLUMNS)


def test_validate_schema_accepts_valid_frames(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """Well-formed tables pass validation without raising."""
    pp.validate_schema(frames, cfg)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        (const.TABLE_LABELED, const.TARGET_STABILITY),
        (const.TABLE_LABELED, const.TARGET_FUNCTION),
        (const.TABLE_LABELED, const.COL_PROTEIN_ID),
        (const.TABLE_CANDIDATES, const.COL_SECONDARY_STRUCTURE),
        (const.TABLE_CANDIDATES, const.COL_WT_VOLUME),
        (const.TABLE_EMBEDDINGS, const.EMBEDDING_COLUMNS[0]),
        (const.TABLE_EMBEDDINGS, const.EMBEDDING_COLUMNS[-1]),
    ],
)
def test_validate_schema_missing_column_raises(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], table: str, column: str
) -> None:
    """Dropping any required column raises ``ValueError`` naming the table and the column."""
    frames[table] = frames[table].drop(columns=column)

    with pytest.raises(ValueError, match=rf"\[{table}\].*{column}"):
        pp.validate_schema(frames, cfg)


def test_validate_schema_duplicate_ids_raise(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """Duplicate mutation IDs raise when ``fail_on_duplicate_ids`` is enabled."""
    labeled = frames[const.TABLE_LABELED]
    labeled.loc[1, const.COL_MUTATION_ID] = labeled.loc[0, const.COL_MUTATION_ID]
    cfg["quality"]["fail_on_duplicate_ids"] = True

    with pytest.raises(ValueError, match="duplicate"):
        pp.validate_schema(frames, cfg)


def test_validate_schema_duplicate_ids_warn_when_tolerated(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """With ``fail_on_duplicate_ids`` disabled, duplicates are logged as a warning instead."""
    labeled = frames[const.TABLE_LABELED]
    labeled.loc[1, const.COL_MUTATION_ID] = labeled.loc[0, const.COL_MUTATION_ID]
    cfg["quality"]["fail_on_duplicate_ids"] = False

    with caplog.at_level(logging.WARNING):
        pp.validate_schema(frames, cfg)
    assert any("duplicate" in r.message for r in caplog.records)


def test_validate_schema_warns_on_unexpected_counts(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Table sizes that differ from ``EXPECTED_COUNTS`` warn but do not fail."""
    with caplog.at_level(logging.WARNING):
        pp.validate_schema(frames, cfg)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    for label in const.EXPECTED_COUNTS:
        assert any(label in message for message in warnings), f"no count warning for {label}"


def test_validate_schema_warns_on_id_overlap(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """A mutation ID present in both labeled and candidate sets is flagged as possible leakage."""
    shared_id = frames[const.TABLE_LABELED].loc[0, const.COL_MUTATION_ID]
    frames[const.TABLE_CANDIDATES].loc[0, const.COL_MUTATION_ID] = shared_id

    with caplog.at_level(logging.WARNING):
        pp.validate_schema(frames, cfg)
    assert any("appear in both" in r.getMessage() for r in caplog.records)


# =============================================================================
# 3. Relational merge
# =============================================================================
@pytest.mark.parametrize("table", [const.TABLE_LABELED, const.TABLE_CANDIDATES])
def test_merge_preserves_rows_and_order(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], table: str
) -> None:
    """The inner join loses no rows and keeps the mutation table's row order."""
    mutations = frames[table]

    merged = pp.merge_embeddings(mutations, frames[const.TABLE_EMBEDDINGS], cfg, table)

    assert len(merged) == len(mutations)
    pd.testing.assert_series_equal(
        merged[const.COL_MUTATION_ID], mutations[const.COL_MUTATION_ID], check_names=True
    )


def test_merge_adds_full_embedding_block(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """All 256 embedding columns are added, in order, and nothing else changes name."""
    mutations = frames[const.TABLE_LABELED]

    merged = pp.merge_embeddings(mutations, frames[const.TABLE_EMBEDDINGS], cfg, const.TABLE_LABELED)

    emb_cols = [c for c in merged.columns if c.startswith(const.EMBEDDING_PREFIX)]
    assert emb_cols == list(const.EMBEDDING_COLUMNS)
    assert len(emb_cols) == const.EMBEDDING_DIM
    assert list(merged.columns) == [*mutations.columns, *const.EMBEDDING_COLUMNS]


def test_merge_does_not_duplicate_shared_columns(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> None:
    """Shared protein columns appear once, with no ``_x`` / ``_y`` merge suffixes."""
    merged = pp.merge_embeddings(
        frames[const.TABLE_LABELED], frames[const.TABLE_EMBEDDINGS], cfg, const.TABLE_LABELED
    )

    for col in const.SHARED_PROTEIN_COLUMNS:
        assert list(merged.columns).count(col) == 1
    assert not [c for c in merged.columns if c.endswith(("_x", "_y"))]


@pytest.mark.parametrize("table", [const.TABLE_LABELED, const.TABLE_CANDIDATES])
def test_merge_aligns_each_mutation_with_its_protein_vector(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], table: str
) -> None:
    """Every merged row carries exactly the embedding vector of its own protein."""
    embeddings = frames[const.TABLE_EMBEDDINGS].set_index(const.JOIN_KEY)
    emb_cols = list(const.EMBEDDING_COLUMNS)

    merged = pp.merge_embeddings(frames[table], frames[const.TABLE_EMBEDDINGS], cfg, table)

    for _, row in merged.iterrows():
        expected = embeddings.loc[row[const.JOIN_KEY], emb_cols].to_numpy(dtype=float)
        np.testing.assert_array_equal(row[emb_cols].to_numpy(dtype=float), expected)


def test_merge_vectors_differ_between_proteins(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """Rows from different proteins get different vectors (guards against a broadcast bug)."""
    merged = pp.merge_embeddings(
        frames[const.TABLE_LABELED], frames[const.TABLE_EMBEDDINGS], cfg, const.TABLE_LABELED
    )

    per_protein = merged.groupby(const.JOIN_KEY)[list(const.EMBEDDING_COLUMNS)].nunique()
    assert (per_protein == 1).all().all(), "one vector per protein"
    unique_vectors = merged.drop_duplicates(subset=const.JOIN_KEY)[list(const.EMBEDDING_COLUMNS)]
    assert len(unique_vectors.drop_duplicates()) == merged[const.JOIN_KEY].nunique()


def test_merge_unmatched_protein_raises(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """A mutation whose protein has no embedding stops the pipeline when row loss is fatal."""
    embeddings = frames[const.TABLE_EMBEDDINGS]
    missing_protein = frames[const.TABLE_LABELED].loc[0, const.JOIN_KEY]
    embeddings = embeddings[embeddings[const.JOIN_KEY] != missing_protein]
    cfg["quality"]["fail_on_row_loss_in_merge"] = True

    with pytest.raises(ValueError, match="no embedding"):
        pp.merge_embeddings(frames[const.TABLE_LABELED], embeddings, cfg, const.TABLE_LABELED)


def test_merge_unmatched_protein_dropped_when_tolerated(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """With row loss tolerated, unmatched rows are dropped and the loss is logged."""
    labeled = frames[const.TABLE_LABELED]
    embeddings = frames[const.TABLE_EMBEDDINGS]
    missing_protein = labeled.loc[0, const.JOIN_KEY]
    embeddings = embeddings[embeddings[const.JOIN_KEY] != missing_protein]
    cfg["quality"]["fail_on_row_loss_in_merge"] = False

    with caplog.at_level(logging.WARNING):
        merged = pp.merge_embeddings(labeled, embeddings, cfg, const.TABLE_LABELED)

    assert len(merged) == int((labeled[const.JOIN_KEY] != missing_protein).sum())
    assert missing_protein not in set(merged[const.JOIN_KEY])
    assert any("lost" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("column", list(const.SHARED_PROTEIN_COLUMNS))
def test_merge_shared_column_disagreement_raises(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], column: str
) -> None:
    """A protein whose family or length differs between tables is a data error, not a merge key."""
    labeled = frames[const.TABLE_LABELED]
    original = labeled.loc[0, column]
    labeled[column] = labeled[column].astype(object)
    labeled.loc[0, column] = f"{original}_conflict"

    with pytest.raises(ValueError, match=column):
        pp.merge_embeddings(labeled, frames[const.TABLE_EMBEDDINGS], cfg, const.TABLE_LABELED)


# =============================================================================
# 4. Missing values & anomalies
# =============================================================================
def test_inspect_anomalies_clean_frame_reports_nothing(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> None:
    """Clean synthetic data yields an empty anomaly report."""
    train, _ = merged_pair(frames, cfg)

    report = pp.inspect_anomalies(train, const.TABLE_LABELED)

    assert report["rows"] == len(train)
    assert report["missing"] == report["infinite"] == report["out_of_range"] == {}


def test_inspect_anomalies_detects_nan_inf_and_out_of_range(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> None:
    """NaN, ±inf and out-of-range values are each counted per column, without altering data."""
    train, _ = merged_pair(frames, cfg)
    train.loc[0, const.COL_CONSERVATION] = np.nan
    train.loc[1, const.COL_WT_VOLUME] = np.inf
    train.loc[2, const.COL_SOLVENT_ACCESSIBILITY] = 1.5
    train.loc[3, const.TARGET_FUNCTION] = -0.2
    snapshot = train.copy()

    report = pp.inspect_anomalies(train, const.TABLE_LABELED)

    assert report["missing"] == {const.COL_CONSERVATION: 1}
    assert report["infinite"] == {const.COL_WT_VOLUME: 1}
    assert report["out_of_range"] == {const.COL_SOLVENT_ACCESSIBILITY: 1, const.TARGET_FUNCTION: 1}
    pd.testing.assert_frame_equal(train, snapshot)


def test_handle_anomalies_noop_on_clean_data(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """Clean inputs come back unchanged and the action log records nothing."""
    train, cands = merged_pair(frames, cfg)
    features = list(const.NUMERIC_FEATURE_COLUMNS)

    out_train, out_cands, actions = pp.handle_anomalies(train, cands, cfg, features)

    pd.testing.assert_frame_equal(out_train, train)
    pd.testing.assert_frame_equal(out_cands, cands)
    assert actions == {"inf_replaced": 0, "target_rows_dropped": 0, "imputed": {}}


def test_handle_anomalies_drops_rows_with_missing_targets(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> None:
    """Labels are never imputed: rows missing either target are removed and counted."""
    train, cands = merged_pair(frames, cfg)
    train.loc[0, const.TARGET_STABILITY] = np.nan
    train.loc[1, const.TARGET_FUNCTION] = np.nan
    cfg["quality"]["drop_rows_with_missing_target"] = True

    out_train, _, actions = pp.handle_anomalies(train, cands, cfg, list(const.NUMERIC_FEATURE_COLUMNS))

    assert len(out_train) == len(train) - 2
    assert not out_train[list(const.TARGETS.values())].isna().any().any()
    assert actions["target_rows_dropped"] == 2


@pytest.mark.parametrize("strategy", list(const.Imputation))
def test_handle_anomalies_imputes_with_training_statistics_only(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], strategy: const.Imputation
) -> None:
    """Missing and infinite features are imputed from training statistics, never candidate ones."""
    train, cands = merged_pair(frames, cfg)
    col = const.COL_CONSERVATION
    train.loc[0, col] = np.nan
    train.loc[1, col] = np.inf
    cands.loc[0, col] = np.nan
    cands.loc[1:, col] = 1.0  # would shift the statistic if candidates leaked into it
    cfg["quality"]["numeric_imputation"] = strategy.value
    cfg["quality"]["replace_inf_with_nan"] = True
    observed = train.loc[2:, col]
    expected_fill = observed.median() if strategy is const.Imputation.MEDIAN else observed.mean()

    out_train, out_cands, actions = pp.handle_anomalies(train, cands, cfg, [col])

    assert out_train.loc[0, col] == pytest.approx(expected_fill)
    assert out_train.loc[1, col] == pytest.approx(expected_fill)
    assert out_cands.loc[0, col] == pytest.approx(expected_fill)
    assert actions["inf_replaced"] == 1
    assert actions["imputed"] == {"train": {col: 2}, "candidates": {col: 1}}


def test_handle_anomalies_does_not_clip_out_of_range(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> None:
    """Out-of-range values are reported elsewhere but never silently altered."""
    train, cands = merged_pair(frames, cfg)
    train.loc[0, const.COL_SOLVENT_ACCESSIBILITY] = 1.7

    out_train, _, _ = pp.handle_anomalies(train, cands, cfg, list(const.NUMERIC_FEATURE_COLUMNS))

    assert out_train.loc[0, const.COL_SOLVENT_ACCESSIBILITY] == pytest.approx(1.7)


def test_handle_anomalies_does_not_mutate_inputs(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> None:
    """Callers' dataframes are left untouched (functions return new frames)."""
    train, cands = merged_pair(frames, cfg)
    train.loc[0, const.COL_WT_CHARGE] = np.nan
    train_before, cands_before = train.copy(), cands.copy()

    pp.handle_anomalies(train, cands, cfg, list(const.NUMERIC_FEATURE_COLUMNS))

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(cands, cands_before)


# =============================================================================
# 5. Categorical encoding
# =============================================================================
def test_onehot_encoding_columns_and_values(labeled: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """One-hot produces one int column per H/E/C code, exactly one 1 per row, matching the code."""
    cfg = with_encoding(cfg, const.Encoding.ONEHOT)

    encoded, new_cols = pp.encode_secondary_structure(labeled, cfg, const.TABLE_LABELED)

    assert new_cols == ONEHOT_COLUMNS
    assert (encoded[new_cols].sum(axis=1) == 1).all()
    for code, col in zip(const.SS_CATEGORIES, ONEHOT_COLUMNS, strict=True):
        expected = (labeled[const.COL_SECONDARY_STRUCTURE] == code).astype(np.int8)
        pd.testing.assert_series_equal(encoded[col], expected, check_names=False)
        assert encoded[col].dtype == np.int8


def test_onehot_columns_stable_when_category_absent(labeled: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """A table missing a category still gets the full, identically ordered column set."""
    cfg = with_encoding(cfg, const.Encoding.ONEHOT)
    only_first = labeled[labeled[const.COL_SECONDARY_STRUCTURE] == const.SS_CATEGORIES[0]]

    encoded, new_cols = pp.encode_secondary_structure(only_first, cfg, const.TABLE_CANDIDATES)

    assert new_cols == ONEHOT_COLUMNS
    assert (encoded[ONEHOT_COLUMNS[0]] == 1).all()
    assert (encoded[ONEHOT_COLUMNS[1:]] == 0).all().all()


def test_onehot_unknown_code_yields_zero_row_and_warning(
    labeled: pd.DataFrame, cfg: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """An unrecognized structure code becomes an all-zero row and is logged, not silently mapped."""
    cfg = with_encoding(cfg, const.Encoding.ONEHOT)
    labeled.loc[0, const.COL_SECONDARY_STRUCTURE] = "X"

    with caplog.at_level(logging.WARNING):
        encoded, new_cols = pp.encode_secondary_structure(labeled, cfg, const.TABLE_LABELED)

    assert encoded.loc[0, new_cols].sum() == 0
    assert any("unknown" in r.getMessage() for r in caplog.records)


def test_onehot_keeps_raw_column_and_input(labeled: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """The raw code column is retained for reporting, and the input frame is not modified."""
    cfg = with_encoding(cfg, const.Encoding.ONEHOT)
    before = labeled.copy()

    encoded, _ = pp.encode_secondary_structure(labeled, cfg, const.TABLE_LABELED)

    pd.testing.assert_series_equal(
        encoded[const.COL_SECONDARY_STRUCTURE], labeled[const.COL_SECONDARY_STRUCTURE]
    )
    pd.testing.assert_frame_equal(labeled, before)


def test_ordinal_encoding_uses_constant_map(labeled: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Ordinal encoding maps each code via ``SS_ORDINAL_MAP`` into a single column."""
    cfg = with_encoding(cfg, const.Encoding.ORDINAL)

    encoded, new_cols = pp.encode_secondary_structure(labeled, cfg, const.TABLE_LABELED)

    assert new_cols == [const.SS_ORDINAL_COLUMN]
    expected = labeled[const.COL_SECONDARY_STRUCTURE].map(const.SS_ORDINAL_MAP)
    assert encoded[const.SS_ORDINAL_COLUMN].tolist() == expected.tolist()


def test_ordinal_unknown_code_raises(labeled: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Ordinal encoding cannot assign a rank to an unknown code, so it raises."""
    cfg = with_encoding(cfg, const.Encoding.ORDINAL)
    labeled.loc[0, const.COL_SECONDARY_STRUCTURE] = "X"

    with pytest.raises(ValueError, match="unknown"):
        pp.encode_secondary_structure(labeled, cfg, const.TABLE_LABELED)


# =============================================================================
# 6. Physicochemical delta features
# =============================================================================
@pytest.mark.parametrize(("feature", "pair"), list(const.DELTA_PAIRS.items()))
def test_delta_equals_mutant_minus_wildtype(
    labeled: pd.DataFrame, cfg: dict[str, Any], feature: str, pair: tuple[str, str]
) -> None:
    """Each signed delta is exactly mutant − wild-type, and its absolute version is |delta|."""
    wt_col, mut_col = pair
    cfg["features"]["add_abs_deltas"] = True

    out, _ = pp.engineer_features(labeled, cfg, const.TABLE_LABELED)

    delta = out[f"{const.DELTA_PREFIX}{feature}"]
    np.testing.assert_allclose(delta, labeled[mut_col] - labeled[wt_col])
    np.testing.assert_allclose(out[f"{const.ABS_DELTA_PREFIX}{feature}"], delta.abs())


def test_delta_known_values(labeled: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Hand-computed example: V→D (Kyte-Doolittle, charge, volume) gives the textbook shifts."""
    row = labeled.index[0]
    labeled.loc[row, [const.COL_WT_HYDROPHOBICITY, const.COL_MUT_HYDROPHOBICITY]] = [4.2, -3.5]
    labeled.loc[row, [const.COL_WT_CHARGE, const.COL_MUT_CHARGE]] = [0.0, -1.0]
    labeled.loc[row, [const.COL_WT_VOLUME, const.COL_MUT_VOLUME]] = [140.0, 111.1]

    out, _ = pp.engineer_features(labeled, cfg, const.TABLE_LABELED)

    assert out.loc[row, f"{const.DELTA_PREFIX}hydrophobicity"] == pytest.approx(-7.7)
    assert out.loc[row, f"{const.DELTA_PREFIX}charge"] == pytest.approx(-1.0)
    assert out.loc[row, f"{const.DELTA_PREFIX}volume"] == pytest.approx(-28.9)
    assert out.loc[row, f"{const.ABS_DELTA_PREFIX}hydrophobicity"] == pytest.approx(7.7)


def test_relative_position(labeled: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Relative position is position / protein_length and lies in (0, 1] for valid data."""
    cfg["features"]["add_relative_position"] = True

    out, _ = pp.engineer_features(labeled, cfg, const.TABLE_LABELED)

    rel = out[const.COL_RELATIVE_POSITION]
    np.testing.assert_allclose(rel, labeled[const.COL_POSITION] / labeled[const.COL_PROTEIN_LENGTH])
    assert ((rel > 0) & (rel <= 1)).all()


def test_relative_position_beyond_length_warns(
    labeled: pd.DataFrame, cfg: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """A position past the protein's length is flagged as a data problem."""
    labeled.loc[0, const.COL_POSITION] = labeled.loc[0, const.COL_PROTEIN_LENGTH] + 1

    with caplog.at_level(logging.WARNING):
        pp.engineer_features(labeled, cfg, const.TABLE_LABELED)
    assert any("position > protein_length" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("add_abs", [True, False])
@pytest.mark.parametrize("add_rel", [True, False])
def test_feature_toggles_control_output_columns(
    labeled: pd.DataFrame, cfg: dict[str, Any], add_abs: bool, add_rel: bool
) -> None:
    """Optional features appear exactly when their config toggle is on; originals are retained."""
    cfg["features"]["add_abs_deltas"] = add_abs
    cfg["features"]["add_relative_position"] = add_rel

    out, new_cols = pp.engineer_features(labeled, cfg, const.TABLE_LABELED)

    expected = []
    for feature in const.DELTA_PAIRS:
        expected.append(f"{const.DELTA_PREFIX}{feature}")
        if add_abs:
            expected.append(f"{const.ABS_DELTA_PREFIX}{feature}")
    if add_rel:
        expected.append(const.COL_RELATIVE_POSITION)
    assert new_cols == expected
    assert list(out.columns) == [*labeled.columns, *expected]


def test_engineer_features_does_not_mutate_input(labeled: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Feature engineering returns a new frame and leaves the input unchanged."""
    before = labeled.copy()

    pp.engineer_features(labeled, cfg, const.TABLE_LABELED)

    pd.testing.assert_frame_equal(labeled, before)


# =============================================================================
# 7. Export
# =============================================================================
@pytest.mark.parametrize("suffix", list(const.SUPPORTED_TABLE_FORMATS))
def test_export_round_trips_supported_formats(
    frames: dict[str, pd.DataFrame], cfg: dict[str, Any], suffix: str
) -> None:
    """Each supported table format is written and reads back identically."""
    train, cands = merged_pair(frames, cfg)
    for key in ("processed_train", "processed_candidates"):
        cfg["files"][key] = f"{key}{suffix}"
    manifest = pp.build_feature_manifest(cfg, [], [], {})

    paths = pp.export_artifacts(train, cands, manifest, cfg)

    reader = pd.read_parquet if suffix == ".parquet" else pd.read_csv
    pd.testing.assert_frame_equal(reader(paths["processed_train"]), train, check_dtype=False)
    pd.testing.assert_frame_equal(reader(paths["processed_candidates"]), cands, check_dtype=False)
    assert json.loads(paths["feature_manifest"].read_text()) == manifest


def test_export_unsupported_format_raises(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """An unsupported output extension fails instead of writing an unreadable file."""
    train, cands = merged_pair(frames, cfg)
    cfg["files"]["processed_train"] = "processed_train.xlsx"

    with pytest.raises(ValueError, match="Unsupported output format"):
        pp.export_artifacts(train, cands, pp.build_feature_manifest(cfg, [], [], {}), cfg)


def test_feature_manifest_contents(cfg: dict[str, Any], seed: int) -> None:
    """The manifest lists ids, groups, targets, ordered feature sets and the run seed."""
    manifest = pp.build_feature_manifest(cfg, ["enc"], ["eng"], {"ok": True})

    assert manifest["id_column"] == const.COL_MUTATION_ID
    assert manifest["group_columns"] == dict(const.GROUP_COLUMNS)
    assert manifest["targets"] == dict(const.TARGETS)
    assert manifest["tabular_features"] == [*const.NUMERIC_FEATURE_COLUMNS, "enc", "eng"]
    assert manifest["embedding_features"] == list(const.EMBEDDING_COLUMNS)
    assert manifest["random_seed"] == seed
    assert manifest["quality_report"] == {"ok": True}


# =============================================================================
# Integration: full Phase 1 run
# =============================================================================
def test_run_preprocessing_end_to_end(
    cfg: dict[str, Any], raw_files: dict[str, Path], frames: dict[str, pd.DataFrame]
) -> None:
    """The full pipeline writes consistent train/candidate artifacts and a complete manifest."""
    train, cands, manifest = pp.run_preprocessing(cfg)

    assert len(train) == len(frames[const.TABLE_LABELED])
    assert len(cands) == len(frames[const.TABLE_CANDIDATES])
    features = manifest["tabular_features"] + manifest["embedding_features"]
    assert set(features) <= set(train.columns)
    assert set(features) <= set(cands.columns)
    assert not train[features].isna().any().any()
    assert set(const.TARGETS.values()) <= set(train.columns)
    assert not set(const.TARGETS.values()) & set(cands.columns)

    for key in ("processed_train", "processed_candidates", "feature_manifest"):
        assert (Path(cfg["paths"]["processed_dir"]) / cfg["files"][key]).is_file()


def test_run_preprocessing_reports_unseen_groups(
    cfg: dict[str, Any], raw_files: dict[str, Path], frames: dict[str, pd.DataFrame]
) -> None:
    """Candidates on proteins absent from training are reported for OOD flagging downstream."""
    _, _, manifest = pp.run_preprocessing(cfg)

    unseen_proteins = sorted(
        set(frames[const.TABLE_CANDIDATES][const.COL_PROTEIN_ID])
        - set(frames[const.TABLE_LABELED][const.COL_PROTEIN_ID])
    )
    report = manifest["quality_report"]["unseen_protein"]
    assert unseen_proteins, "fixture must contain an unseen protein for this test to be meaningful"
    assert report["values"] == unseen_proteins
    assert report["candidate_rows"] == int(
        frames[const.TABLE_CANDIDATES][const.COL_PROTEIN_ID].isin(unseen_proteins).sum()
    )


def test_run_preprocessing_is_deterministic(cfg: dict[str, Any], raw_files: dict[str, Path]) -> None:
    """Two runs on the same inputs and config produce identical artifacts (reproducibility)."""
    train_a, cands_a, manifest_a = pp.run_preprocessing(cfg)
    train_b, cands_b, manifest_b = pp.run_preprocessing(cfg)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(cands_a, cands_b)
    assert manifest_a == manifest_b
