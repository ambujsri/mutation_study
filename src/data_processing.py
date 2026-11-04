"""Data loading, relational merging, anomaly checks and delta feature engineering.

Phase 1 of the Medvolt.AI mutation-effect pipeline. Dataset facts (column names,
codes, value ranges) come from ``src.common.constants``; run-level decisions
(paths, encoding, imputation policy) come from ``config/config.yaml``.

Steps (each is a standalone function so it can be unit-tested or reused):

1. ``load_raw_data``              — read the three raw CSVs.
2. ``validate_schema``            — assert required columns are present and IDs unique.
3. ``merge_embeddings``           — inner-join mutations to protein embeddings on
                                    ``protein_id`` with zero-row-loss and alignment checks.
4. ``inspect_anomalies`` /
   ``handle_anomalies``           — detect NaN / ±inf / out-of-range values and apply
                                    the configured handling policy.
5. ``encode_secondary_structure`` — one-hot (or ordinal) encode H / E / C.
6. ``engineer_features``          — Δhydrophobicity, Δcharge, Δvolume (+ optional |Δ|
                                    and relative position).
7. ``export_artifacts``           — write processed tables and a feature manifest.

Run directly::

    python -m src.data_processing --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common import constants as const
from src.common.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_path,
    setup_logging,
)
from src.common.io import write_table

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Ingestion
# =============================================================================
def load_raw_data(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Read the labeled, candidate and embedding CSVs.

    Returns:
        Dict keyed by ``const.TABLE_LABELED``, ``const.TABLE_CANDIDATES``, ``const.TABLE_EMBEDDINGS``.
    """
    frames = {}
    for key in (const.TABLE_LABELED, const.TABLE_CANDIDATES, const.TABLE_EMBEDDINGS):
        path = resolve_path(cfg, "raw_dir", key)
        frames[key] = pd.read_csv(path)
        logger.info("Loaded %-10s %-40s shape=%s", key, path.name, frames[key].shape)
    return frames


# =============================================================================
# 2. Schema validation
# =============================================================================
def expected_columns() -> dict[str, list[str]]:
    """Return the required column set for each raw table."""
    mutation_cols = list(
        dict.fromkeys(  # dedupe, keep order
            [
                const.COL_MUTATION_ID,
                const.JOIN_KEY,
                *const.SHARED_PROTEIN_COLUMNS,
                *const.RESIDUE_COLUMNS,
                const.COL_SECONDARY_STRUCTURE,
                *const.NUMERIC_FEATURE_COLUMNS,
            ]
        )
    )
    return {
        const.TABLE_LABELED: mutation_cols + list(const.TARGETS.values()),
        const.TABLE_CANDIDATES: mutation_cols,
        const.TABLE_EMBEDDINGS: [const.JOIN_KEY, *const.SHARED_PROTEIN_COLUMNS, *const.EMBEDDING_COLUMNS],
    }


def validate_schema(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """Check required columns, ID uniqueness and expected table sizes.

    Missing columns and duplicate primary keys raise; deviations from
    ``const.EXPECTED_COUNTS`` are logged as warnings so the pipeline still runs on
    a refreshed dataset of a different size.

    Raises:
        ValueError: On missing columns or duplicate IDs (if configured to fail).
    """
    for name, required in expected_columns().items():
        missing = [c for c in required if c not in frames[name].columns]
        if missing:
            raise ValueError(f"[{name}] missing {len(missing)} required columns: {missing[:10]}")
        logger.info("[%s] schema OK (%d required columns present)", name, len(required))

    id_keys = {
        const.TABLE_LABELED: const.COL_MUTATION_ID,
        const.TABLE_CANDIDATES: const.COL_MUTATION_ID,
        const.TABLE_EMBEDDINGS: const.JOIN_KEY,
    }
    for name, key in id_keys.items():
        n_dup = int(frames[name][key].duplicated().sum())
        if n_dup:
            msg = f"[{name}] {n_dup} duplicate values in key column '{key}'"
            if cfg["quality"]["fail_on_duplicate_ids"]:
                raise ValueError(msg)
            logger.warning(msg)

    overlap = set(frames[const.TABLE_LABELED][const.COL_MUTATION_ID]) & set(
        frames[const.TABLE_CANDIDATES][const.COL_MUTATION_ID]
    )
    if overlap:
        logger.warning("%d mutation IDs appear in both labeled and candidate sets", len(overlap))

    emb = frames[const.TABLE_EMBEDDINGS]
    observed = {
        "labeled_rows": len(frames[const.TABLE_LABELED]),
        "candidate_rows": len(frames[const.TABLE_CANDIDATES]),
        "proteins": emb[const.JOIN_KEY].nunique(),
        "families": emb[const.COL_PROTEIN_FAMILY].nunique(),
    }
    for label, got in observed.items():
        want = const.EXPECTED_COUNTS[label]
        if got != want:
            logger.warning("Expected %d %s, found %d", want, label, got)
        else:
            logger.info("Count check OK: %s = %d", label, got)


# =============================================================================
# 3. Relational merge
# =============================================================================
def merge_embeddings(
    mutations: pd.DataFrame, embeddings: pd.DataFrame, cfg: dict[str, Any], name: str
) -> pd.DataFrame:
    """Inner-join a mutation table to protein embeddings on ``const.JOIN_KEY``.

    Columns shared by both tables (``const.SHARED_PROTEIN_COLUMNS``) are first
    checked for per-protein agreement, then kept from the mutation side only so
    the merge produces no ``_x`` / ``_y`` duplicates.

    Guarantees:
        * many-to-one relationship (each mutation maps to exactly one protein);
        * zero row loss (raises if any mutation has no embedding, when
          ``quality.fail_on_row_loss_in_merge`` is true);
        * embedding vectors in the output equal the source vector for their
          protein (alignment check).

    Args:
        mutations: Labeled or candidate mutation table.
        embeddings: Protein embedding table (one row per protein).
        cfg: Configuration dictionary.
        name: Label used in log messages.

    Returns:
        Merged dataframe, row order preserved from ``mutations``.

    Raises:
        ValueError: On shared-column disagreement, row loss or misalignment.
    """
    key, emb_cols = const.JOIN_KEY, list(const.EMBEDDING_COLUMNS)
    fail_on_loss = cfg["quality"]["fail_on_row_loss_in_merge"]

    for col in const.SHARED_PROTEIN_COLUMNS:
        mapped = mutations[key].map(embeddings.set_index(key)[col])
        mismatch = mutations[col].ne(mapped) & mapped.notna()
        if mismatch.any():
            raise ValueError(
                f"[{name}] '{col}' disagrees with embedding table for {int(mismatch.sum())} "
                f"rows (e.g. {mutations.loc[mismatch, key].unique()[:5]})"
            )
    logger.info("[%s] shared columns %s consistent with embedding table", name, const.SHARED_PROTEIN_COLUMNS)

    unmatched = sorted(set(mutations[key]) - set(embeddings[key]))
    if unmatched:
        msg = f"[{name}] {len(unmatched)} proteins have no embedding: {unmatched[:10]}"
        if fail_on_loss:
            raise ValueError(msg)
        logger.warning(msg)

    merged = mutations.merge(
        embeddings[[key, *emb_cols]], on=key, how="inner", validate="many_to_one", sort=False
    )

    if len(merged) != len(mutations):
        msg = f"[{name}] merge lost {len(mutations) - len(merged)} rows"
        if fail_on_loss:
            raise ValueError(msg)
        logger.warning(msg)

    source = embeddings.set_index(key).loc[merged[key], emb_cols].to_numpy()
    if not np.array_equal(merged[emb_cols].to_numpy(), source, equal_nan=True):
        raise ValueError(f"[{name}] embedding vectors misaligned after merge")

    logger.info(
        "[%s] merged %d rows x %d cols; %d proteins, zero row loss, embeddings aligned",
        name,
        len(merged),
        merged.shape[1],
        merged[key].nunique(),
    )
    return merged


# =============================================================================
# 4. Missing values & anomalies
# =============================================================================
def inspect_anomalies(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Count NaN, ±inf and out-of-range values without modifying the data.

    Args:
        df: Merged dataframe.
        name: Label used in log messages.

    Returns:
        Report with per-column counts of ``missing``, ``infinite`` and
        ``out_of_range`` values (non-zero entries only).
    """
    missing = df.isna().sum()
    infinite = np.isinf(df.select_dtypes(include=np.number)).sum()

    out_of_range: dict[str, int] = {}
    for col, (lo, hi) in const.VALUE_RANGES.items():
        if col not in df.columns:
            continue
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[col] < lo
        if hi is not None:
            bad |= df[col] > hi
        if bad.any():
            out_of_range[col] = int(bad.sum())

    report = {
        "rows": len(df),
        "missing": {k: int(v) for k, v in missing[missing > 0].items()},
        "infinite": {k: int(v) for k, v in infinite[infinite > 0].items()},
        "out_of_range": out_of_range,
    }
    findings = {k: report[k] for k in ("missing", "infinite", "out_of_range") if report[k]}
    if findings:
        logger.warning("[%s] anomalies found: %s", name, findings)
    else:
        logger.info("[%s] no missing, infinite or out-of-range values", name)
    return report


def handle_anomalies(
    train: pd.DataFrame, candidates: pd.DataFrame, cfg: dict[str, Any], feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the configured anomaly-handling policy.

    Policy (rationale in brackets):
        * ±inf → NaN, so they are treated as missing [an infinite feature value
          is never physically meaningful here and would break scalers].
        * Rows with a missing target are dropped from training [imputing a
          label fabricates supervision signal].
        * Missing numeric features are imputed with the median/mean computed on
          the **labeled** set only and applied to both tables [candidates must
          not influence training statistics; median is robust to skew].
        * Out-of-range values are reported but not clipped [they may be real
          signal; silently altering them would hide data issues].

    Imputation statistics are fit on the full labeled set before CV. This is a
    negligible leak for this dataset (no missing values were found); if
    anomalies appear in a future refresh, move imputation inside the CV
    pipeline in ``train.py``.

    Returns:
        ``(train, candidates, actions)`` where ``actions`` records what was done.
    """
    q = cfg["quality"]
    targets = list(const.TARGETS.values())
    actions: dict[str, Any] = {"inf_replaced": 0, "target_rows_dropped": 0, "imputed": {}}
    train, candidates = train.copy(), candidates.copy()

    if q["replace_inf_with_nan"]:
        for df in (train, candidates):
            num = df.select_dtypes(include=np.number).columns
            n_inf = int(np.isinf(df[num]).sum().sum())
            if n_inf:
                df[num] = df[num].replace([np.inf, -np.inf], np.nan)
                actions["inf_replaced"] += n_inf

    if q["drop_rows_with_missing_target"]:
        mask = train[targets].isna().any(axis=1)
        if mask.any():
            logger.warning("Dropping %d training rows with missing targets", int(mask.sum()))
            train = train.loc[~mask].reset_index(drop=True)
            actions["target_rows_dropped"] = int(mask.sum())

    if train[feature_cols].isna().any().any() or candidates[feature_cols].isna().any().any():
        strategy = const.Imputation(q["numeric_imputation"]).value
        stats = getattr(train[feature_cols], strategy)()
        for df_name, df in (("train", train), ("candidates", candidates)):
            counts = df[feature_cols].isna().sum()
            counts = counts[counts > 0]
            if not counts.empty:
                df[counts.index] = df[counts.index].fillna(stats[counts.index])
                actions["imputed"][df_name] = {k: int(v) for k, v in counts.items()}
                logger.warning("[%s] imputed (%s) %s", df_name, strategy, actions["imputed"][df_name])
    else:
        logger.info("No feature imputation required")

    return train, candidates, actions


# =============================================================================
# 5. Categorical encoding
# =============================================================================
def encode_secondary_structure(
    df: pd.DataFrame, cfg: dict[str, Any], name: str
) -> tuple[pd.DataFrame, list[str]]:
    """Encode secondary-structure codes (H / E / C).

    One-hot encoding uses the fixed ``const.SS_CATEGORIES`` list, so train and
    candidate tables always get identical columns in identical order, even if a
    category is absent from one of them. Unknown codes produce an all-zero row
    and a warning (one-hot) or raise (ordinal).

    One-hot is the default: H/E/C have no natural numeric order with respect to
    ΔΔG, and with three levels the dimensionality cost is trivial.

    Returns:
        ``(encoded_df, new_columns)``; the raw code column is retained for
        stratification and reporting.

    Raises:
        ValueError: If ordinal encoding meets an unknown code.
    """
    col = const.COL_SECONDARY_STRUCTURE
    encoding = const.Encoding(cfg["features"]["secondary_structure_encoding"])
    df = df.copy()

    unknown = sorted(set(df[col].dropna()) - set(const.SS_CATEGORIES))
    if unknown:
        logger.warning("[%s] unknown %s codes %s", name, col, unknown)

    if encoding is const.Encoding.ONEHOT:
        # Map unknown codes to NaN explicitly: pandas deprecates passing values
        # outside `categories`, and an all-zero row is the documented behaviour.
        known = df[col].where(df[col].isin(const.SS_CATEGORIES))
        cat = pd.Categorical(known, categories=list(const.SS_CATEGORIES))
        dummies = pd.get_dummies(cat, prefix=const.SS_ONEHOT_PREFIX, prefix_sep="_", dtype=np.int8)
        dummies.index = df.index
        df = pd.concat([df, dummies], axis=1)
        new_cols = list(dummies.columns)
    else:
        if unknown:
            raise ValueError(f"[{name}] cannot ordinally encode unknown codes {unknown}")
        df[const.SS_ORDINAL_COLUMN] = df[col].map(const.SS_ORDINAL_MAP).astype(np.int8)
        new_cols = [const.SS_ORDINAL_COLUMN]

    logger.info(
        "[%s] encoded %s (%s) -> %s; counts=%s",
        name,
        col,
        encoding.value,
        new_cols,
        df[col].value_counts().to_dict(),
    )
    return df, new_cols


# =============================================================================
# 6. Feature engineering
# =============================================================================
def engineer_features(df: pd.DataFrame, cfg: dict[str, Any], name: str) -> tuple[pd.DataFrame, list[str]]:
    """Add physicochemical delta features (mutant − wild-type).

    For every entry in ``const.DELTA_PAIRS``::

        delta_<name>     = mutant_<name> − wt_<name>
        abs_delta_<name> = |delta_<name>|          (if features.add_abs_deltas)

    Signed deltas capture direction (e.g. hydrophobic→polar on the surface
    vs. in the core); absolute deltas capture magnitude of perturbation, which
    tree models would otherwise need two splits to express. Relative position
    (``position / protein_length``) makes sequence location comparable across
    proteins of different length.

    Original physicochemical columns are retained.

    Returns:
        ``(df_with_features, new_columns)``.
    """
    f = cfg["features"]
    df = df.copy()
    new_cols: list[str] = []

    for feat, (wt_col, mut_col) in const.DELTA_PAIRS.items():
        delta_col = f"{const.DELTA_PREFIX}{feat}"
        df[delta_col] = df[mut_col] - df[wt_col]
        new_cols.append(delta_col)
        if f["add_abs_deltas"]:
            abs_col = f"{const.ABS_DELTA_PREFIX}{feat}"
            df[abs_col] = df[delta_col].abs()
            new_cols.append(abs_col)

    if f["add_relative_position"]:
        df[const.COL_RELATIVE_POSITION] = df[const.COL_POSITION] / df[const.COL_PROTEIN_LENGTH]
        n_beyond = int((df[const.COL_RELATIVE_POSITION] > 1).sum())
        if n_beyond:
            logger.warning("[%s] %d rows have position > protein_length", name, n_beyond)
        new_cols.append(const.COL_RELATIVE_POSITION)

    logger.info("[%s] engineered %d features: %s", name, len(new_cols), new_cols)
    return df, new_cols


# =============================================================================
# 7. Export
# =============================================================================
def build_feature_manifest(
    cfg: dict[str, Any],
    encoded_cols: list[str],
    engineered_cols: list[str],
    quality_report: dict[str, Any],
) -> dict[str, Any]:
    """Describe the processed tables so downstream phases need no column logic.

    The manifest separates tabular features from embedding features because
    embeddings are reduced (PCA) inside each CV fold in ``train.py``.
    """
    return {
        "id_column": const.COL_MUTATION_ID,
        "group_columns": dict(const.GROUP_COLUMNS),
        "targets": dict(const.TARGETS),
        "tabular_features": [*const.NUMERIC_FEATURE_COLUMNS, *encoded_cols, *engineered_cols],
        "embedding_features": list(const.EMBEDDING_COLUMNS),
        "random_seed": cfg["project"]["random_seed"],
        "quality_report": quality_report,
    }


def export_artifacts(
    train: pd.DataFrame, candidates: pd.DataFrame, manifest: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Path]:
    """Write processed tables (format chosen by file extension) and manifest.

    Returns:
        Mapping of artifact name to written path.

    Raises:
        ValueError: If a configured output extension is not supported.
    """
    resolve_path(cfg, "processed_dir").mkdir(parents=True, exist_ok=True)
    paths = {
        key: resolve_path(cfg, "processed_dir", key)
        for key in ("processed_train", "processed_candidates", "feature_manifest")
    }
    for key, df in (("processed_train", train), ("processed_candidates", candidates)):
        write_table(df, paths[key])
        logger.info("Wrote %s shape=%s -> %s", key, df.shape, paths[key])

    with paths["feature_manifest"].open("w", encoding=const.CONFIG_ENCODING) as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote feature manifest -> %s", paths["feature_manifest"])
    return paths


# =============================================================================
# Orchestration
# =============================================================================
def run_preprocessing(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute the full Phase 1 pipeline and export artifacts.

    Order matters: anomalies are handled on the raw numeric columns *before*
    deltas are computed, so an imputed component yields a consistent delta.

    Returns:
        ``(processed_train, processed_candidates, manifest)``.
    """
    frames = load_raw_data(cfg)
    validate_schema(frames, cfg)

    emb = frames[const.TABLE_EMBEDDINGS]
    train = merge_embeddings(frames[const.TABLE_LABELED], emb, cfg, const.TABLE_LABELED)
    cands = merge_embeddings(frames[const.TABLE_CANDIDATES], emb, cfg, const.TABLE_CANDIDATES)

    report: dict[str, Any] = {
        "before": {
            const.TABLE_LABELED: inspect_anomalies(train, const.TABLE_LABELED),
            const.TABLE_CANDIDATES: inspect_anomalies(cands, const.TABLE_CANDIDATES),
        }
    }
    raw_features = [*const.NUMERIC_FEATURE_COLUMNS, *const.EMBEDDING_COLUMNS]
    train, cands, report["actions"] = handle_anomalies(train, cands, cfg, raw_features)

    train, encoded_cols = encode_secondary_structure(train, cfg, const.TABLE_LABELED)
    cands, _ = encode_secondary_structure(cands, cfg, const.TABLE_CANDIDATES)
    train, engineered_cols = engineer_features(train, cfg, const.TABLE_LABELED)
    cands, _ = engineer_features(cands, cfg, const.TABLE_CANDIDATES)

    report["after"] = {
        const.TABLE_LABELED: inspect_anomalies(train, const.TABLE_LABELED),
        const.TABLE_CANDIDATES: inspect_anomalies(cands, const.TABLE_CANDIDATES),
    }

    # Distribution-shift summary: candidates whose protein/family is unseen in training.
    for level, col in const.GROUP_COLUMNS.items():
        unseen = sorted(set(cands[col]) - set(train[col]))
        n_rows = int(cands[col].isin(unseen).sum())
        report[f"unseen_{level}"] = {"values": unseen, "candidate_rows": n_rows}
        logger.info("Candidates with unseen %s: %d rows across %s", level, n_rows, unseen)

    manifest = build_feature_manifest(cfg, encoded_cols, engineered_cols, report)
    export_artifacts(train, cands, manifest, cfg)
    logger.info(
        "Phase 1 complete: %d tabular + %d embedding features",
        len(manifest["tabular_features"]),
        len(manifest["embedding_features"]),
    )
    return train, cands, manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    run_preprocessing(config)
