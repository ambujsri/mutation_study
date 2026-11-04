"""Candidate inference, uncertainty estimation and Top-K prioritization.

Phase 4 of the Medvolt.AI mutation-effect pipeline. Takes the processed
candidate table from Phase 1 and the fitted artifacts from Phase 3, and
produces a ranked, flagged, justified shortlist.

Steps:

1. ``load_artifacts``      — selected models, their CV fold pipelines and the
                             feature lists recorded at training time.
2. ``predict_ensemble``    — predict each target with every fold pipeline; the
                             mean is the point estimate and the standard
                             deviation across folds is the uncertainty.
3. ``flag_distribution_shift`` — mark candidates whose protein, or whole
                             family, never appeared in training.
4. ``score_candidates``    — normalize both objectives within the candidate
                             pool, combine with the configured weights, and
                             subtract the configured uncertainty penalty.
5. ``build_top_k`` / ``export_artifacts`` — write all scored candidates, plus
                             the Top-K table as CSV and Markdown.

Sign conventions (``README_DATASET.txt``): ``stability_ddG`` is negative for
stabilizing mutations, so the stability objective maximizes **-ddG**;
``functional_score`` is already "higher is better".

Two honesty caveats are built into the output rather than hidden:
    * Scores are **relative to this candidate pool**, because normalization is
      pool-internal. They rank candidates against each other; they are not
      absolute measures of merit.
    * The fold spread measures disagreement between models trained on
      different family subsets. It is a lower bound on true predictive
      uncertainty: it captures neither label noise nor error shared by every
      fold, which is exactly what unseen families risk.

Run directly::

    python -m src.predict --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table, write_table
from src.model import FeatureSet

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TrainedArtifacts:
    """Everything Phase 3 left behind that inference needs.

    Attributes:
        features: Feature columns the pipelines were fitted on, in order.
        fold_pipelines: Fold pipelines per target, in fold order.
        final_pipelines: Model per target refit on all labeled data (may be empty
            if Phase 3 ran with ``models.refit_on_full_data: false``).
        selection: Phase 3 model-selection record, per target.
        metadata: The full training metadata document, for provenance.
    """

    features: FeatureSet
    fold_pipelines: dict[str, list[Pipeline]]
    final_pipelines: dict[str, Pipeline]
    selection: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


# =============================================================================
# 1. Artifact & data loading
# =============================================================================
def load_artifacts(cfg: dict[str, Any]) -> TrainedArtifacts:
    """Load the training metadata and every fitted pipeline it references.

    Fold pipelines are grouped by target and kept in the fold order recorded in
    the metadata, so the ensemble is assembled deterministically.

    Raises:
        FileNotFoundError: If Phase 3 artifacts are missing.
        ValueError: If a target has fewer fold pipelines than
            ``prioritization.min_fold_models``.
    """
    models_dir = resolve_path(cfg, "models_dir")
    metadata_path = resolve_path(cfg, "models_dir", "training_metadata")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {metadata_path} (run Phase 3 first)")
    metadata = json.loads(metadata_path.read_text(encoding=const.CONFIG_ENCODING))

    features = FeatureSet(metadata["features"]["tabular"], metadata["features"]["embedding"])
    fold_pipelines: dict[str, list[Pipeline]] = {target: [] for target in const.TARGETS.values()}
    for relative in metadata["fold_pipelines"]:
        path = models_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Fold pipeline listed in metadata is missing: {path}")
        target = next(t for t in const.TARGETS.values() if f"__{t}__" in Path(relative).name)
        fold_pipelines[target].append(joblib.load(path))

    minimum = cfg["prioritization"]["min_fold_models"]
    for target, pipelines in fold_pipelines.items():
        if len(pipelines) < minimum:
            raise ValueError(
                f"Target '{target}' has {len(pipelines)} fold pipelines, need at least {minimum} "
                "for an uncertainty estimate. Re-run Phase 3 with models.save_fold_pipelines: true."
            )

    final_pipelines = {
        target: joblib.load(models_dir / name) for target, name in metadata["final_models"].items()
    }
    logger.info(
        "Loaded artifacts: %s fold pipelines per target, %d final models, trained with seed %s",
        {t: len(p) for t, p in fold_pipelines.items()},
        len(final_pipelines),
        metadata["random_seed"],
    )
    return TrainedArtifacts(features, fold_pipelines, final_pipelines, metadata["selection"], metadata)


def load_candidates(cfg: dict[str, Any], features: FeatureSet) -> pd.DataFrame:
    """Load the processed candidate table and check it carries every training feature.

    Raises:
        FileNotFoundError: If Phase 1 has not been run.
        KeyError: If features the pipelines expect are absent.
    """
    df = read_table(resolve_path(cfg, "processed_dir", "processed_candidates"))
    missing = [column for column in features.all if column not in df.columns]
    if missing:
        raise KeyError(f"Candidate table lacks {len(missing)} training features, e.g. {missing[:5]}")
    logger.info("Loaded %d candidates across %d proteins", len(df), df[const.COL_PROTEIN_ID].nunique())
    return df


# =============================================================================
# 2. Ensemble inference
# =============================================================================
def predict_ensemble(df: pd.DataFrame, artifacts: TrainedArtifacts, cfg: dict[str, Any]) -> pd.DataFrame:
    """Predict both targets with the fold ensemble and record per-candidate spread.

    For each target the fold pipelines each produce a prediction; the mean
    becomes ``pred_<target>`` (when ``point_estimate`` is ``ensemble_mean``) and
    the sample standard deviation across folds becomes ``pred_std_<target>``.
    The refit-on-all-data model is always reported as ``pred_final_<target>``
    so the two can be compared.

    Returns:
        ``df`` with prediction, spread and fold-count columns added.
    """
    point_estimate = const.PointEstimate(cfg["prioritization"]["point_estimate"])
    features = artifacts.features
    inputs = df[features.all]
    out = df.copy()

    for target, pipelines in artifacts.fold_pipelines.items():
        fold_predictions = np.vstack([np.asarray(p.predict(inputs), dtype=float) for p in pipelines])
        ensemble_mean = fold_predictions.mean(axis=0)
        # ddof=1: spread between the fold models as a sample, not a population.
        spread = fold_predictions.std(axis=0, ddof=1)

        final_pipeline = artifacts.final_pipelines.get(target)
        if final_pipeline is not None:
            out[f"{const.PRED_FINAL_PREFIX}{target}"] = np.asarray(
                final_pipeline.predict(inputs), dtype=float
            )
        elif point_estimate is const.PointEstimate.FINAL_MODEL:
            raise ValueError(
                f"point_estimate='{point_estimate.value}' needs a final model for '{target}', "
                "but Phase 3 ran with models.refit_on_full_data: false"
            )

        if point_estimate is const.PointEstimate.ENSEMBLE_MEAN:
            out[f"{const.PRED_PREFIX}{target}"] = ensemble_mean
        else:
            out[f"{const.PRED_PREFIX}{target}"] = out[f"{const.PRED_FINAL_PREFIX}{target}"]
        out[f"{const.PRED_STD_PREFIX}{target}"] = spread
        out[f"{const.COL_N_FOLD_MODELS}_{target}"] = len(pipelines)

        logger.info(
            "[%s] %s over %d fold models: mean=%.3f sd across candidates=%.3f | "
            "mean fold disagreement=%.3f (max %.3f)",
            target,
            point_estimate.value,
            len(pipelines),
            out[f"{const.PRED_PREFIX}{target}"].mean(),
            out[f"{const.PRED_PREFIX}{target}"].std(),
            spread.mean(),
            spread.max(),
        )
    return out


# =============================================================================
# 3. Distribution-shift flagging
# =============================================================================
def flag_distribution_shift(
    candidates: pd.DataFrame, train: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Flag candidates whose protein, or whole family, was unseen during training.

    An unseen family is the stronger warning and takes precedence in
    ``ood_flag``: predictions there rest only on the protein embedding
    generalizing across families, which family-grouped CV showed to be the
    weakest regime.

    Returns:
        ``candidates`` with ``is_unseen_protein``, ``is_unseen_family`` and the
        combined ``ood_flag`` column added.
    """
    out = candidates.copy()
    seen_proteins = set(train[const.COL_PROTEIN_ID])
    seen_families = set(train[const.COL_PROTEIN_FAMILY])
    out[const.COL_UNSEEN_PROTEIN] = ~out[const.COL_PROTEIN_ID].isin(seen_proteins)
    out[const.COL_UNSEEN_FAMILY] = ~out[const.COL_PROTEIN_FAMILY].isin(seen_families)

    flag_families = bool(cfg["prioritization"]["ood_flag_unseen_family"])
    out[const.COL_OOD_FLAG] = np.select(
        [out[const.COL_UNSEEN_FAMILY] & flag_families, out[const.COL_UNSEEN_PROTEIN]],
        [const.OOD_UNSEEN_FAMILY, const.OOD_UNSEEN_PROTEIN],
        default=const.OOD_IN_DISTRIBUTION,
    )
    logger.info(
        "Distribution shift: %d/%d candidates from unseen families %s, %d from unseen proteins",
        int(out[const.COL_UNSEEN_FAMILY].sum()),
        len(out),
        sorted(set(out.loc[out[const.COL_UNSEEN_FAMILY], const.COL_PROTEIN_FAMILY])),
        int(out[const.COL_UNSEEN_PROTEIN].sum()),
    )
    return out


# =============================================================================
# 4. Multi-objective scoring
# =============================================================================
def normalize_objective(values: pd.Series, method: const.ScoreNormalization) -> pd.Series:
    """Map values onto ``[0, 1]`` with higher always better.

    ``rank`` uses the average percentile rank, which is robust to outliers and
    spaces candidates evenly; ``minmax`` preserves the real spacing between
    predictions but lets a single extreme value compress the rest. A constant
    input yields 0.5 everywhere under both methods, so a degenerate objective
    cannot silently decide the ranking.
    """
    if values.dropna().unique().size <= 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    if method is const.ScoreNormalization.RANK:
        return values.rank(pct=True, method="average").astype(float)
    low, high = values.min(), values.max()
    return ((values - low) / (high - low)).astype(float)


def score_candidates(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Combine the stability and function objectives into one priority score.

    The score is::

        priority = w_stab * norm(-ddG) + w_func * norm(functional_score)
                   - penalty * norm(combined uncertainty)

    Weights are rescaled to sum to 1 so the objective part stays within
    ``[0, 1]``. ``combined uncertainty`` adds the two fold-spread columns in
    quadrature (they come from independently trained models), then normalizes
    it the same way as the objectives so the penalty is on a comparable scale.

    Returns:
        ``df`` with normalized contributions, uncertainty, score and rank added,
        sorted best-first.

    Raises:
        ValueError: If the configured weights are not all finite and
            non-negative with a positive sum.
    """
    p = cfg["prioritization"]
    method = const.ScoreNormalization(p["score_normalization"])
    weights = {
        const.TARGET_STABILITY: float(p["weights"]["stability"]),
        const.TARGET_FUNCTION: float(p["weights"]["function"]),
    }
    total = sum(weights.values())
    if not np.isfinite(total) or total <= 0 or any(w < 0 for w in weights.values()):
        raise ValueError(f"prioritization.weights must be non-negative with a positive sum, got {weights}")
    weights = {target: weight / total for target, weight in weights.items()}

    out = df.copy()
    objective = pd.Series(0.0, index=out.index)
    for target, weight in weights.items():
        prediction = out[f"{const.PRED_PREFIX}{target}"]
        # Stabilizing mutations have negative ddG, so maximize -ddG.
        oriented = -prediction if target == const.TARGET_STABILITY else prediction
        normalized = normalize_objective(oriented, method)
        out[f"{const.NORM_PREFIX}{target}"] = normalized
        objective += weight * normalized

    spreads = np.vstack([out[f"{const.PRED_STD_PREFIX}{t}"].to_numpy(dtype=float) for t in weights])
    out[const.COL_TOTAL_UNCERTAINTY] = np.sqrt((spreads**2).sum(axis=0))
    out[const.COL_NORM_UNCERTAINTY] = normalize_objective(out[const.COL_TOTAL_UNCERTAINTY], method)

    penalty = float(p["uncertainty_penalty"])
    out[const.COL_PRIORITY_SCORE] = objective - penalty * out[const.COL_NORM_UNCERTAINTY]
    out = out.sort_values(
        [const.COL_PRIORITY_SCORE, const.COL_MUTATION_ID], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    out[const.COL_PRIORITY_RANK] = np.arange(1, len(out) + 1)

    logger.info(
        "Scored %d candidates: weights=%s normalization=%s uncertainty_penalty=%.2f",
        len(out),
        {t: round(w, 3) for t, w in weights.items()},
        method.value,
        penalty,
    )
    return out


def justify(row: pd.Series, cfg: dict[str, Any]) -> str:
    """Write one short, factual sentence explaining a candidate's standing.

    Only quantities present in the row are cited: predicted effects, the
    percentile of each objective within the pool, fold disagreement and any
    distribution-shift caveat. No claim is made that is not in the data.
    """
    ddg = row[f"{const.PRED_PREFIX}{const.TARGET_STABILITY}"]
    func = row[f"{const.PRED_PREFIX}{const.TARGET_FUNCTION}"]
    direction = "stabilizing" if ddg < 0 else "destabilizing"
    parts = [
        f"Predicted {direction} (ddG {ddg:+.2f}) with functional retention {func:.2f}",
        f"stability percentile {100 * row[f'{const.NORM_PREFIX}{const.TARGET_STABILITY}']:.0f}, "
        f"function percentile {100 * row[f'{const.NORM_PREFIX}{const.TARGET_FUNCTION}']:.0f}",
        f"fold disagreement {row[const.COL_TOTAL_UNCERTAINTY]:.3f}",
    ]
    if row[const.COL_OOD_FLAG] == const.OOD_UNSEEN_FAMILY:
        parts.append(f"family {row[const.COL_PROTEIN_FAMILY]} unseen in training, so treat as exploratory")
    elif row[const.COL_OOD_FLAG] == const.OOD_UNSEEN_PROTEIN:
        parts.append(f"protein {row[const.COL_PROTEIN_ID]} unseen in training, family represented")
    else:
        parts.append("protein represented in training data")
    return "; ".join(parts) + "."


def build_top_k(scored: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Take the highest-scoring ``top_k`` candidates and add a justification column.

    Returns:
        A reporting table holding only ``TOP_K_REPORT_COLUMNS``, best first.
    """
    top_k = int(cfg["prioritization"]["top_k"])
    if top_k > len(scored):
        logger.warning("top_k=%d exceeds the %d candidates available; reporting all", top_k, len(scored))
    table = scored.head(top_k).copy()
    table[const.COL_JUSTIFICATION] = table.apply(lambda row: justify(row, cfg), axis=1)

    missing = [column for column in const.TOP_K_REPORT_COLUMNS if column not in table.columns]
    if missing:
        raise KeyError(f"Top-K report columns missing from scored candidates: {missing}")
    report = table[list(const.TOP_K_REPORT_COLUMNS)]
    logger.info(
        "Top %d: %d from unseen families; score range %.3f to %.3f",
        len(report),
        int((table[const.COL_OOD_FLAG] == const.OOD_UNSEEN_FAMILY).sum()),
        report[const.COL_PRIORITY_SCORE].min(),
        report[const.COL_PRIORITY_SCORE].max(),
    )
    return report


# =============================================================================
# 5. Export
# =============================================================================
def to_markdown(
    report: pd.DataFrame, scored: pd.DataFrame, cfg: dict[str, Any], artifacts: TrainedArtifacts
) -> str:
    """Render the Top-K table as a Markdown section for the technical report.

    The justification column is moved below the table as a numbered list, since
    full sentences inside a Markdown table are unreadable.
    """
    p = cfg["prioritization"]
    table = report.drop(columns=const.COL_JUSTIFICATION)
    models = {target: choice[const.COL_MODEL] for target, choice in artifacts.selection.items()}
    lines = [
        f"# Top {len(report)} prioritized candidate mutations",
        "",
        f"- Candidates scored: {len(scored)}; shortlisted: {len(report)}",
        f"- Models: {models}, selected under `{artifacts.metadata['primary_strategy']}` cross-validation",
        f"- Point estimate: `{p['point_estimate']}` over "
        f"{ {t: len(v) for t, v in artifacts.fold_pipelines.items()} } fold models per target",
        f"- Score: {p['weights']['stability']} x normalized(-ddG) + {p['weights']['function']} x "
        f"normalized(functional_score) - {p['uncertainty_penalty']} x normalized(uncertainty), "
        f"`{p['score_normalization']}` normalization",
        f"- Random seed: {artifacts.metadata['random_seed']}",
        "",
        "Scores are relative to this candidate pool, not absolute merit. Fold disagreement is a lower "
        "bound on predictive uncertainty: it misses error shared by every fold, which is the main risk "
        "for unseen families.",
        "",
        table.to_markdown(index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT),
        "",
        "## Justification",
        "",
    ]
    lines += [
        f"{row[const.COL_PRIORITY_RANK]}. **{row[const.COL_MUTATION_ID]}** — {row[const.COL_JUSTIFICATION]}"
        for _, row in report.iterrows()
    ]
    return "\n".join(lines) + "\n"


def export_artifacts(
    scored: pd.DataFrame, report: pd.DataFrame, markdown: str, cfg: dict[str, Any]
) -> dict[str, Path]:
    """Write all scored candidates, the Top-K table and the Markdown report.

    Embedding columns are dropped from the full predictions file: they are
    reproducible from Phase 1 and would make the report unreadable.

    Returns:
        Mapping of artifact name to written path.
    """
    paths = {
        key: resolve_path(cfg, "rankings_dir", key)
        for key in ("all_predictions", "top_candidates", "top_candidates_markdown")
    }
    reportable = scored.drop(columns=[c for c in scored.columns if c.startswith(const.EMBEDDING_PREFIX)])
    write_table(reportable, paths["all_predictions"])
    write_table(report, paths["top_candidates"])
    paths["top_candidates_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["top_candidates_markdown"].write_text(markdown, encoding=const.CONFIG_ENCODING)
    for key, path in paths.items():
        logger.info("Wrote %s -> %s", key, path)
    return paths


# =============================================================================
# Orchestration
# =============================================================================
def run_prediction(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Phase 4: ensemble inference, flagging, scoring and Top-K export.

    Returns:
        ``(scored_candidates, top_k_report)``.
    """
    artifacts = load_artifacts(cfg)
    candidates = load_candidates(cfg, artifacts.features)
    train = read_table(resolve_path(cfg, "processed_dir", "processed_train"))

    predicted = predict_ensemble(candidates, artifacts, cfg)
    flagged = flag_distribution_shift(predicted, train, cfg)
    scored = score_candidates(flagged, cfg)
    report = build_top_k(scored, cfg)
    export_artifacts(scored, report, to_markdown(report, scored, cfg, artifacts), cfg)
    logger.info("Phase 4 complete: %d candidates scored, top %d reported", len(scored), len(report))
    return scored, report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    run_prediction(config)
