"""Per-candidate scientific interpretation of the top-ranked mutations.

Phase 5b of the Medvolt.AI mutation-effect pipeline (assignment Task 9).

Turns the Top-K table from Phase 4 into a structured physicochemical and
structural write-up: for each selected candidate it reports the two
predictions with their ensemble uncertainty, the driving input features in
context, and the caveats that apply.

Design constraint: **every statement is derived from a value in the data.**
This module composes sentences from measured feature values, percentile ranks
within the candidate pool and fold-ensemble spread. It does not assert a
mechanism it cannot evidence, and it does not consult any external knowledge
about the (synthetic) proteins involved. Qualitative wording comes from
thresholds on the observed distribution of the candidate pool, so a phrase
like "highly conserved" always means a specific percentile, and the numeric
value is printed alongside it.

Run directly (requires Phase 4)::

    python -m src.interpret --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table

logger = logging.getLogger(__name__)

# Percentile cut-offs used to attach words to numbers. These describe the
# candidate pool, not absolute biophysical categories.
LOW_PERCENTILE: float = 0.33
HIGH_PERCENTILE: float = 0.67


@dataclass(frozen=True)
class PoolContext:
    """Distribution of the candidate pool, used to place one candidate in context.

    Attributes:
        quantiles: Per-column low/high cut-offs taken from the pool.
        n_candidates: Size of the pool the percentiles refer to.
    """

    quantiles: dict[str, tuple[float, float]]
    n_candidates: int

    def band(self, column: str, value: float) -> str:
        """Return which third of the pool's distribution a value falls in.

        Worded as thirds rather than low/high because several reported features
        are signed deltas, where "low" would ambiguously suggest small magnitude
        rather than the bottom of the range.
        """
        low, high = self.quantiles[column]
        if value <= low:
            return "lower third"
        return "upper third" if value >= high else "middle third"


def build_pool_context(candidates: pd.DataFrame, columns: list[str]) -> PoolContext:
    """Compute the pool's low/high cut-offs for every reported feature."""
    quantiles = {
        column: (
            float(candidates[column].quantile(LOW_PERCENTILE)),
            float(candidates[column].quantile(HIGH_PERCENTILE)),
        )
        for column in columns
        if column in candidates.columns
    }
    return PoolContext(quantiles, len(candidates))


# =============================================================================
# Data loading
# =============================================================================
def load_reports(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Top-K table and the full scored candidate set from Phase 4.

    Raises:
        FileNotFoundError: If Phase 4 has not been run.
    """
    top_path = resolve_path(cfg, "reports_dir", "top_candidates")
    all_path = resolve_path(cfg, "reports_dir", "all_predictions")
    for path in (top_path, all_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run `python -m src.predict` first")
    top = read_table(top_path)
    everything = read_table(all_path)
    logger.info("Loaded Top %d of %d scored candidates", len(top), len(everything))
    return top, everything


def enrich_with_full_columns(selected: pd.DataFrame, everything: pd.DataFrame) -> pd.DataFrame:
    """Attach columns the Top-K table does not carry (deltas, confidence, flags).

    ``TOP_K_REPORT_COLUMNS`` is deliberately a narrow reporting view, so feature
    values quoted in the write-up are taken from the full scored table, joined
    on mutation id.

    Raises:
        KeyError: If a selected candidate is absent from the full table.
    """
    # Check membership on the key itself: a column-based check is unreliable when
    # the Top-K view already carries every column (nothing left to be NaN).
    absent = sorted(set(selected[const.COL_MUTATION_ID]) - set(everything[const.COL_MUTATION_ID]))
    if absent:
        raise KeyError(f"Top-K candidates absent from the scored table: {absent}")

    missing_cols = [c for c in everything.columns if c not in selected.columns]
    if not missing_cols:
        return selected.copy()
    return selected.merge(
        everything[[const.COL_MUTATION_ID, *missing_cols]],
        on=const.COL_MUTATION_ID,
        how="left",
        validate="one_to_one",
    )


def select_candidates(top: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Choose which ranked candidates to write up.

    Takes the highest-ranked ``interpretation.n_candidates``. If any flagged
    out-of-distribution candidate made the Top K but falls outside that head,
    the lowest-ranked selection is swapped for the best-ranked OOD one, so the
    write-up always discusses at least one domain-adaptation case when one
    exists.
    """
    n = int(cfg["interpretation"]["n_candidates"])
    ranked = top.sort_values(const.COL_PRIORITY_RANK, kind="stable")
    chosen = ranked.head(n)
    if const.COL_OOD_FLAG not in ranked.columns:
        return chosen

    ood = ranked[ranked[const.COL_OOD_FLAG] != const.OOD_IN_DISTRIBUTION]
    if not ood.empty and ood[const.COL_PRIORITY_RANK].min() not in set(chosen[const.COL_PRIORITY_RANK]):
        swap_in = ood.iloc[[0]]
        chosen = pd.concat([chosen.iloc[: n - 1], swap_in]).sort_values(
            const.COL_PRIORITY_RANK, kind="stable"
        )
        logger.info(
            "Included rank %d (%s) as the out-of-distribution case study",
            int(swap_in[const.COL_PRIORITY_RANK].iloc[0]),
            swap_in[const.COL_OOD_FLAG].iloc[0],
        )
    return chosen


# =============================================================================
# Narrative construction
# =============================================================================
def describe_stability(row: pd.Series, uncertainty: float) -> str:
    """Describe the predicted stability effect, stating when it is not distinguishable from zero."""
    ddg = float(row[f"{const.PRED_PREFIX}{const.TARGET_STABILITY}"])
    direction = "stabilizing" if ddg < 0 else "destabilizing"
    sentence = (
        f"Predicted ddG is {ddg:+.3f} kcal/mol ({direction}), with fold-ensemble spread +/-{uncertainty:.3f}."
    )
    if abs(ddg) < uncertainty:
        sentence += (
            " The magnitude is smaller than the ensemble spread, so the sign is not resolved: "
            "treat this as approximately neutral rather than as a confirmed "
            f"{direction} mutation."
        )
    return sentence


def describe_function(row: pd.Series, uncertainty: float) -> str:
    """Describe predicted functional retention on its documented 0-1 scale."""
    score = float(row[f"{const.PRED_PREFIX}{const.TARGET_FUNCTION}"])
    return (
        f"Predicted functional retention is {score:.3f} on the 0-1 scale (1 = wild-type-like), "
        f"with fold-ensemble spread +/-{uncertainty:.3f}."
    )


def describe_features(row: pd.Series, context: PoolContext, columns: list[str]) -> list[str]:
    """Explain each reported feature's value, its pool band, and why it matters.

    The physicochemical reasoning attached to each feature is standard protein
    biophysics stated as a general expectation, never as a claim about this
    specific prediction's internals: the model is a gradient-boosted ensemble
    and these are inputs, not attributions.
    """
    explanations = {
        const.COL_CONSERVATION: (
            "evolutionary conservation at the mutated position; highly conserved positions are "
            "typically under functional or structural constraint, so substitutions there carry "
            "more risk of functional loss"
        ),
        const.COL_SOLVENT_ACCESSIBILITY: (
            "solvent exposure; buried positions tolerate hydrophobic-to-polar changes poorly, "
            "whereas exposed positions are generally more permissive"
        ),
        const.COL_DIST_ACTIVE_SITE: (
            "distance from the active site; substitutions close to the catalytic centre can "
            "disrupt function even when they are thermodynamically tolerated"
        ),
        const.COL_LOCAL_CONFIDENCE: (
            "local structural confidence; low values mean the structural context itself is less "
            "certain, which weakens any structure-based reasoning about this position"
        ),
        f"{const.DELTA_PREFIX}hydrophobicity": (
            "change in residue hydrophobicity (mutant minus wild type); large shifts in a buried "
            "environment tend to perturb core packing"
        ),
        f"{const.DELTA_PREFIX}charge": (
            "change in formal residue charge; introducing or removing charge in a buried or "
            "active-site environment can disrupt electrostatics and salt bridges"
        ),
        f"{const.DELTA_PREFIX}volume": (
            "change in residue volume (A^3); large increases risk steric clashes in packed "
            "regions, while large decreases can leave destabilizing cavities"
        ),
    }
    lines = []
    for column in columns:
        if column not in row.index or column not in context.quantiles:
            continue
        value = float(row[column])
        band = context.band(column, value)
        note = explanations.get(column, "model input feature")
        lines.append(f"- `{column}` = {value:+.3f} ({band} for this pool) — {note}.")
    return lines


def describe_caveats(row: pd.Series) -> list[str]:
    """State the caveats that apply to this candidate, by flag and by rank context."""
    caveats = []
    flag = row.get(const.COL_OOD_FLAG, const.OOD_IN_DISTRIBUTION)
    if flag == const.OOD_UNSEEN_FAMILY:
        caveats.append(
            f"**Domain adaptation:** protein family `{row[const.COL_PROTEIN_FAMILY]}` is absent from "
            "the labeled training set. The prediction rests entirely on the protein embedding "
            "generalizing across families, which family-grouped CV showed to be the weakest "
            "regime. Treat this candidate as exploratory and prioritize it for experimental "
            "confirmation rather than acting on the point estimate."
        )
    elif flag == const.OOD_UNSEEN_PROTEIN:
        caveats.append(
            f"**Partial extrapolation:** protein `{row[const.COL_PROTEIN_ID]}` has no labeled "
            "mutations, though its family is represented. Protein-grouped CV is the applicable "
            "performance estimate here, not the more optimistic random split."
        )
    else:
        caveats.append(
            f"Protein `{row[const.COL_PROTEIN_ID]}` is represented in the labeled training data, "
            "so this is an interpolation within the training distribution."
        )
    caveats.append(
        "The priority score is rank-normalized **within this 180-candidate pool**, so it ranks "
        "this mutation against the others rather than measuring absolute merit."
    )
    return caveats


def interpret_candidate(row: pd.Series, context: PoolContext, cfg: dict[str, Any]) -> str:
    """Compose the full Markdown write-up for one candidate."""
    columns = list(cfg["interpretation"]["report_features"])
    stability_sd = float(row[f"{const.PRED_STD_PREFIX}{const.TARGET_STABILITY}"])
    function_sd = float(row[f"{const.PRED_STD_PREFIX}{const.TARGET_FUNCTION}"])
    mutation = f"{row[const.COL_WT_AA]}{int(row[const.COL_POSITION])}{row[const.COL_MUTANT_AA]}"
    lines = [
        f"### Rank {int(row[const.COL_PRIORITY_RANK])} — {row[const.COL_MUTATION_ID]} "
        f"({row[const.COL_PROTEIN_ID]}, {row[const.COL_PROTEIN_FAMILY]}): {mutation}",
        "",
        f"**Substitution.** {row[const.COL_WT_AA]} to {row[const.COL_MUTANT_AA]} at position "
        f"{int(row[const.COL_POSITION])}, in a `{row[const.COL_SECONDARY_STRUCTURE]}` "
        "secondary-structure element "
        f"(priority score {float(row[const.COL_PRIORITY_SCORE]):.3f}, flag "
        f"`{row.get(const.COL_OOD_FLAG, const.OOD_IN_DISTRIBUTION)}`).",
        "",
        "**Why the model ranks it here.**",
        f"{describe_stability(row, stability_sd)} {describe_function(row, function_sd)} "
        "The score balances these two objectives, so a candidate can rank highly on strong "
        "predicted function while being only neutral on stability.",
        "",
        "**Driving input features.**",
        *describe_features(row, context, columns),
        "",
        "**Confidence and caveats.**",
        *[f"- {caveat}" for caveat in describe_caveats(row)],
        "",
    ]
    return "\n".join(lines)


def build_report(selected: pd.DataFrame, everything: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Assemble the interpretation document for every selected candidate."""
    context = build_pool_context(everything, list(cfg["interpretation"]["report_features"]))
    enriched = enrich_with_full_columns(selected, everything)
    top_k = int(cfg["prioritization"]["top_k"])
    header = [
        "# Scientific interpretation of top-ranked candidates",
        "",
        f"Case studies for {len(enriched)} of the Top {top_k} ranked mutations, placed "
        f"against the distribution of all {context.n_candidates} scored candidates. Qualitative "
        f"bands are the pool's {LOW_PERCENTILE:.0%}/{HIGH_PERCENTILE:.0%} quantiles, and the "
        "underlying number is always printed. For signed delta features a lower-third value "
        "means the bottom of the pool's range, not a small magnitude.",
        "",
        "The physicochemical notes state general expectations from protein biophysics to make the "
        "inputs interpretable. They are not feature attributions: the model is a gradient-boosted "
        "tree ensemble, and no per-prediction attribution was computed.",
        "",
    ]
    bodies = [interpret_candidate(row, context, cfg) for _, row in enriched.iterrows()]
    return "\n".join([*header, *bodies])


def export_report(markdown: str, cfg: dict[str, Any]) -> Path:
    """Write the interpretation Markdown document.

    Returns:
        The path written.
    """
    path = resolve_path(cfg, "reports_dir", "interpretation_markdown")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding=const.CONFIG_ENCODING)
    logger.info("Wrote interpretation -> %s", path)
    return path


def run_interpretation(cfg: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Load Phase 4 output, select candidates and write their interpretation.

    Returns:
        ``(selected_candidates, markdown)``.
    """
    top, everything = load_reports(cfg)
    selected = select_candidates(top, cfg)
    markdown = build_report(selected, everything, cfg)
    export_report(markdown, cfg)
    logger.info("Interpreted ranks %s", sorted(int(r) for r in selected[const.COL_PRIORITY_RANK]))
    return selected, markdown


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    run_interpretation(config)
