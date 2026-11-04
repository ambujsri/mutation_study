"""Unit and integration tests for ``src.interpret`` (Task 9).

The emphasis is on scientific honesty of the generated text: that the sign of a
prediction is not asserted when it falls inside the ensemble spread, that
out-of-distribution candidates always carry the domain-adaptation caveat, and
that every quoted feature value matches the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src import interpret as ip
from src.common import constants as const

REPORT_FEATURES: list[str] = [
    const.COL_CONSERVATION,
    const.COL_SOLVENT_ACCESSIBILITY,
    f"{const.DELTA_PREFIX}hydrophobicity",
]


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture()
def interpret_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config with a small interpretation scope."""
    cfg["interpretation"]["n_candidates"] = 2
    cfg["interpretation"]["report_features"] = list(REPORT_FEATURES)
    cfg["prioritization"]["top_k"] = 5
    return cfg


def make_row(
    rank: int,
    ddg: float = 0.5,
    func: float = 0.8,
    std_ddg: float = 0.05,
    std_func: float = 0.01,
    flag: str = const.OOD_IN_DISTRIBUTION,
    conservation: float = 0.5,
) -> dict[str, Any]:
    """Build one scored-candidate record with known values."""
    return {
        const.COL_PRIORITY_RANK: rank,
        const.COL_MUTATION_ID: f"CAND{rank:04d}",
        const.COL_PROTEIN_ID: f"PROT{rank:03d}",
        const.COL_PROTEIN_FAMILY: f"PF{rank:02d}",
        const.COL_WT_AA: "A",
        const.COL_POSITION: 100 + rank,
        const.COL_MUTANT_AA: "V",
        const.COL_SECONDARY_STRUCTURE: const.SS_CATEGORIES[0],
        const.COL_PRIORITY_SCORE: 1.0 - rank / 100,
        const.COL_OOD_FLAG: flag,
        f"{const.PRED_PREFIX}{const.TARGET_STABILITY}": ddg,
        f"{const.PRED_PREFIX}{const.TARGET_FUNCTION}": func,
        f"{const.PRED_STD_PREFIX}{const.TARGET_STABILITY}": std_ddg,
        f"{const.PRED_STD_PREFIX}{const.TARGET_FUNCTION}": std_func,
        const.COL_CONSERVATION: conservation,
        const.COL_SOLVENT_ACCESSIBILITY: 0.5,
        f"{const.DELTA_PREFIX}hydrophobicity": -1.5,
    }


@pytest.fixture()
def scored() -> pd.DataFrame:
    """A pool of scored candidates with a spread of conservation values."""
    rows = [make_row(rank, conservation=rank / 10) for rank in range(1, 10)]
    return pd.DataFrame(rows)


@pytest.fixture()
def context(scored: pd.DataFrame) -> ip.PoolContext:
    """Pool context derived from the scored fixture."""
    return ip.build_pool_context(scored, REPORT_FEATURES)


# =============================================================================
# Pool context
# =============================================================================
def test_pool_context_bands_split_distribution(context: ip.PoolContext, scored: pd.DataFrame) -> None:
    """Bands report which third of the pool a value falls in."""
    low, high = context.quantiles[const.COL_CONSERVATION]

    assert context.band(const.COL_CONSERVATION, low - 1) == "lower third"
    assert context.band(const.COL_CONSERVATION, high + 1) == "upper third"
    assert context.band(const.COL_CONSERVATION, (low + high) / 2) == "middle third"
    assert context.n_candidates == len(scored)


def test_pool_context_skips_absent_columns(scored: pd.DataFrame) -> None:
    """A configured feature missing from the data is skipped rather than raising."""
    ctx = ip.build_pool_context(scored, [*REPORT_FEATURES, "not_a_column"])

    assert "not_a_column" not in ctx.quantiles


# =============================================================================
# Honesty of the generated narrative
# =============================================================================
def test_stability_sign_not_asserted_within_uncertainty() -> None:
    """When |ddG| is below the ensemble spread, the text refuses to claim a direction."""
    row = pd.Series(make_row(1, ddg=-0.01, std_ddg=0.08))

    text = ip.describe_stability(row, 0.08)

    assert "not resolved" in text
    assert "approximately neutral" in text


def test_stability_sign_asserted_when_clear() -> None:
    """A prediction well outside the spread is described with its direction and no hedge."""
    row = pd.Series(make_row(1, ddg=1.2, std_ddg=0.05))

    text = ip.describe_stability(row, 0.05)

    assert "destabilizing" in text
    assert "not resolved" not in text


def test_stabilizing_direction_follows_sign_convention() -> None:
    """Negative ddG is called stabilizing, per the dataset's documented convention."""
    row = pd.Series(make_row(1, ddg=-1.5, std_ddg=0.05))

    assert "stabilizing" in ip.describe_stability(row, 0.05)


def test_function_description_states_scale() -> None:
    """The functional score is reported with its 0-1 meaning, not as a bare number."""
    row = pd.Series(make_row(1, func=0.83))

    text = ip.describe_function(row, 0.02)

    assert "0.830" in text
    assert "wild-type-like" in text


def test_feature_lines_quote_exact_values(context: ip.PoolContext, interpret_cfg: dict[str, Any]) -> None:
    """Each reported feature line cites the row's actual value."""
    row = pd.Series(make_row(1, conservation=0.42))

    lines = ip.describe_features(row, context, REPORT_FEATURES)

    assert len(lines) == len(REPORT_FEATURES)
    assert any("+0.420" in line for line in lines)
    assert all(line.startswith("- `") for line in lines)


def test_feature_lines_skip_missing_columns(context: ip.PoolContext) -> None:
    """A feature absent from the row is omitted instead of producing a blank claim."""
    row = pd.Series(make_row(1)).drop(labels=[const.COL_CONSERVATION])

    lines = ip.describe_features(row, context, REPORT_FEATURES)

    assert not any(const.COL_CONSERVATION in line for line in lines)


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (const.OOD_UNSEEN_FAMILY, "Domain adaptation"),
        (const.OOD_UNSEEN_PROTEIN, "Partial extrapolation"),
        (const.OOD_IN_DISTRIBUTION, "represented in the labeled training data"),
    ],
)
def test_caveats_match_distribution_flag(flag: str, expected: str) -> None:
    """Each flag produces its specific caveat, and the pool-relative caveat is always present."""
    row = pd.Series(make_row(1, flag=flag))

    caveats = ip.describe_caveats(row)

    assert any(expected in caveat for caveat in caveats)
    assert any("within this 180-candidate pool" in caveat for caveat in caveats)


def test_unseen_family_caveat_names_the_family() -> None:
    """The domain-adaptation caveat identifies which family is unrepresented."""
    row = pd.Series(make_row(7, flag=const.OOD_UNSEEN_FAMILY))

    caveats = " ".join(ip.describe_caveats(row))

    assert row[const.COL_PROTEIN_FAMILY] in caveats
    assert "exploratory" in caveats


# =============================================================================
# Candidate selection
# =============================================================================
def test_select_takes_highest_ranked(interpret_cfg: dict[str, Any], scored: pd.DataFrame) -> None:
    """By default the top ``n_candidates`` by rank are chosen."""
    selected = ip.select_candidates(scored, interpret_cfg)

    assert selected[const.COL_PRIORITY_RANK].tolist() == [1, 2]


def test_select_swaps_in_out_of_distribution_case(interpret_cfg: dict[str, Any]) -> None:
    """A flagged candidate outside the head replaces the last pick, so OOD is always discussed."""
    rows = [make_row(1), make_row(2), make_row(3, flag=const.OOD_UNSEEN_FAMILY)]

    selected = ip.select_candidates(pd.DataFrame(rows), interpret_cfg)

    assert selected[const.COL_PRIORITY_RANK].tolist() == [1, 3]
    assert const.OOD_UNSEEN_FAMILY in set(selected[const.COL_OOD_FLAG])


def test_select_does_not_swap_when_ood_already_included(interpret_cfg: dict[str, Any]) -> None:
    """No swap happens if a flagged candidate is already in the head."""
    rows = [make_row(1), make_row(2, flag=const.OOD_UNSEEN_FAMILY), make_row(3)]

    selected = ip.select_candidates(pd.DataFrame(rows), interpret_cfg)

    assert selected[const.COL_PRIORITY_RANK].tolist() == [1, 2]


def test_select_handles_all_in_distribution(interpret_cfg: dict[str, Any], scored: pd.DataFrame) -> None:
    """With no flagged candidates at all, selection is simply the top of the ranking."""
    assert set(scored[const.COL_OOD_FLAG]) == {const.OOD_IN_DISTRIBUTION}

    selected = ip.select_candidates(scored, interpret_cfg)

    assert len(selected) == interpret_cfg["interpretation"]["n_candidates"]


# =============================================================================
# Enrichment and report assembly
# =============================================================================
def test_enrich_adds_columns_absent_from_top_table(scored: pd.DataFrame) -> None:
    """Feature columns missing from the narrow Top-K view are recovered from the scored table."""
    narrow = scored[[const.COL_MUTATION_ID, const.COL_PRIORITY_RANK]].head(2)

    enriched = ip.enrich_with_full_columns(narrow, scored)

    assert const.COL_CONSERVATION in enriched.columns
    assert len(enriched) == 2


def test_enrich_detects_unknown_candidate(scored: pd.DataFrame) -> None:
    """A Top-K candidate absent from the scored table is an error, not a row of NaN."""
    narrow = pd.DataFrame({const.COL_MUTATION_ID: ["GHOST"], const.COL_PRIORITY_RANK: [1]})

    with pytest.raises(KeyError, match="absent from the scored table"):
        ip.enrich_with_full_columns(narrow, scored)


def test_report_contains_section_per_candidate(interpret_cfg: dict[str, Any], scored: pd.DataFrame) -> None:
    """Each selected candidate gets a titled section with all four narrative blocks."""
    selected = ip.select_candidates(scored, interpret_cfg)

    markdown = ip.build_report(selected, scored, interpret_cfg)

    for _, row in selected.iterrows():
        assert f"Rank {int(row[const.COL_PRIORITY_RANK])}" in markdown
        assert row[const.COL_MUTATION_ID] in markdown
    assert markdown.count("**Why the model ranks it here.**") == len(selected)
    assert markdown.count("**Driving input features.**") == len(selected)
    assert markdown.count("**Confidence and caveats.**") == len(selected)


def test_report_states_top_k_not_selection_size(interpret_cfg: dict[str, Any], scored: pd.DataFrame) -> None:
    """The header distinguishes how many were written up from the size of the Top K."""
    selected = ip.select_candidates(scored, interpret_cfg)

    markdown = ip.build_report(selected, scored, interpret_cfg)

    assert (
        f"Case studies for {len(selected)} of the Top {interpret_cfg['prioritization']['top_k']}" in markdown
    )


def test_report_disclaims_feature_attribution(interpret_cfg: dict[str, Any], scored: pd.DataFrame) -> None:
    """The report states the notes are not attributions, since none were computed."""
    markdown = ip.build_report(ip.select_candidates(scored, interpret_cfg), scored, interpret_cfg)

    assert "not feature attributions" in markdown


def test_mutation_label_formats_substitution(interpret_cfg: dict[str, Any], scored: pd.DataFrame) -> None:
    """Each section title carries the conventional wtPOSmut substitution label."""
    markdown = ip.build_report(scored.head(1), scored, interpret_cfg)

    assert "A101V" in markdown


# =============================================================================
# End-to-end
# =============================================================================
def test_run_interpretation_requires_phase4(interpret_cfg: dict[str, Any], tmp_path: Path) -> None:
    """Without Phase 4 output the error names the command to run."""
    interpret_cfg["paths"]["reports_dir"] = str(tmp_path / "empty")

    with pytest.raises(FileNotFoundError, match="src.predict"):
        ip.run_interpretation(interpret_cfg)


def test_run_interpretation_writes_markdown(interpret_cfg: dict[str, Any], scored: pd.DataFrame) -> None:
    """The document is written where configured and covers the selected candidates."""
    reports = Path(interpret_cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    scored.head(5).to_csv(reports / interpret_cfg["files"]["top_candidates"], index=False)
    scored.to_csv(reports / interpret_cfg["files"]["all_predictions"], index=False)

    selected, markdown = ip.run_interpretation(interpret_cfg)

    path = reports / interpret_cfg["files"]["interpretation_markdown"]
    assert path.is_file()
    assert path.read_text() == markdown
    assert len(selected) == interpret_cfg["interpretation"]["n_candidates"]
