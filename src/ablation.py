"""Feature-subset ablation study under the leakage-controlled CV strategy.

Phase 5a of the Medvolt.AI mutation-effect pipeline (assignment Task 6).

Answers: *which information source actually drives predictive performance?*
Each subset named in ``ablation.subsets`` is cross-validated from scratch under
``ablation.strategy`` (default: ``validation.primary_strategy``, i.e. the
family-grouped split), and its pooled out-of-fold metrics are compared against
the configured baseline subset.

The whole pipeline — imputation, scaling and embedding PCA — is rebuilt inside
every training fold for every subset, so no subset borrows statistics from
another, and none sees its validation fold. Only the feature columns change
between subsets; the folds, the seed and the model are held fixed, so the
metric differences are attributable to the features alone.

Feature groups are declared in ``src.common.constants.FEATURE_GROUPS`` and
resolved against the Phase 1 feature manifest, so a group whose columns were
never engineered (e.g. ``deltas`` when ``features.add_abs_deltas`` was off)
contributes nothing rather than failing.

Run directly (requires Phase 1, and Phase 3 unless ``ablation.model`` is set)::

    python -m src.ablation --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import write_table
from src.evaluate import compute_metrics, make_ablation_figures
from src.model import FeatureSet
from src.train import cross_validate_model, load_training_data
from src.validation import generate_folds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Subset:
    """One ablation arm.

    Attributes:
        name: Subset label as written in config.
        groups: Feature groups it includes.
        features: The resolved feature columns, split into tabular and embedding.
    """

    name: str
    groups: list[str]
    features: FeatureSet


# =============================================================================
# Subset construction
# =============================================================================
def resolve_subset(name: str, groups: list[str], manifest_features: FeatureSet) -> Subset:
    """Map feature-group names onto the columns actually present after Phase 1.

    Embedding columns go to the embedding block (so they are PCA-reduced inside
    each fold); every other group goes to the tabular block. Manifest order is
    preserved so column order is deterministic.

    Raises:
        ValueError: If the subset resolves to no columns at all.
    """
    wanted: set[str] = set()
    for group in groups:
        wanted.update(const.FEATURE_GROUPS[group])

    tabular = [column for column in manifest_features.tabular if column in wanted]
    embedding = [column for column in manifest_features.embedding if column in wanted]
    if not tabular and not embedding:
        raise ValueError(f"Ablation subset '{name}' resolves to no available columns (groups={groups})")

    requested = sum(len(const.FEATURE_GROUPS[group]) for group in groups)
    found = len(tabular) + len(embedding)
    if found < requested:
        # Expected when a group is partly optional, e.g. only one of the
        # one-hot/ordinal secondary-structure encodings exists in a given run.
        logger.debug("[%s] %d of %d declared group columns present after Phase 1", name, found, requested)
    return Subset(name, groups, FeatureSet(tabular, embedding))


def build_subsets(cfg: dict[str, Any], manifest_features: FeatureSet) -> list[Subset]:
    """Resolve every configured subset, baseline last so it is logged after its comparators."""
    subsets = [
        resolve_subset(name, groups, manifest_features) for name, groups in cfg["ablation"]["subsets"].items()
    ]
    for subset in subsets:
        logger.info(
            "Subset %-26s groups=%-52s features=%d tabular + %d embedding",
            subset.name,
            ",".join(subset.groups),
            len(subset.features.tabular),
            len(subset.features.embedding),
        )
    return subsets


def resolve_models(cfg: dict[str, Any]) -> dict[str, const.ModelName]:
    """Decide which model to ablate for each target.

    ``ablation.model`` pins one model for both targets; ``null`` reuses whatever
    Phase 3 selected per target, which keeps the ablation faithful to the
    model that will actually be deployed.

    Raises:
        FileNotFoundError: If the model must come from Phase 3 but its metadata is absent.
    """
    configured = cfg["ablation"]["model"]
    if configured is not None:
        return {target: const.ModelName(configured) for target in const.TARGETS.values()}

    metadata_path = resolve_path(cfg, "models_dir", "training_metadata")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"ablation.model is null, so the Phase 3 selection is needed, but {metadata_path} is "
            "missing. Run `python -m src.train` first or set ablation.model in config."
        )
    metadata = json.loads(metadata_path.read_text(encoding=const.CONFIG_ENCODING))
    return {
        target: const.ModelName(choice[const.COL_MODEL]) for target, choice in metadata["selection"].items()
    }


# =============================================================================
# Ablation run
# =============================================================================
def run_ablation(cfg: dict[str, Any]) -> pd.DataFrame:
    """Cross-validate every subset on both targets and tabulate pooled OOF metrics.

    Returns:
        One row per subset x target, with pooled metrics, per-fold spread and
        the signed difference against the baseline subset.
    """
    df, manifest_features = load_training_data(cfg)
    subsets = build_subsets(cfg, manifest_features)
    models = resolve_models(cfg)

    strategy = const.SplitStrategy(cfg["ablation"]["strategy"] or cfg["validation"]["primary_strategy"])
    validation = cfg["validation"]
    folds = generate_folds(
        df, strategy, validation["n_splits"], validation["shuffle"], cfg["project"]["random_seed"]
    )
    metrics = [const.Metric(name) for name in validation["metrics"]]
    logger.info(
        "Ablating %d subsets x %d targets under '%s' (%d folds); models=%s",
        len(subsets),
        len(const.TARGETS),
        strategy.value,
        len(folds),
        {target: model.value for target, model in models.items()},
    )

    rows = []
    for subset in subsets:
        for target in const.TARGETS.values():
            oof, fold_records, _ = cross_validate_model(
                df, folds, cfg, models[target], target, subset.features
            )
            pooled = compute_metrics(
                oof[const.COL_Y_TRUE].to_numpy(), oof[const.COL_Y_PRED].to_numpy(), metrics
            )
            fold_metrics = pd.DataFrame(fold_records)
            rows.append(
                {
                    const.ABL_COL_SUBSET: subset.name,
                    const.ABL_COL_GROUPS: ",".join(subset.groups),
                    const.COL_TARGET: target,
                    const.COL_MODEL: models[target].value,
                    const.CV_COL_STRATEGY: strategy.value,
                    const.ABL_COL_N_TABULAR: len(subset.features.tabular),
                    const.ABL_COL_N_EMBEDDING: len(subset.features.embedding),
                    const.COL_N_SAMPLES: len(oof),
                    **{f"{const.POOLED_PREFIX}{name}": value for name, value in pooled.items()},
                    **{
                        f"{const.FOLD_STD_PREFIX}{metric.value}": float(fold_metrics[metric.value].std())
                        for metric in metrics
                    },
                }
            )
            logger.info(
                "[%s|%s] %s",
                subset.name,
                target,
                ", ".join(f"{name}={value:.3f}" for name, value in pooled.items()),
            )
    return add_baseline_deltas(pd.DataFrame(rows), cfg, metrics)


def add_baseline_deltas(
    table: pd.DataFrame, cfg: dict[str, Any], metrics: list[const.Metric]
) -> pd.DataFrame:
    """Add each subset's signed metric difference against the baseline subset.

    The sign is left raw: for ``rmse``/``mae`` a negative difference means the
    subset is better than the baseline, for ``r2``/``spearman`` a positive one
    does. ``METRIC_GREATER_IS_BETTER`` records which is which.
    """
    baseline_name = cfg["ablation"]["baseline"]
    baseline = table[table[const.ABL_COL_SUBSET] == baseline_name].set_index(const.COL_TARGET)
    out = table.copy()
    for metric in metrics:
        column = f"{const.POOLED_PREFIX}{metric.value}"
        reference = out[const.COL_TARGET].map(baseline[column])
        out[f"{const.ABL_DELTA_PREFIX}{metric.value}"] = out[column] - reference

    selection_metric = cfg["models"]["selection_metric"]
    delta_column = f"{const.ABL_DELTA_PREFIX}{selection_metric}"
    for target in const.TARGETS.values():
        rows = out[out[const.COL_TARGET] == target]
        summary = ", ".join(
            f"{row[const.ABL_COL_SUBSET]}={row[delta_column]:+.3f}" for _, row in rows.iterrows()
        )
        logger.info("[%s] %s vs baseline '%s': %s", target, delta_column, baseline_name, summary)
    return out


def to_markdown(table: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Render the ablation table as a Markdown section for the technical report."""
    baseline = cfg["ablation"]["baseline"]
    strategy = table[const.CV_COL_STRATEGY].iloc[0]
    # Both headline metrics get a level and a delta: a rank correlation says
    # whether the shortlist would be ordered correctly, R2 whether the
    # predicted magnitudes mean anything. A feature group can matter for one
    # and not the other, and reporting only the selection metric hides that.
    columns = [
        const.ABL_COL_SUBSET,
        const.ABL_COL_N_TABULAR,
        const.ABL_COL_N_EMBEDDING,
        f"{const.POOLED_PREFIX}{const.Metric.RMSE.value}",
        *[f"{const.POOLED_PREFIX}{metric.value}" for metric in const.ABL_HEADLINE_METRICS],
        f"{const.POOLED_PREFIX}{const.Metric.PEARSON.value}",
        *[f"{const.ABL_DELTA_PREFIX}{metric.value}" for metric in const.ABL_HEADLINE_METRICS],
    ]
    columns = [column for column in columns if column in table.columns]
    lines = [
        "# Feature-subset ablation",
        "",
        f"- Cross-validation: `{strategy}`, {int(table[const.COL_N_SAMPLES].iloc[0])} pooled "
        "out-of-fold predictions per row",
        f"- Baseline for the delta columns: `{baseline}`",
        f"- Models: {dict(zip(table[const.COL_TARGET], table[const.COL_MODEL], strict=True))}",
        "",
        "Every subset is re-fitted from scratch inside each training fold, so differences are "
        "attributable to the feature columns alone; folds, seed and model are held fixed.",
        "",
        "A `delta_vs_baseline_` column is this subset's score minus the baseline's. For "
        "`r2` and `spearman` a negative delta is a loss; the sign is left raw rather than "
        "normalised so the column can be read straight against the level beside it.",
        "",
    ]
    for target in const.TARGETS.values():
        rows = table[table[const.COL_TARGET] == target]
        if rows.empty:
            continue
        lines += [
            f"## {target}",
            "",
            rows[columns].to_markdown(index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT),
            "",
        ]
    return "\n".join(lines) + "\n"


def export_artifacts(table: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Path]:
    """Write the ablation table as CSV and Markdown, plus the attribution figure.

    Returns:
        Mapping of artifact name to written path.
    """
    paths = {
        key: resolve_path(cfg, "metrics_dir", key)
        for key in ("ablation_summary", "ablation_summary_markdown")
    }
    write_table(table, paths["ablation_summary"])
    paths["ablation_summary_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["ablation_summary_markdown"].write_text(
        to_markdown(table, cfg), encoding=const.CONFIG_ENCODING
    )
    paths.update(make_ablation_figures(table, cfg))
    for key, path in paths.items():
        logger.info("Wrote %s -> %s", key, path)
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    run_ablation_table = run_ablation(config)
    export_artifacts(run_ablation_table, config)
    logger.info("Ablation complete: %d rows", len(run_ablation_table))
