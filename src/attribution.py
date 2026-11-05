"""Feature attribution for the selected model (assignment Task 7).

Answers "which inputs does the production model actually use?" for the model
``src/train.py`` selected, using the fold pipelines it saved. Nothing is
refit: the pipelines scored here are the very ones whose out-of-fold metrics
are published, so the attribution describes the reported model rather than a
fresh one that happens to share its hyperparameters.

Run directly (requires Phase 3 with ``models.save_fold_pipelines: true``)::

    python -m src.attribution --config config/config.yaml

**Two kinds of importance, and why both are reported.**

*Permutation importance* shuffles one input column on a fold's **validation**
rows and measures how far the score falls. It is the honest measure: it is
computed out of sample, it is defined on the original named columns (the
shuffle happens before the pipeline, so imputation, scaling and PCA all see
the permuted column), and it is model-agnostic. Its weakness is correlated
features — if two columns carry the same information, shuffling either one
alone barely hurts, and both look unimportant.

*Impurity importance* reads the gradient-boosted trees' accumulated split
gains. It is nearly free to compute, but it is measured on training rows and
is biased toward features with many distinct values. It is also defined on the
model's **post-PCA** inputs, so its embedding entries are principal components
(``embedding_pc0`` …) rather than named columns and mean little individually.

Reporting both, and saying which is which in the output table, is the
defensible position: they disagree in informative ways, and a reader who is
given only the convenient one cannot tell.

**Why the embedding block is permuted as a unit.** With
``attribution.group_embeddings_as_block: true`` all 256 embedding columns are
shuffled together, in the same row order, and scored once. Permuting them one
at a time would be 256× slower and would measure almost nothing: the columns
are a single protein-level vector and are heavily correlated, so individually
they are the textbook case where permutation importance under-reports. The
block result is directly comparable with the ablation study's
``B_embeddings_only`` arm, which asks the same question by a different route.

Aggregation across folds is a mean with a standard deviation, over the five
family-grouped folds. The standard deviation is the number to read before
trusting a ranking: a feature whose importance swings more between folds than
its mean is worth is not a stable finding on 840 rows and 7 families.
"""

from __future__ import annotations

import argparse
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import write_table
from src.evaluate import compute_metrics, make_attribution_figures
from src.model import FeatureSet
from src.train import load_training_data
from src.validation import Fold, generate_folds

logger = logging.getLogger(__name__)


# =============================================================================
# Loading the fold pipelines Phase 3 saved
# =============================================================================
def load_fold_pipelines(cfg: dict[str, Any]) -> dict[str, dict[int, Pipeline]]:
    """Load the selected model's fold pipelines, keyed by target then fold index.

    The fold index is parsed from the filename rather than inferred from the
    listing order, so a missing fold cannot silently shift the others.

    Raises:
        FileNotFoundError: If Phase 3 has not been run, or ran with
            ``models.save_fold_pipelines: false``.
    """
    metadata_path = resolve_path(cfg, "models_dir", "training_metadata")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Training metadata not found: {metadata_path}. Run `python -m src.train` first."
        )
    metadata = json.loads(metadata_path.read_text(encoding=const.CONFIG_ENCODING))
    relatives = metadata.get("fold_pipelines", [])
    if not relatives:
        raise FileNotFoundError(
            "No fold pipelines were saved. Re-run Phase 3 with models.save_fold_pipelines: true."
        )

    models_dir = resolve_path(cfg, "models_dir")
    pipelines: dict[str, dict[int, Pipeline]] = {t: {} for t in const.TARGETS.values()}
    for relative in relatives:
        path = models_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Fold pipeline listed in metadata is missing: {path}")
        name = Path(relative).name
        target = next(t for t in const.TARGETS.values() if f"__{t}__" in name)
        fold = int(name.split("__")[1].removeprefix("fold"))
        pipelines[target][fold] = joblib.load(path)
    logger.info(
        "Loaded fold pipelines: %s", {t: sorted(folds) for t, folds in pipelines.items()}
    )
    return pipelines


def permutation_units(features: FeatureSet, cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Map each permutation unit to the columns shuffled together.

    Every tabular column is its own unit. The embedding block is one unit when
    ``attribution.group_embeddings_as_block`` is set, and otherwise one unit
    per embedding column.
    """
    units: dict[str, list[str]] = {column: [column] for column in features.tabular}
    if not features.embedding:
        return units
    if cfg["attribution"]["group_embeddings_as_block"]:
        units[const.GROUP_EMBEDDINGS] = list(features.embedding)
    else:
        units.update({column: [column] for column in features.embedding})
    return units


def feature_group_of(name: str, units: dict[str, list[str]]) -> str:
    """Return the feature group a permutation unit belongs to.

    The embedding block keeps its own name, and a PCA component or an
    unrecognised column is reported as unassigned rather than guessed at.
    """
    if name == const.GROUP_EMBEDDINGS:
        return const.GROUP_EMBEDDINGS
    columns = set(units.get(name, [name]))
    for group, members in const.FEATURE_GROUPS.items():
        if columns & set(members):
            return group
    return const.ATTR_GROUP_UNKNOWN


# =============================================================================
# Importance strategies
# =============================================================================
class ImportanceStrategy(ABC):
    """One way of scoring how much a model relies on each input.

    The strategies differ in what they measure, what they are defined on, and
    what they cost, so they are classes rather than branches on an enum:
    adding SHAP means adding a subclass and an :class:`~src.common.constants.
    ImportanceKind` member, with no existing code to edit.

    Attributes:
        kind: The enum member this strategy implements, recorded in the
            ``importance_kind`` column so a reader always knows which measure
            produced a number.
    """

    kind: const.ImportanceKind

    def __init__(self, cfg: dict[str, Any], features: FeatureSet) -> None:
        self.cfg = cfg
        self.features = features

    @abstractmethod
    def score_fold(
        self, pipeline: Pipeline, df: pd.DataFrame, fold: Fold, target: str
    ) -> dict[str, float]:
        """Importance of every unit for one fold. Empty when not applicable."""

    def units(self) -> dict[str, list[str]]:
        """Map each scored unit to the original columns behind it.

        The default is one unit per original column, with the embedding block
        optionally collapsed into one. A strategy defined on transformed
        inputs overrides this.
        """
        return permutation_units(self.features, self.cfg)


class PermutationImportance(ImportanceStrategy):
    """Shuffle a column on held-out rows and measure the skill lost.

    The honest measure, and the one the figures use: it is computed out of
    sample, it is model-agnostic, and because the shuffle happens *before* the
    pipeline, imputation, scaling and PCA all see the permuted column — so
    importance is defined on the original named features rather than on
    principal components.

    Its known weakness is correlated features: if two columns carry the same
    information, shuffling either alone barely hurts and both look
    unimportant. That is precisely why the embedding block is shuffled as one
    unit.
    """

    kind = const.ImportanceKind.PERMUTATION

    def score_fold(
        self, pipeline: Pipeline, df: pd.DataFrame, fold: Fold, target: str
    ) -> dict[str, float]:
        """Score every unit on this fold's validation rows."""
        section = self.cfg["attribution"]
        rows = df.iloc[fold.val_idx]
        return self.permute(
            pipeline,
            rows[self.features.all],
            rows[target].to_numpy(dtype=float),
            self.units(),
            const.Metric(section["scoring_metric"]),
            int(section["n_repeats"]),
            # Fold-specific but deterministic: folds must not share a
            # permutation order, and a re-run must reproduce it.
            int(self.cfg["project"]["random_seed"]) + fold.index,
        )

    @staticmethod
    def score_frame(
        pipeline: Pipeline, frame: pd.DataFrame, y_true: np.ndarray, metric: const.Metric
    ) -> float:
        """Score a fitted pipeline on ``frame`` with one configured metric."""
        predicted = np.asarray(pipeline.predict(frame), dtype=float)
        return compute_metrics(y_true, predicted, [metric])[metric.value]

    @classmethod
    def permute(
        cls,
        pipeline: Pipeline,
        frame: pd.DataFrame,
        y_true: np.ndarray,
        units: dict[str, list[str]],
        metric: const.Metric,
        n_repeats: int,
        seed: int,
    ) -> dict[str, float]:
        """Importance of every unit on one set of held-out rows.

        Each unit's columns are shuffled together under a single permutation
        of the row order, which preserves the correlations *within* the unit
        while destroying its relationship to the target. The score is the mean
        drop across ``n_repeats`` shuffles, signed so that **larger is always
        worse for the model**, i.e. larger means the feature mattered more::

            importance = baseline - permuted      (r2, pearson, spearman)
            importance = permuted - baseline      (rmse, mae)

        A small negative importance is expected noise, not a bug: shuffling an
        unused column can improve the score by chance.
        """
        baseline = cls.score_frame(pipeline, frame, y_true, metric)
        higher_is_better = const.METRIC_GREATER_IS_BETTER[metric]
        rng = np.random.default_rng(seed)

        importances: dict[str, float] = {}
        for unit, columns in units.items():
            drops = []
            for _ in range(n_repeats):
                shuffled = frame.copy()
                order = rng.permutation(len(shuffled))
                shuffled[columns] = shuffled[columns].to_numpy()[order]
                permuted = cls.score_frame(pipeline, shuffled, y_true, metric)
                drops.append(baseline - permuted if higher_is_better else permuted - baseline)
            importances[unit] = float(np.mean(drops))
        return importances


class ImpurityImportance(ImportanceStrategy):
    """Read the tree ensemble's accumulated split gains.

    Nearly free to compute, but measured on *training* rows and biased toward
    features with many distinct values. It is also defined on the model's
    **post-PCA** inputs, so its embedding entries are principal components
    rather than named columns and mean little individually — which is why they
    are labelled ``embedding_pc{i}`` and why this strategy overrides
    :meth:`units`.
    """

    kind = const.ImportanceKind.IMPURITY

    def units(self) -> dict[str, list[str]]:
        """Tabular columns map to themselves; PCA components map to nothing.

        A component is not any original column, so it is given an empty
        membership list and falls through to the unassigned feature group,
        rather than being misattributed to the embeddings it summarises.
        """
        return {column: [column] for column in self.features.tabular}

    def score_fold(
        self, pipeline: Pipeline, df: pd.DataFrame, fold: Fold, target: str
    ) -> dict[str, float]:
        """Read ``feature_importances_`` from the fitted estimator, if it has one.

        Returns an empty mapping for an estimator without it (Ridge, for
        instance) rather than substituting coefficients, which are not on a
        comparable scale.

        Raises:
            ValueError: If the estimator reports fewer importances than the
                tabular columns it was given, which would mean the names and
                the values have come apart.
        """
        estimator = pipeline[-1]
        raw = getattr(estimator, "feature_importances_", None)
        if raw is None:
            logger.info(
                "%s exposes no feature_importances_; skipping impurity importance",
                type(estimator).__name__,
            )
            return {}
        values = np.asarray(raw, dtype=float)
        n_reduced = values.size - len(self.features.tabular)
        if n_reduced < 0:
            raise ValueError(
                f"Estimator reports {values.size} importances, fewer than the "
                f"{len(self.features.tabular)} tabular columns it was given"
            )
        names = [
            *self.features.tabular,
            *[const.ATTR_PCA_FEATURE_TEMPLATE.format(index=i) for i in range(n_reduced)],
        ]
        return dict(zip(names, (float(v) for v in values), strict=True))


# Enum member -> strategy. Adding a measure means adding a class and one line.
STRATEGY_REGISTRY: dict[const.ImportanceKind, type[ImportanceStrategy]] = {
    const.ImportanceKind.PERMUTATION: PermutationImportance,
    const.ImportanceKind.IMPURITY: ImpurityImportance,
}


def build_strategies(cfg: dict[str, Any], features: FeatureSet) -> list[ImportanceStrategy]:
    """Instantiate the strategies named in ``attribution.kinds``, in config order."""
    return [
        STRATEGY_REGISTRY[const.ImportanceKind(name)](cfg, features)
        for name in cfg["attribution"]["kinds"]
    ]


# =============================================================================
# Aggregation across folds
# =============================================================================
def aggregate_folds(
    per_fold: dict[int, dict[str, float]],
    target: str,
    kind: const.ImportanceKind,
    units: dict[str, list[str]],
) -> pd.DataFrame:
    """Average each unit's importance over folds, with the between-fold spread.

    ``importance_share`` normalises the *positive* part of the mean importances
    to sum to 1, so the column reads as "share of the attributable signal".
    Negative means are noise around zero and are excluded from the
    denominator; including them would let a noisy feature inflate everyone
    else's share.
    """
    if not per_fold:
        return pd.DataFrame()
    table = pd.DataFrame(per_fold).T  # rows = folds, columns = units
    means = table.mean(axis=0)
    positive_total = float(means.clip(lower=0).sum())
    rows = [
        {
            const.COL_TARGET: target,
            const.ATTR_COL_KIND: kind.value,
            const.ATTR_COL_FEATURE: unit,
            const.ATTR_COL_GROUP: feature_group_of(unit, units),
            const.ATTR_COL_MEAN: float(means[unit]),
            const.ATTR_COL_STD: float(table[unit].std(ddof=0)),
            const.ATTR_COL_SHARE: (
                max(float(means[unit]), 0.0) / positive_total if positive_total > 0 else float("nan")
            ),
            const.ATTR_COL_N_FOLDS: int(table[unit].notna().sum()),
        }
        for unit in table.columns
    ]
    out = pd.DataFrame(rows).sort_values(const.ATTR_COL_MEAN, ascending=False, ignore_index=True)
    out[const.ATTR_COL_RANK] = range(1, len(out) + 1)
    return out


def group_importances(table: pd.DataFrame) -> pd.DataFrame:
    """Sum per-feature importance within each feature group.

    This is the table that speaks to the ablation study: it says how much of
    the model's realised skill each *source of information* accounts for,
    rather than how much any single column does.
    """
    if table.empty:
        return pd.DataFrame()
    grouped = (
        table.groupby([const.COL_TARGET, const.ATTR_COL_KIND, const.ATTR_COL_GROUP], sort=False)
        .agg(
            **{
                const.ATTR_COL_MEAN: (const.ATTR_COL_MEAN, "sum"),
                const.ATTR_COL_SHARE: (const.ATTR_COL_SHARE, "sum"),
                "n_features": (const.ATTR_COL_FEATURE, "count"),
            }
        )
        .reset_index()
    )
    return grouped.sort_values(
        [const.COL_TARGET, const.ATTR_COL_KIND, const.ATTR_COL_MEAN],
        ascending=[True, True, False],
        ignore_index=True,
    )


# =============================================================================
# Orchestration
# =============================================================================
def compute_attribution(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attribute the selected model's skill to its inputs, fold by fold.

    The loop is strategy-agnostic: it asks each configured
    :class:`ImportanceStrategy` for a per-fold score and aggregates whatever
    comes back, so a new measure needs no change here.

    Returns:
        ``(per_feature, per_group)``. Both are empty only if no fold pipeline
        supports any configured importance kind.
    """
    df, features = load_training_data(cfg)
    pipelines = load_fold_pipelines(cfg)
    strategies = build_strategies(cfg, features)
    section = cfg["attribution"]

    split = const.SplitStrategy(cfg["validation"]["primary_strategy"])
    validation = cfg["validation"]
    folds: list[Fold] = generate_folds(
        df, split, validation["n_splits"], validation["shuffle"],
        int(cfg["project"]["random_seed"]),
    )
    logger.info(
        "Attributing over %d folds | kinds=%s | metric=%s | repeats=%d",
        len(folds),
        [strategy.kind.value for strategy in strategies],
        section["scoring_metric"],
        int(section["n_repeats"]),
    )

    tables = []
    for target, fold_models in pipelines.items():
        for strategy in strategies:
            per_fold: dict[int, dict[str, float]] = {}
            for fold in folds:
                pipeline = fold_models.get(fold.index)
                if pipeline is None:
                    logger.warning("No %s pipeline for fold %d; skipping", target, fold.index)
                    continue
                scores = strategy.score_fold(pipeline, df, fold, target)
                if scores:
                    per_fold[fold.index] = scores
            table = aggregate_folds(per_fold, target, strategy.kind, strategy.units())
            if table.empty:
                continue
            tables.append(table)
            logger.info(
                "[%s|%s] top: %s",
                target,
                strategy.kind.value,
                ", ".join(
                    f"{row[const.ATTR_COL_FEATURE]}={row[const.ATTR_COL_MEAN]:+.4f}"
                    for _, row in table.head(3).iterrows()
                ),
            )
    if not tables:
        return pd.DataFrame(), pd.DataFrame()
    per_feature = pd.concat(tables, ignore_index=True)
    return per_feature, group_importances(per_feature)


def to_markdown(per_feature: pd.DataFrame, per_group: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Render the attribution tables as a Markdown section for the report."""
    section = cfg["attribution"]
    top_k = int(section["top_k"])
    strategy = cfg["validation"]["primary_strategy"]
    lines = [
        "# Feature attribution",
        "",
        f"- Model explained: the one Phase 3 selected per target, as fitted inside each of the "
        f"`{strategy}` folds; nothing is refit here",
        f"- Permutation importance: `{section['scoring_metric']}` lost when a column is shuffled "
        f"on that fold's **validation** rows, {section['n_repeats']} shuffles per feature, "
        "averaged over folds",
        f"- Embeddings permuted as a single block: `{section['group_embeddings_as_block']}`",
        "",
        "Larger is more important for every metric: the sign is normalised so that a positive "
        "value always means the model got worse when the feature was shuffled. Small negative "
        "values are noise around zero, which is what an unused feature looks like.",
        "",
        "`importance_std` is the spread across folds. Read it before trusting a rank: on seven "
        "families a feature whose between-fold spread exceeds its mean is not a stable finding.",
        "",
    ]
    if not per_group.empty:
        lines += ["## By feature group", ""]
        for target in const.TARGETS.values():
            rows = per_group[per_group[const.COL_TARGET] == target]
            if rows.empty:
                continue
            lines += [
                f"### {target}",
                "",
                rows.drop(columns=const.COL_TARGET).to_markdown(
                    index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT
                ),
                "",
            ]
    lines += [f"## Top {top_k} features", ""]
    for target in const.TARGETS.values():
        for kind in dict.fromkeys(per_feature[const.ATTR_COL_KIND]):
            rows = per_feature[
                (per_feature[const.COL_TARGET] == target)
                & (per_feature[const.ATTR_COL_KIND] == kind)
            ].head(top_k)
            if rows.empty:
                continue
            lines += [
                f"### {target} — {kind}",
                "",
                rows.drop(columns=[const.COL_TARGET, const.ATTR_COL_KIND]).to_markdown(
                    index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT
                ),
                "",
            ]
    return "\n".join(lines) + "\n"


def export_artifacts(
    per_feature: pd.DataFrame, per_group: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, Path]:
    """Write both tables, the Markdown write-up and the figures.

    Returns:
        Mapping of artifact name to written path.
    """
    paths: dict[str, Path] = {}
    for key, table in (
        ("feature_importances", per_feature),
        ("feature_group_importances", per_group),
    ):
        paths[key] = resolve_path(cfg, "metrics_dir", key)
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        write_table(table, paths[key])
    paths["attribution_markdown"] = resolve_path(cfg, "metrics_dir", "attribution_markdown")
    paths["attribution_markdown"].write_text(
        to_markdown(per_feature, per_group, cfg), encoding=const.CONFIG_ENCODING
    )
    paths.update(make_attribution_figures(per_feature, per_group, cfg))
    for key, path in paths.items():
        logger.info("Wrote %s -> %s", key, path)
    return paths


def run_attribution(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Task 7: attribute the selected model's skill and write the artifacts.

    Returns:
        ``(per_feature, per_group)``.

    Raises:
        ValueError: If no importance could be computed at all, which means the
            configured kinds do not apply to the selected estimator.
    """
    per_feature, per_group = compute_attribution(cfg)
    if per_feature.empty:
        raise ValueError(
            "No feature importances could be computed. Check attribution.kinds against the "
            "selected estimator (impurity needs a tree ensemble)."
        )
    export_artifacts(per_feature, per_group, cfg)
    logger.info(
        "Attribution complete: %d feature rows, %d group rows",
        len(per_feature),
        len(per_group),
    )
    return per_feature, per_group


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    run_attribution(config)
