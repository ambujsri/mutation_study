"""Evaluation, metric aggregation, model selection and visual quality control.

This module owns every question of the form *"how good is it, and how do we
know?"*. It is imported by ``src.train`` (which calls it after each fold fit)
and by ``src.ablation``, and it can also be run on its own against the
artifacts a previous training run left behind::

    python -m src.evaluate --config config/config.yaml

Metrics are reported two ways:

* **per fold**, then mean ± std across folds; and
* **pooled** over all out-of-fold (OOF) predictions at once.

Family-grouped folds are small and uneven (1-2 families each), so pooled OOF
metrics are the primary, lower-variance estimate and the one used for model
selection. ``log_optimism_gap`` quantifies how much better each model looks
under a random split than under the grouped one, which is the single most
important number for judging whether reported performance would survive
deployment on unseen proteins.

Run as a script, the module reloads the saved OOF predictions and metric
summary, re-derives the headline table and the Task 3 **baseline benchmark**
(every model × target × strategy), and writes the evaluation figures to
``paths.figures_dir`` — nothing is recomputed from the models, so this stage is
cheap and safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table, write_table
from src.model import CVResults, FloatArray

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(y_true: FloatArray, y_pred: FloatArray, metrics: list[const.Metric]) -> dict[str, float]:
    """Compute the requested regression metrics.

    Spearman is ``NaN`` when either vector is constant (rank correlation is
    undefined); it is never silently replaced by 0.
    """
    calculators = {
        const.Metric.RMSE: lambda: float(np.sqrt(mean_squared_error(y_true, y_pred))),
        const.Metric.MAE: lambda: float(mean_absolute_error(y_true, y_pred)),
        const.Metric.R2: lambda: float(r2_score(y_true, y_pred)),
        const.Metric.PEARSON: lambda: (
            float("nan")
            if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0
            else float(pearsonr(y_true, y_pred).statistic)
        ),
        const.Metric.SPEARMAN: lambda: (
            float("nan")
            if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0
            else float(spearmanr(y_true, y_pred).statistic)
        ),
    }
    return {metric.value: calculators[metric]() for metric in metrics}


def pooled_metrics(oof: pd.DataFrame, metrics: list[const.Metric]) -> dict[str, float]:
    """Compute metrics over every out-of-fold prediction in ``oof`` at once."""
    return compute_metrics(
        oof[const.COL_Y_TRUE].to_numpy(dtype=float), oof[const.COL_Y_PRED].to_numpy(dtype=float), metrics
    )


def configured_metrics(cfg: dict[str, Any]) -> list[const.Metric]:
    """Return ``validation.metrics`` as enum members, in config order."""
    return [const.Metric(name) for name in cfg["validation"]["metrics"]]


# =============================================================================
# Aggregation
# =============================================================================
def summarize_metrics(results: CVResults, cfg: dict[str, Any]) -> pd.DataFrame:
    """Combine pooled-OOF metrics with per-fold mean and std per strategy × model × target."""
    metrics = configured_metrics(cfg)
    keys = [const.CV_COL_STRATEGY, const.COL_MODEL, const.COL_TARGET]
    rows = []
    for key_values, group in results.oof.groupby(keys, sort=False):
        pooled = pooled_metrics(group, metrics)
        rows.append(
            {
                **dict(zip(keys, key_values, strict=True)),
                const.COL_N_SAMPLES: len(group),
                **{f"{const.POOLED_PREFIX}{k}": v for k, v in pooled.items()},
            }
        )
    summary = pd.DataFrame(rows)
    per_fold = results.fold_metrics.groupby(keys, sort=False)[[m.value for m in metrics]]
    spread = (
        per_fold.mean()
        .add_prefix(const.FOLD_MEAN_PREFIX)
        .join(per_fold.std().add_prefix(const.FOLD_STD_PREFIX))
    )
    return summary.merge(spread.reset_index(), on=keys, how="left")


def select_best_models(summary: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pick the best model per target by the pooled selection metric under the primary strategy.

    Ties are broken by model order in ``models.candidates`` (first wins), which
    keeps selection deterministic.

    Returns:
        ``{target: {"model", "metric", "value", "strategy", "ranking"}}`` where
        ``ranking`` lists every candidate's score, best first.

    Raises:
        ValueError: If no model has a finite score for a target.
    """
    metric = const.Metric(cfg["models"]["selection_metric"])
    primary = cfg["validation"]["primary_strategy"]
    column = f"{const.POOLED_PREFIX}{metric.value}"
    ascending = not const.METRIC_GREATER_IS_BETTER[metric]
    selection: dict[str, dict[str, Any]] = {}
    for target in const.TARGETS.values():
        rows = summary[(summary[const.CV_COL_STRATEGY] == primary) & (summary[const.COL_TARGET] == target)]
        rows = rows[np.isfinite(rows[column])]
        if rows.empty:
            raise ValueError(f"No model has a finite {column} for target '{target}' under '{primary}'")
        ranked = rows.sort_values(column, ascending=ascending, kind="stable")
        best = ranked.iloc[0]
        selection[target] = {
            const.COL_MODEL: best[const.COL_MODEL],
            "metric": column,
            "value": float(best[column]),
            const.CV_COL_STRATEGY: primary,
            "ranking": {r[const.COL_MODEL]: float(r[column]) for _, r in ranked.iterrows()},
        }
        logger.info(
            "Selected %s for %s: %s=%.3f under %s",
            best[const.COL_MODEL],
            target,
            column,
            best[column],
            primary,
        )
    return selection


def log_optimism_gap(summary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Log how much better each model looks under the random split than under the primary one.

    A large gap means the random split rewards memorising protein/family
    identity, i.e. it would over-promise performance on the OOD candidates.

    Returns:
        One row per model × target with both strategies' scores and the signed
        gap, or an empty frame when there is no random split to compare against.
    """
    metric = f"{const.POOLED_PREFIX}{cfg['models']['selection_metric']}"
    primary = cfg["validation"]["primary_strategy"]
    random_ = const.SplitStrategy.RANDOM.value
    if random_ not in set(summary[const.CV_COL_STRATEGY]) or primary == random_:
        return pd.DataFrame()
    pivot = summary.pivot_table(
        index=[const.COL_MODEL, const.COL_TARGET], columns=const.CV_COL_STRATEGY, values=metric
    )
    rows = []
    for (model, target), row in pivot.iterrows():
        logger.info(
            "Optimism gap [%s|%s] %s: random=%.3f vs %s=%.3f (diff %+.3f)",
            model,
            target,
            metric,
            row[random_],
            primary,
            row[primary],
            row[random_] - row[primary],
        )
        rows.append(
            {
                const.COL_MODEL: model,
                const.COL_TARGET: target,
                const.EVAL_COL_METRIC: metric,
                random_: float(row[random_]),
                primary: float(row[primary]),
                const.EVAL_COL_OPTIMISM_GAP: float(row[random_] - row[primary]),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Standalone evaluation of saved artifacts
# =============================================================================
def load_evaluation_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the OOF predictions, metric summary and model selection written by Phase 3.

    Raises:
        FileNotFoundError: If the training stage has not been run.
    """
    selection_path = resolve_path(cfg, "metrics_dir", "model_selection")
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Model selection not found: {selection_path}. Run `python -m src.train` first."
        )
    oof = read_table(resolve_path(cfg, "metrics_dir", "oof_predictions"))
    summary = read_table(resolve_path(cfg, "metrics_dir", "cv_metrics_summary"))
    selection = json.loads(selection_path.read_text(encoding=const.CONFIG_ENCODING))
    logger.info("Loaded %d OOF predictions and %d metric rows", len(oof), len(summary))
    return oof, summary, selection


def evaluation_strategy(cfg: dict[str, Any]) -> str:
    """Return the CV strategy the evaluation report focuses on."""
    return str(cfg["evaluation"]["strategy"] or cfg["validation"]["primary_strategy"])


def selected_oof(
    oof: pd.DataFrame, selection: dict[str, dict[str, Any]], strategy: str
) -> pd.DataFrame:
    """Keep only the rows belonging to each target's selected model under ``strategy``."""
    keep = [
        (oof[const.CV_COL_STRATEGY] == strategy)
        & (oof[const.COL_TARGET] == target)
        & (oof[const.COL_MODEL] == choice[const.COL_MODEL])
        for target, choice in selection.items()
    ]
    if not keep:
        return oof.iloc[:0]
    mask = keep[0]
    for extra in keep[1:]:
        mask = mask | extra
    return oof[mask].copy()


def headline_table(oof: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Recompute pooled metrics per target from the selected models' OOF predictions."""
    metrics = configured_metrics(cfg)
    rows = []
    for (target, model), group in oof.groupby([const.COL_TARGET, const.COL_MODEL], sort=False):
        rows.append(
            {
                const.COL_TARGET: target,
                const.COL_MODEL: model,
                const.COL_N_SAMPLES: len(group),
                **{f"{const.POOLED_PREFIX}{k}": v for k, v in pooled_metrics(group, metrics).items()},
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Baseline benchmark (assignment Task 3)
# =============================================================================
def benchmark_table(summary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Tabulate every baseline model on both targets under every CV strategy.

    This is the Task 3 deliverable view of the cross-validation ``src.train``
    already ran: one row per model × target × strategy with all configured
    pooled metrics, plus the per-fold spread and the signed gap against the
    optimistic random split. No model is refit — the table is a presentation of
    the saved out-of-fold metrics, so it cannot disagree with them.

    Returns:
        Rows ordered by target, then by config model order, then by strategy
        order, so the table reads the way the config declares the experiment.
    """
    metrics = configured_metrics(cfg)
    pooled_columns = [f"{const.POOLED_PREFIX}{metric.value}" for metric in metrics]
    strategies = [s for s in cfg["validation"]["strategies"] if s in set(summary[const.CV_COL_STRATEGY])]
    models = [m for m in cfg["models"]["candidates"] if m in set(summary[const.COL_MODEL])]
    random_ = const.SplitStrategy.RANDOM.value
    selection_column = f"{const.POOLED_PREFIX}{cfg['models']['selection_metric']}"

    indexed = summary.set_index([const.COL_TARGET, const.COL_MODEL, const.CV_COL_STRATEGY])
    rows = []
    for target in const.TARGETS.values():
        for model in models:
            for strategy in strategies:
                key = (target, model, strategy)
                if key not in indexed.index:
                    continue
                row = indexed.loc[key]
                record: dict[str, Any] = {
                    const.COL_TARGET: target,
                    const.COL_MODEL: model,
                    const.CV_COL_STRATEGY: strategy,
                    const.COL_N_SAMPLES: int(row[const.COL_N_SAMPLES]),
                }
                record.update({column: float(row[column]) for column in pooled_columns})
                fold_std = f"{const.FOLD_STD_PREFIX}{cfg['models']['selection_metric']}"
                record[fold_std] = float(row[fold_std]) if fold_std in row else float("nan")
                # How much of this score the optimistic split would have added.
                optimistic = (target, model, random_)
                record[const.EVAL_COL_OPTIMISM_GAP] = (
                    float(indexed.loc[optimistic, selection_column]) - float(row[selection_column])
                    if optimistic in indexed.index and strategy != random_
                    else float("nan")
                )
                rows.append(record)
    return pd.DataFrame(rows)


def benchmark_to_markdown(table: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Render the baseline benchmark as a Markdown section for the technical report."""
    primary = cfg["validation"]["primary_strategy"]
    selection_metric = cfg["models"]["selection_metric"]
    lines = [
        "# Baseline model benchmark",
        "",
        f"- Models: {', '.join(dict.fromkeys(table[const.COL_MODEL]))}",
        f"- Targets: {', '.join(const.TARGETS.values())}, modelled independently",
        f"- Primary strategy: `{primary}`; selection metric: "
        f"`{const.POOLED_PREFIX}{selection_metric}`",
        "",
        "All metrics are pooled over every out-of-fold prediction. Each model is re-fitted from "
        "scratch inside every training fold, with imputation, scaling and embedding PCA as steps "
        "**inside** the pipeline, so no fold sees its validation rows at fit time.",
        "",
        f"`{const.EVAL_COL_OPTIMISM_GAP}` is the random split's "
        f"`{const.POOLED_PREFIX}{selection_metric}` minus this row's: the credit a model gets for "
        "recognising proteins and families it has already seen.",
        "",
    ]
    for target in const.TARGETS.values():
        rows = table[table[const.COL_TARGET] == target]
        if rows.empty:
            continue
        lines += [
            f"## {target}",
            "",
            rows.drop(columns=const.COL_TARGET).to_markdown(
                index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT
            ),
            "",
        ]
    return "\n".join(lines) + "\n"


# =============================================================================
# Comparative analysis across model families and splitting strategies (Task 5)
# =============================================================================
def unified_summary(cfg: dict[str, Any]) -> pd.DataFrame:
    """Concatenate the sklearn baselines' metrics with the neural network's.

    Both stages write the same schema — one row per strategy × model × target
    with ``pooled_*``, ``fold_mean_*`` and ``fold_std_*`` columns — so the two
    tables stack without reconciliation. The network's file is optional: Task 5
    reports on whatever has been trained, and says in the log what is missing
    rather than silently presenting a partial comparison as a complete one.
    """
    tables = [read_table(resolve_path(cfg, "metrics_dir", "cv_metrics_summary"))]
    for key, module in (
        ("torch_cv_metrics", "src.train_torch"),
        ("torch_v2_cv_metrics", "src.train_torch_v2"),
    ):
        path = resolve_path(cfg, "metrics_dir", key)
        if not path.is_file():
            logger.info("No metrics at %s: run `python -m %s` to include it", path, module)
            continue
        network = read_table(path)
        logger.info("Including %d rows from %s in the comparison", len(network), path.name)
        tables.append(network)
    return pd.concat(tables, ignore_index=True) if len(tables) > 1 else tables[0]


def model_comparison_table(summary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build the single table Task 5 is judged on: every model × target × strategy.

    This is the join of the two comparative axes the assignment asks for:

    * **model family** — a penalised linear fit, two tree ensembles and a
      neural network, each identified by ``model_family`` so the comparison is
      between kinds of estimator rather than between names; and
    * **splitting strategy** — the same model under the optimistic random split
      and under the grouped splits, with an ``optimism_gap_<metric>`` column
      per metric giving the random split's score minus this row's.

    Every metric gets its own gap column, not just the selection metric,
    because the inflation is not uniform across metrics: a random split can
    leave RMSE almost unchanged while adding a great deal to R², since the
    latter is scored against the variance of a validation set that has itself
    been made easier.

    Nothing is refit. The table is a reshaping of the saved out-of-fold
    metrics, so it cannot disagree with the numbers the training stages logged.

    Returns:
        Rows ordered by target, then config model order, then strategy order,
        with ``NaN`` gaps on the random-split rows themselves.
    """
    metrics = configured_metrics(cfg)
    strategies = [s for s in cfg["validation"]["strategies"] if s in set(summary[const.CV_COL_STRATEGY])]
    configured_models = [
        *cfg["models"]["candidates"],
        const.TORCH_MODEL_NAME,
        const.TORCH_V2_MODEL_NAME,
    ]
    models = [m for m in configured_models if m in set(summary[const.COL_MODEL])]
    random_ = const.SplitStrategy.RANDOM.value

    indexed = summary.set_index([const.COL_TARGET, const.COL_MODEL, const.CV_COL_STRATEGY])
    rows = []
    for target in const.TARGETS.values():
        for model in models:
            for strategy in strategies:
                key = (target, model, strategy)
                if key not in indexed.index:
                    continue
                row = indexed.loc[key]
                record: dict[str, Any] = {
                    const.COL_TARGET: target,
                    const.COL_MODEL: model,
                    const.EVAL_COL_MODEL_FAMILY: const.MODEL_FAMILIES.get(
                        model, const.MODEL_FAMILY_UNKNOWN
                    ),
                    const.CV_COL_STRATEGY: strategy,
                    const.COL_N_SAMPLES: int(row[const.COL_N_SAMPLES]),
                }
                optimistic = (target, model, random_)
                for metric in metrics:
                    pooled = f"{const.POOLED_PREFIX}{metric.value}"
                    fold_std = f"{const.FOLD_STD_PREFIX}{metric.value}"
                    record[pooled] = float(row[pooled])
                    record[fold_std] = float(row[fold_std]) if fold_std in row else float("nan")
                    record[f"{const.EVAL_OPTIMISM_GAP_PREFIX}{metric.value}"] = (
                        float(indexed.loc[optimistic, pooled]) - float(row[pooled])
                        if optimistic in indexed.index and strategy != random_
                        else float("nan")
                    )
                rows.append(record)
    return pd.DataFrame(rows)


def family_leaderboard(comparison: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Rank the model families on the reported strategy, best first per target.

    Ranking happens *within* a target because the two targets are on different
    scales and are not jointly orderable, and only on the reported strategy
    because ranking models on the optimistic split is how one ends up shipping
    the model that memorises best.
    """
    strategy = evaluation_strategy(cfg)
    selection = const.Metric(cfg["models"]["selection_metric"])
    column = f"{const.POOLED_PREFIX}{selection.value}"
    ascending = not const.METRIC_GREATER_IS_BETTER[selection]

    rows = comparison[comparison[const.CV_COL_STRATEGY] == strategy]
    ranked = []
    for target in const.TARGETS.values():
        target_rows = rows[rows[const.COL_TARGET] == target].sort_values(column, ascending=ascending)
        target_rows = target_rows.assign(rank=range(1, len(target_rows) + 1))
        ranked.append(target_rows)
    if not ranked:
        return pd.DataFrame()
    return pd.concat(ranked, ignore_index=True)


def interpretation_notes(comparison: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Generate the scientific commentary from the numbers actually measured.

    The sentences are assembled from this run's values rather than written as
    fixed prose, so the discussion cannot drift out of step with the tables
    beside it when the pipeline is re-run on different data.
    """
    notes: list[str] = []
    strategy = evaluation_strategy(cfg)
    selection = cfg["models"]["selection_metric"]
    gap_column = f"{const.EVAL_OPTIMISM_GAP_PREFIX}{selection}"
    reported = comparison[comparison[const.CV_COL_STRATEGY] == strategy]

    if gap_column in reported and reported[gap_column].notna().any():
        worst = reported.loc[reported[gap_column].idxmax()]
        notes.append(
            f"**Why the family-aware split scores lower.** Across every model the random split "
            f"reports a higher `{selection}` than `{strategy}`; the largest inflation is "
            f"{float(worst[gap_column]):+.3f} for `{worst[const.COL_MODEL]}` on "
            f"`{worst[const.COL_TARGET]}`. A random split scatters mutations of the same protein "
            "across the train and validation sides, so a model can recognise the protein it is "
            "scoring and recall the average effect of mutating it. The family-grouped split "
            "withholds entire evolutionary families, so the model must extrapolate to a fold "
            "and a structural context it has never seen, and the family-level baseline effect "
            "it would otherwise lean on is unavailable. The grouped number is the honest "
            "estimate for the assignment's candidates, which are unlabelled proteins the model "
            "has not been trained on."
        )

    pearson = f"{const.POOLED_PREFIX}{const.Metric.PEARSON.value}"
    spearman = f"{const.POOLED_PREFIX}{const.Metric.SPEARMAN.value}"
    if pearson in reported and spearman in reported and not reported.empty:
        divergence = (reported[spearman] - reported[pearson]).abs()
        row = reported.loc[divergence.idxmax()]
        notes.append(
            f"**Pearson vs Spearman.** Pearson measures how close the relationship is to a "
            f"straight line and is therefore sensitive to a few badly-placed extreme values; "
            f"Spearman measures only whether the ordering is right. The widest divergence here "
            f"is `{row[const.COL_MODEL]}` on `{row[const.COL_TARGET]}`, at "
            f"{float(row[pearson]):.3f} Pearson against {float(row[spearman]):.3f} Spearman. "
            "For this project Spearman is the one that matters: the deliverable is a ranked "
            "shortlist of candidates to take into the wet lab, so being right about which "
            "mutation is more promising counts for more than being numerically calibrated."
        )

    r2 = f"{const.POOLED_PREFIX}{const.Metric.R2.value}"
    if r2 in reported and (reported[r2] < 0).any():
        losers = reported[reported[r2] < 0]
        names = ", ".join(
            f"`{m}` on `{t}`"
            for m, t in zip(losers[const.COL_MODEL], losers[const.COL_TARGET], strict=True)
        )
        verb = "scores" if len(losers) == 1 else "score"
        notes.append(
            f"**Negative R².** {names} {verb} below zero, meaning the predictions are worse than "
            "simply predicting the training mean. A negative R² alongside a positive rank "
            "correlation says the model has some signal about ordering but is badly "
            "mis-calibrated in magnitude; a negative R² with near-zero rank correlation says "
            "it has not learned the target at all."
        )
    return notes


def comparison_to_markdown(
    comparison: pd.DataFrame, leaderboard: pd.DataFrame, notes: list[str], cfg: dict[str, Any]
) -> str:
    """Render the Task 5 comparative analysis as a Markdown section."""
    strategy = evaluation_strategy(cfg)
    selection = cfg["models"]["selection_metric"]
    families = ", ".join(dict.fromkeys(comparison[const.EVAL_COL_MODEL_FAMILY]))
    lines = [
        "# Comparative evaluation",
        "",
        f"- Model families compared: {families}",
        f"- Splitting strategies compared: "
        f"{', '.join(dict.fromkeys(comparison[const.CV_COL_STRATEGY]))}",
        f"- Reported strategy: `{strategy}`; ranking metric: "
        f"`{const.POOLED_PREFIX}{selection}`",
        "",
        f"`{const.EVAL_OPTIMISM_GAP_PREFIX}<metric>` is the random split's score for that model "
        "and target minus this row's, so a positive gap is the credit the optimistic split "
        "awards for recognising proteins and families already seen in training.",
        "",
    ]
    if not leaderboard.empty:
        keep = [
            const.COL_MODEL,
            const.EVAL_COL_MODEL_FAMILY,
            "rank",
            f"{const.POOLED_PREFIX}{selection}",
        ]
        for target in const.TARGETS.values():
            rows = leaderboard[leaderboard[const.COL_TARGET] == target]
            if rows.empty:
                continue
            lines += [
                f"## {target}: family ranking under `{strategy}`",
                "",
                rows[keep].to_markdown(index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT),
                "",
            ]
    for target in const.TARGETS.values():
        rows = comparison[comparison[const.COL_TARGET] == target]
        if rows.empty:
            continue
        lines += [
            f"## {target}: all models, all strategies",
            "",
            rows.drop(columns=const.COL_TARGET).to_markdown(
                index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT
            ),
            "",
        ]
    if notes:
        lines += ["## Interpretation", ""]
        for note in notes:
            lines += [note, ""]
    return "\n".join(lines) + "\n"


def evaluation_record(
    headline: pd.DataFrame,
    comparison: pd.DataFrame,
    leaderboard: pd.DataFrame,
    gaps: pd.DataFrame,
    notes: list[str],
    figures: dict[str, Path],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the machine-readable evaluation summary.

    The JSON mirrors the Markdown rather than adding to it, so a downstream
    consumer and a human reader cannot be looking at different conclusions.
    """
    return {
        "random_seed": cfg["project"]["random_seed"],
        "reported_strategy": evaluation_strategy(cfg),
        "selection_metric": f"{const.POOLED_PREFIX}{cfg['models']['selection_metric']}",
        "metrics": [metric.value for metric in configured_metrics(cfg)],
        "targets": list(const.TARGETS.values()),
        "model_families": {
            model: const.MODEL_FAMILIES.get(model, const.MODEL_FAMILY_UNKNOWN)
            for model in dict.fromkeys(comparison[const.COL_MODEL])
        },
        "selected_models": headline.to_dict(orient="records"),
        "family_ranking": leaderboard.to_dict(orient="records"),
        "model_comparison": comparison.to_dict(orient="records"),
        "optimism_gap": gaps.to_dict(orient="records"),
        "interpretation": notes,
        "figures": {key: path.name for key, path in figures.items()},
    }


# =============================================================================
# Figures
# =============================================================================
def _figure_paths(cfg: dict[str, Any], keys: tuple[str, ...] = const.FIGURE_FILE_KEYS) -> dict[str, Path]:
    """Resolve the configured figure filenames under ``paths.figures_dir``."""
    return {key: resolve_path(cfg, "figures_dir", key) for key in keys}


def _pyplot() -> Any:
    """Import pyplot against the non-interactive backend.

    Deferred to call time, and to each plotting function, so that importing
    this module costs nothing on a headless machine or when figures are off.
    """
    import matplotlib

    matplotlib.use(const.MATPLOTLIB_BACKEND)
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, path: Path, dpi: int) -> Path:
    """Write ``fig`` to ``path``, creating the directory, and close it."""
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    _pyplot().close(fig)
    return path


def plot_predicted_vs_actual(oof: pd.DataFrame, path: Path, dpi: int) -> Path:
    """Scatter OOF prediction against ground truth, one panel per target."""
    plt = _pyplot()

    targets = list(const.TARGETS.values())
    fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 5.0))
    for ax, target in zip(np.atleast_1d(axes), targets, strict=True):
        group = oof[oof[const.COL_TARGET] == target]
        y_true = group[const.COL_Y_TRUE].to_numpy(dtype=float)
        y_pred = group[const.COL_Y_PRED].to_numpy(dtype=float)
        ax.scatter(y_true, y_pred, s=14, alpha=0.55, edgecolor="none")
        if y_true.size:
            limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax.plot(limits, limits, linestyle="--", linewidth=1.0, color="grey")
        ax.set_xlabel(f"observed {target}")
        ax.set_ylabel(f"out-of-fold predicted {target}")
        ax.set_title(target)
    fig.suptitle("Out-of-fold predictions vs ground truth")
    return _save(fig, path, dpi)


def plot_residual_distribution(oof: pd.DataFrame, path: Path, dpi: int) -> Path:
    """Histogram the OOF residuals per target to expose bias and heavy tails."""
    plt = _pyplot()

    targets = list(const.TARGETS.values())
    fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 4.0))
    for ax, target in zip(np.atleast_1d(axes), targets, strict=True):
        group = oof[oof[const.COL_TARGET] == target]
        residual = group[const.COL_Y_PRED].to_numpy(dtype=float) - group[const.COL_Y_TRUE].to_numpy(
            dtype=float
        )
        ax.hist(residual, bins=const.RESIDUAL_HIST_BINS)
        ax.axvline(0.0, linestyle="--", linewidth=1.0, color="grey")
        ax.set_xlabel(f"predicted - observed ({target})")
        ax.set_ylabel("mutations")
        ax.set_title(f"{target}: mean {residual.mean():+.3f}" if residual.size else target)
    fig.suptitle("Out-of-fold residual distribution")
    return _save(fig, path, dpi)


def plot_strategy_comparison(summary: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int) -> Path:
    """Bar-chart the pooled selection metric per model under every CV strategy.

    This is the visual form of the optimism gap: bars that shrink from left to
    right as the split gets stricter are the signature of leakage-driven
    optimism in the random split.
    """
    plt = _pyplot()

    metric = f"{const.POOLED_PREFIX}{cfg['models']['selection_metric']}"
    strategies = [s for s in cfg["validation"]["strategies"] if s in set(summary[const.CV_COL_STRATEGY])]
    targets = list(const.TARGETS.values())
    fig, axes = plt.subplots(1, len(targets), figsize=(6.0 * len(targets), 4.5), sharey=True)
    for ax, target in zip(np.atleast_1d(axes), targets, strict=True):
        rows = summary[summary[const.COL_TARGET] == target]
        models = list(dict.fromkeys(rows[const.COL_MODEL]))
        width = 0.8 / max(len(strategies), 1)
        positions = np.arange(len(models), dtype=float)
        for offset, strategy in enumerate(strategies):
            values = [
                rows[(rows[const.COL_MODEL] == model) & (rows[const.CV_COL_STRATEGY] == strategy)][
                    metric
                ].to_numpy(dtype=float)
                for model in models
            ]
            heights = [float(v[0]) if v.size else float("nan") for v in values]
            ax.bar(positions + offset * width, heights, width=width, label=strategy)
        ax.set_xticks(positions + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(models, rotation=const.FIGURE_XTICK_ROTATION, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(target)
        ax.axhline(0.0, linewidth=0.8, color="black")
    np.atleast_1d(axes)[0].legend(title="CV strategy")
    fig.suptitle(f"{metric} by model and cross-validation strategy")
    return _save(fig, path, dpi)


def make_figures(oof: pd.DataFrame, summary: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Path]:
    """Write every evaluation figure; returns the written paths (empty if disabled)."""
    if not cfg["evaluation"]["figures"]:
        logger.info("evaluation.figures is false: skipping figure generation")
        return {}
    dpi = cfg["evaluation"]["figure_dpi"]
    paths = _figure_paths(cfg)
    written = {
        "fig_pred_vs_actual": plot_predicted_vs_actual(oof, paths["fig_pred_vs_actual"], dpi),
        "fig_residuals": plot_residual_distribution(oof, paths["fig_residuals"], dpi),
        "fig_strategy_comparison": plot_strategy_comparison(
            summary, cfg, paths["fig_strategy_comparison"], dpi
        ),
    }
    for key, path in written.items():
        logger.info("Wrote %s -> %s", key, path)
    return written


# =============================================================================
# Figures for the neural network and the model comparison (assignment Task 4)
# =============================================================================
def plot_learning_curves(
    history: pd.DataFrame, fold_summary: pd.DataFrame, path: Path, dpi: int
) -> Path:
    """Plot training and inner-validation loss per epoch, one panel per fold.

    This is the figure that shows early stopping actually working: the dashed
    line marks the epoch whose weights were kept, and it should sit at the
    minimum of the inner-validation curve, well before the run ends. A fold
    whose best epoch is its last one never converged and was cut off by
    ``torch.max_epochs`` instead of by the stopping rule.

    The validation curve here is the **inner** split carved from the training
    fold, not the out-of-fold rows the metrics are computed on, so it says
    nothing about reported performance — only about when to stop.
    """
    plt = _pyplot()

    folds = sorted(history[const.CV_COL_FOLD].unique())
    columns = min(const.LEARNING_CURVE_COLUMNS, max(len(folds), 1))
    rows = -(-len(folds) // columns)  # ceiling division
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.6 * columns, 3.4 * rows), squeeze=False, sharex=False
    )
    flat = axes.ravel()
    best_epochs = (
        fold_summary.set_index(const.CV_COL_FOLD)[const.TORCH_COL_BEST_EPOCH]
        if const.TORCH_COL_BEST_EPOCH in fold_summary
        else pd.Series(dtype=float)
    )
    for ax, fold in zip(flat, folds, strict=False):
        curve = history[history[const.CV_COL_FOLD] == fold].sort_values(const.TORCH_COL_EPOCH)
        epochs = curve[const.TORCH_COL_EPOCH].to_numpy(dtype=float)
        ax.plot(epochs, curve[const.TORCH_COL_TRAIN_LOSS].to_numpy(dtype=float), label="train")
        ax.plot(
            epochs,
            curve[const.TORCH_COL_VAL_LOSS].to_numpy(dtype=float),
            label="inner validation",
        )
        if fold in best_epochs.index:
            best = float(best_epochs.loc[fold])
            ax.axvline(best, linestyle="--", linewidth=1.0, color="grey")
            ax.set_title(f"fold {fold} (kept epoch {best:.0f})")
        else:
            ax.set_title(f"fold {fold}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("weighted multitask loss")
    for ax in flat[len(folds) :]:
        ax.set_visible(False)
    flat[0].legend()
    fig.suptitle("Neural network learning curves (loss per fold)")
    return _save(fig, path, dpi)


def plot_model_comparison(comparison: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int) -> Path:
    """Bar-chart every configured metric for every model, baselines and network alike.

    One panel per target × metric, because the metrics are on incompatible
    scales — RMSE and MAE are in target units, R² and the two correlations are
    dimensionless — and putting them on shared axes would misrepresent them.
    Each panel's title states the direction, since lower is better for the
    error metrics and higher for the rest.
    """
    plt = _pyplot()

    metrics = configured_metrics(cfg)
    targets = [t for t in const.TARGETS.values() if t in set(comparison[const.COL_TARGET])]
    models = list(dict.fromkeys(comparison[const.COL_MODEL]))
    fig, axes = plt.subplots(
        len(targets),
        len(metrics),
        figsize=(3.1 * len(metrics), 3.4 * len(targets)),
        squeeze=False,
    )
    for row, target in enumerate(targets):
        rows = comparison[comparison[const.COL_TARGET] == target].set_index(const.COL_MODEL)
        for column, metric in enumerate(metrics):
            ax = axes[row][column]
            name = f"{const.POOLED_PREFIX}{metric.value}"
            heights = [
                float(rows.loc[model, name]) if model in rows.index else float("nan")
                for model in models
            ]
            # The network is the subject of the comparison, so it is the one bar
            # that is filled differently; everything else is a reference point.
            colors = ["tab:orange" if m == const.TORCH_MODEL_NAME else "tab:blue" for m in models]
            ax.bar(np.arange(len(models)), heights, color=colors)
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels(models, rotation=const.FIGURE_XTICK_ROTATION, ha="right")
            direction = "higher is better" if const.METRIC_GREATER_IS_BETTER[metric] else "lower is better"
            ax.set_title(f"{metric.value} ({direction})", fontsize=9)
            ax.axhline(0.0, linewidth=0.8, color="black")
            if column == 0:
                ax.set_ylabel(target)
    strategy = comparison[const.CV_COL_STRATEGY].iloc[0] if not comparison.empty else "n/a"
    fig.suptitle(f"Pooled out-of-fold metrics by model ({strategy} cross-validation)")
    return _save(fig, path, dpi)


def plot_baseline_vs_torch(
    baseline_oof: pd.DataFrame,
    torch_oof: pd.DataFrame,
    path: Path,
    dpi: int,
    model_label: str = const.TORCH_MODEL_NAME,
) -> Path:
    """Overlay the selected baseline and the network on the same out-of-fold rows.

    Top row is predicted against observed with the identity line; bottom row is
    the residual distribution. Two models trained on identical folds predict the
    same rows, so these panels are directly comparable: a model that tracks the
    identity line but spreads wider is worse at the same job, whereas one whose
    residual histogram is shifted off zero is biased rather than merely noisy.
    """
    plt = _pyplot()

    targets = list(const.TARGETS.values())
    fig, axes = plt.subplots(2, len(targets), figsize=(5.4 * len(targets), 8.0), squeeze=False)
    series = [("best baseline", baseline_oof), (model_label, torch_oof)]
    for column, target in enumerate(targets):
        scatter_ax, hist_ax = axes[0][column], axes[1][column]
        bounds: list[float] = []
        for label, frame in series:
            group = frame[frame[const.COL_TARGET] == target]
            if group.empty:
                continue
            y_true = group[const.COL_Y_TRUE].to_numpy(dtype=float)
            y_pred = group[const.COL_Y_PRED].to_numpy(dtype=float)
            name = (
                f"{label} ({group[const.COL_MODEL].iloc[0]})"
                if label == "best baseline"
                else label
            )
            scatter_ax.scatter(
                y_true, y_pred, s=14, alpha=const.FIGURE_SCATTER_ALPHA, edgecolor="none", label=name
            )
            hist_ax.hist(
                y_pred - y_true,
                bins=const.RESIDUAL_HIST_BINS,
                alpha=const.FIGURE_OVERLAY_ALPHA,
                label=f"{name}: mean {float((y_pred - y_true).mean()):+.3f}",
            )
            bounds += [float(y_true.min()), float(y_true.max()), float(y_pred.min()), float(y_pred.max())]
        if bounds:
            limits = [min(bounds), max(bounds)]
            scatter_ax.plot(limits, limits, linestyle="--", linewidth=1.0, color="grey")
        scatter_ax.set_xlabel(f"observed {target}")
        scatter_ax.set_ylabel(f"out-of-fold predicted {target}")
        scatter_ax.set_title(target)
        scatter_ax.legend(fontsize=8)
        hist_ax.axvline(0.0, linestyle="--", linewidth=1.0, color="grey")
        hist_ax.set_xlabel(f"predicted - observed ({target})")
        hist_ax.set_ylabel("mutations")
        hist_ax.legend(fontsize=8)
    fig.suptitle("Selected baseline vs. multitask network on identical out-of-fold rows")
    return _save(fig, path, dpi)


def plot_ablation_performance_drop(
    table: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int
) -> Path:
    """Chart each feature subset's metric change against the baseline subset.

    One row per headline metric, one column per target. Bars are the signed
    difference from the baseline subset, so the zero line *is* the full feature
    set and a bar's length is exactly what removing (or keeping only) that
    group costs. Colour encodes the direction after accounting for each
    metric's polarity, which matters because the same sign means opposite
    things for an error metric and a correlation.

    Horizontal bars, because subset names are long and reading seven rotated
    labels is worse than reading seven horizontal ones.
    """
    plt = _pyplot()

    baseline_name = cfg["ablation"]["baseline"]
    metrics = [m for m in const.ABL_HEADLINE_METRICS if m in set(configured_metrics(cfg))]
    targets = [t for t in const.TARGETS.values() if t in set(table[const.COL_TARGET])]
    # Config order, baseline last, so the arms read in the order they are declared.
    subsets = [s for s in dict.fromkeys(table[const.ABL_COL_SUBSET]) if s != baseline_name]

    fig, axes = plt.subplots(
        len(metrics),
        len(targets),
        figsize=(6.0 * len(targets), 0.55 * len(subsets) * len(metrics) + 2.2),
        squeeze=False,
    )
    positions = np.arange(len(subsets), dtype=float)
    for row, metric in enumerate(metrics):
        column_name = f"{const.ABL_DELTA_PREFIX}{metric.value}"
        higher_is_better = const.METRIC_GREATER_IS_BETTER[metric]
        for column, target in enumerate(targets):
            ax = axes[row][column]
            rows = table[table[const.COL_TARGET] == target].set_index(const.ABL_COL_SUBSET)
            values = [
                float(rows.loc[subset, column_name]) if subset in rows.index else float("nan")
                for subset in subsets
            ]
            # A drop is a loss whatever the metric's polarity: for r2/spearman
            # that is a negative delta, for rmse/mae a positive one.
            colors = [
                const.ABL_COLOR_LOSS
                if (value < 0) == higher_is_better
                else const.ABL_COLOR_GAIN
                for value in values
            ]
            ax.barh(positions, values, color=colors)
            ax.set_yticks(positions)
            ax.set_yticklabels(subsets, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0.0, linewidth=1.0, color="black")
            ax.set_xlabel(f"Δ {metric.value} vs {baseline_name}")
            ax.set_title(f"{target}", fontsize=10)
            span = max((abs(v) for v in values if np.isfinite(v)), default=1.0) or 1.0
            ax.set_xlim(-span * 1.35, span * 1.35)
            for y, value in zip(positions, values, strict=True):
                if not np.isfinite(value):
                    continue
                offset = const.ABL_BAR_LABEL_PAD * span / abs(span)
                ax.text(
                    value + (offset if value >= 0 else -offset),
                    y,
                    f"{value:+.3f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8,
                )
    fig.suptitle(
        f"Feature-group attribution: change from the full set (`{baseline_name}`) "
        f"under {table[const.CV_COL_STRATEGY].iloc[0]} cross-validation"
    )
    return _save(fig, path, dpi)


def plot_feature_importance(
    per_feature: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int
) -> Path:
    """Chart the top features per target, coloured by feature group.

    Only the permutation kind is drawn when it is available: impurity
    importances live on post-PCA inputs and on a different scale, so putting
    the two on one axis would invite a comparison that is not meaningful.

    Error bars are the between-fold standard deviation. A bar whose whisker
    crosses zero is not a stable attribution on this dataset, and showing that
    is the point of drawing them.
    """
    plt = _pyplot()

    top_k = int(cfg["attribution"]["top_k"])
    kinds = list(dict.fromkeys(per_feature[const.ATTR_COL_KIND]))
    kind = const.ImportanceKind.PERMUTATION.value
    kind = kind if kind in kinds else kinds[0]
    rows_for_kind = per_feature[per_feature[const.ATTR_COL_KIND] == kind]
    targets = [t for t in const.TARGETS.values() if t in set(rows_for_kind[const.COL_TARGET])]

    fig, axes = plt.subplots(
        1, len(targets), figsize=(7.0 * len(targets), 0.32 * top_k + 2.0), squeeze=False
    )
    for column, target in enumerate(targets):
        ax = axes[0][column]
        rows = rows_for_kind[rows_for_kind[const.COL_TARGET] == target].head(top_k)
        positions = np.arange(len(rows), dtype=float)
        colors = [
            const.ATTR_GROUP_COLORS.get(group, const.ABL_COLOR_LOSS)
            for group in rows[const.ATTR_COL_GROUP]
        ]
        ax.barh(
            positions,
            rows[const.ATTR_COL_MEAN].to_numpy(dtype=float),
            xerr=rows[const.ATTR_COL_STD].to_numpy(dtype=float),
            color=colors,
            error_kw={"elinewidth": 0.8, "ecolor": "#444444"},
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(rows[const.ATTR_COL_FEATURE], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.0, linewidth=1.0, color="black")
        ax.set_xlabel(f"{cfg['attribution']['scoring_metric']} lost when shuffled")
        ax.set_title(target, fontsize=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for group, color in const.ATTR_GROUP_COLORS.items()
        if group in set(rows_for_kind[const.ATTR_COL_GROUP])
    ]
    labels = [
        group
        for group in const.ATTR_GROUP_COLORS
        if group in set(rows_for_kind[const.ATTR_COL_GROUP])
    ]
    if handles:
        axes[0][0].legend(handles, labels, title="feature group", fontsize=8, loc="lower right")
    fig.suptitle(
        f"Top {top_k} features by {kind} importance, averaged over "
        f"{cfg['validation']['primary_strategy']} folds"
    )
    return _save(fig, path, dpi)


def plot_group_importance(
    per_group: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int
) -> Path:
    """Chart importance summed within each feature group, per target.

    This is the figure that answers the ablation question directly: how much
    of the model's realised skill comes from each *source* of information,
    rather than from any one column.
    """
    plt = _pyplot()

    kind = const.ImportanceKind.PERMUTATION.value
    kinds = list(dict.fromkeys(per_group[const.ATTR_COL_KIND]))
    kind = kind if kind in kinds else kinds[0]
    rows_for_kind = per_group[per_group[const.ATTR_COL_KIND] == kind]
    targets = [t for t in const.TARGETS.values() if t in set(rows_for_kind[const.COL_TARGET])]
    groups = [g for g in const.ATTR_GROUP_COLORS if g in set(rows_for_kind[const.ATTR_COL_GROUP])]

    fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 4.2), squeeze=False)
    for column, target in enumerate(targets):
        ax = axes[0][column]
        rows = rows_for_kind[rows_for_kind[const.COL_TARGET] == target].set_index(
            const.ATTR_COL_GROUP
        )
        heights = [
            float(rows.loc[group, const.ATTR_COL_MEAN]) if group in rows.index else 0.0
            for group in groups
        ]
        positions = np.arange(len(groups), dtype=float)
        ax.bar(positions, heights, color=[const.ATTR_GROUP_COLORS[g] for g in groups])
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=const.FIGURE_XTICK_ROTATION, ha="right", fontsize=8)
        ax.axhline(0.0, linewidth=1.0, color="black")
        ax.set_ylabel(f"summed {cfg['attribution']['scoring_metric']} lost when shuffled")
        ax.set_title(target, fontsize=10)
        for x, height in zip(positions, heights, strict=True):
            ax.text(x, height, f"{height:+.3f}", ha="center",
                    va="bottom" if height >= 0 else "top", fontsize=8)
    fig.suptitle(f"Where the model's skill comes from, by feature group ({kind} importance)")
    return _save(fig, path, dpi)


def make_attribution_figures(
    per_feature: pd.DataFrame, per_group: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, Path]:
    """Write the Task 7 figures; returns the written paths (empty if disabled)."""
    if not cfg["evaluation"]["figures"]:
        logger.info("evaluation.figures is false: skipping the attribution figures")
        return {}
    dpi = cfg["evaluation"]["figure_dpi"]
    paths = _figure_paths(cfg, const.ATTR_FIGURE_FILE_KEYS)
    written: dict[str, Path] = {}
    if not per_feature.empty:
        written["fig_feature_importance"] = plot_feature_importance(
            per_feature, cfg, paths["fig_feature_importance"], dpi
        )
    if not per_group.empty:
        written["fig_group_importance"] = plot_group_importance(
            per_group, cfg, paths["fig_group_importance"], dpi
        )
    for key, figure in written.items():
        logger.info("Wrote %s -> %s", key, figure)
    return written


def make_ablation_figures(table: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Path]:
    """Write the ablation figure; returns the written paths (empty if disabled)."""
    if not cfg["evaluation"]["figures"]:
        logger.info("evaluation.figures is false: skipping the ablation figure")
        return {}
    if table.empty:
        logger.warning("Ablation table is empty: no figure to draw")
        return {}
    paths = _figure_paths(cfg, const.ABLATION_FIGURE_FILE_KEYS)
    written = {
        "fig_ablation_drop": plot_ablation_performance_drop(
            table, cfg, paths["fig_ablation_drop"], cfg["evaluation"]["figure_dpi"]
        )
    }
    for key, figure in written.items():
        logger.info("Wrote %s -> %s", key, figure)
    return written


def make_torch_figures(
    torch_oof: pd.DataFrame,
    history: pd.DataFrame,
    fold_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    cfg: dict[str, Any],
    figure_keys: tuple[str, ...] = const.TORCH_FIGURE_FILE_KEYS,
    model_label: str = const.TORCH_MODEL_NAME,
) -> dict[str, Path]:
    """Write the Task 4 figures; returns the written paths (empty if disabled).

    The baseline overlay is skipped rather than fabricated when the training
    stage has not been run, so a network run on its own still produces the two
    figures that depend only on its own artifacts.

    Args:
        torch_oof: The network's out-of-fold predictions.
        history: Per-epoch losses.
        fold_summary: Per-fold early-stopping record.
        comparison: The network beside the other models.
        cfg: Loaded configuration.
        figure_keys: The three config ``files`` keys to write, in the order
            learning curves, model comparison, baseline overlay. Each network
            architecture passes its own so they do not overwrite each other.
        model_label: Name for the network in the overlay legend.

    Raises:
        ValueError: If ``figure_keys`` does not hold exactly three keys.
    """
    if len(figure_keys) != len(const.TORCH_FIGURE_FILE_KEYS):
        raise ValueError(
            f"figure_keys must hold {len(const.TORCH_FIGURE_FILE_KEYS)} keys "
            f"(learning curves, comparison, overlay), got {figure_keys}"
        )
    if not cfg["evaluation"]["figures"]:
        logger.info("evaluation.figures is false: skipping neural-network figures")
        return {}
    dpi = cfg["evaluation"]["figure_dpi"]
    curves_key, comparison_key, overlay_key = figure_keys
    paths = _figure_paths(cfg, figure_keys)
    written: dict[str, Path] = {}
    if not history.empty:
        written[curves_key] = plot_learning_curves(
            history, fold_summary, paths[curves_key], dpi
        )
    if not comparison.empty:
        written[comparison_key] = plot_model_comparison(
            comparison, cfg, paths[comparison_key], dpi
        )
    try:
        oof, _, selection = load_evaluation_inputs(cfg)
    except FileNotFoundError:
        logger.warning("Baseline artifacts missing: skipping the baseline-vs-network figure")
    else:
        baseline = selected_oof(oof, selection, evaluation_strategy(cfg))
        if baseline.empty:
            logger.warning("No baseline OOF rows to compare against: skipping the overlay figure")
        else:
            written[overlay_key] = plot_baseline_vs_torch(
                baseline, torch_oof, paths[overlay_key], dpi, model_label=model_label
            )
    for key, figure in written.items():
        logger.info("Wrote %s -> %s", key, figure)
    return written


# =============================================================================
# Report
# =============================================================================
def to_markdown(
    headline: pd.DataFrame,
    summary: pd.DataFrame,
    gaps: pd.DataFrame,
    figures: dict[str, Path],
    cfg: dict[str, Any],
) -> str:
    """Render the evaluation summary as a Markdown section for the technical report."""
    strategy = evaluation_strategy(cfg)
    lines = [
        "# Evaluation summary",
        "",
        f"- Reported cross-validation strategy: `{strategy}`",
        f"- Selection metric: `{const.POOLED_PREFIX}{cfg['models']['selection_metric']}` "
        "(pooled over all out-of-fold predictions)",
        "",
        "## Selected models",
        "",
        headline.to_markdown(index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT),
        "",
        "## All models, all strategies",
        "",
        summary.to_markdown(index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT),
        "",
    ]
    if not gaps.empty:
        lines += [
            "## Optimism gap",
            "",
            "A positive gap is the amount by which a random split over-states performance "
            "relative to the grouped split, i.e. the credit the model gets for recognising "
            "proteins or families it has already seen.",
            "",
            gaps.to_markdown(index=False, floatfmt=const.MARKDOWN_FLOAT_FORMAT),
            "",
        ]
    if figures:
        lines += ["## Figures", ""]
        lines += [f"- `{path.name}`" for path in figures.values()]
        lines += [""]
    return "\n".join(lines) + "\n"


def export_artifacts(
    markdown: str,
    benchmark: pd.DataFrame,
    comparison: pd.DataFrame,
    record: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Path]:
    """Write the evaluation summary, the baseline benchmark and the Task 5 comparison.

    Returns:
        Mapping of artifact name to written path.
    """
    paths = {
        key: resolve_path(cfg, "metrics_dir", key)
        for key in (
            "evaluation_markdown",
            "evaluation_json",
            "baseline_benchmark",
            "baseline_benchmark_markdown",
            "model_comparison",
        )
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["evaluation_markdown"].write_text(markdown, encoding=const.CONFIG_ENCODING)
    paths["evaluation_json"].write_text(
        json.dumps(record, indent=2, default=str), encoding=const.CONFIG_ENCODING
    )
    write_table(benchmark, paths["baseline_benchmark"])
    paths["baseline_benchmark_markdown"].write_text(
        benchmark_to_markdown(benchmark, cfg), encoding=const.CONFIG_ENCODING
    )
    write_table(comparison, paths["model_comparison"])
    for key, path in paths.items():
        logger.info("Wrote %s -> %s", key, path)
    return paths


def run_evaluation(cfg: dict[str, Any]) -> pd.DataFrame:
    """Re-report metrics and draw figures from the artifacts Phase 3 saved.

    Returns:
        The headline per-target table for the selected models.
    """
    oof, summary, selection = load_evaluation_inputs(cfg)
    strategy = evaluation_strategy(cfg)
    focus = selected_oof(oof, selection, strategy)
    if focus.empty:
        raise ValueError(
            f"No out-of-fold predictions for the selected models under strategy '{strategy}'"
        )
    headline = headline_table(focus, cfg)
    gaps = log_optimism_gap(summary, cfg)
    benchmark = benchmark_table(summary, cfg)
    figures = make_figures(focus, summary, cfg)

    # Task 5: the same metrics reorganised across both comparative axes, with
    # the neural network folded in when it has been trained.
    combined = unified_summary(cfg)
    comparison = model_comparison_table(combined, cfg)
    leaderboard = family_leaderboard(comparison, cfg)
    notes = interpretation_notes(comparison, cfg)
    for note in notes:
        logger.info("%s", note)
    markdown = to_markdown(headline, summary, gaps, figures, cfg) + "\n" + comparison_to_markdown(
        comparison, leaderboard, notes, cfg
    )
    record = evaluation_record(headline, comparison, leaderboard, gaps, notes, figures, cfg)

    export_artifacts(markdown, benchmark, comparison, record, cfg)
    logger.info(
        "Evaluation complete: %d target rows, %d benchmark rows, %d comparison rows, %d figures",
        len(headline),
        len(benchmark),
        len(comparison),
        len(figures),
    )
    return headline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    run_evaluation(config)
