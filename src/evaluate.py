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
summary, re-derives the headline table, and writes the evaluation figures to
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
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table
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
# Figures
# =============================================================================
def _figure_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Resolve the configured figure filenames under ``paths.figures_dir``."""
    return {key: resolve_path(cfg, "figures_dir", key) for key in const.FIGURE_FILE_KEYS}


def plot_predicted_vs_actual(oof: pd.DataFrame, path: Path, dpi: int) -> Path:
    """Scatter OOF prediction against ground truth, one panel per target."""
    import matplotlib

    matplotlib.use(const.MATPLOTLIB_BACKEND)
    import matplotlib.pyplot as plt

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
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_residual_distribution(oof: pd.DataFrame, path: Path, dpi: int) -> Path:
    """Histogram the OOF residuals per target to expose bias and heavy tails."""
    import matplotlib

    matplotlib.use(const.MATPLOTLIB_BACKEND)
    import matplotlib.pyplot as plt

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
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_strategy_comparison(summary: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int) -> Path:
    """Bar-chart the pooled selection metric per model under every CV strategy.

    This is the visual form of the optimism gap: bars that shrink from left to
    right as the split gets stricter are the signature of leakage-driven
    optimism in the random split.
    """
    import matplotlib

    matplotlib.use(const.MATPLOTLIB_BACKEND)
    import matplotlib.pyplot as plt

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
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


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


def export_artifacts(markdown: str, cfg: dict[str, Any]) -> dict[str, Path]:
    """Write the evaluation Markdown summary.

    Returns:
        Mapping of artifact name to written path.
    """
    path = resolve_path(cfg, "metrics_dir", "evaluation_markdown")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding=const.CONFIG_ENCODING)
    logger.info("Wrote evaluation_markdown -> %s", path)
    return {"evaluation_markdown": path}


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
    figures = make_figures(focus, summary, cfg)
    export_artifacts(to_markdown(headline, summary, gaps, figures, cfg), cfg)
    logger.info("Evaluation complete: %d target rows, %d figures", len(headline), len(figures))
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
