"""Every figure the pipeline draws, as a family of :class:`Figure` classes.

Extracted from ``src/evaluate.py``, which had grown to hold three unrelated
jobs: computing metrics, drawing pictures of them, and writing reports. This
module owns only the middle one.

**Why classes here and not functions.** Each figure needs the same four things
— the non-interactive backend, its configured output path, the configured DPI,
and a save-and-close — and the old free functions repeated all four, eight
times over. :class:`Figure` holds that shared machinery once; a subclass
supplies only ``file_key`` and a ``draw`` method, and :meth:`Figure.render`
does the rest. Adding a figure is a new subclass plus a filename in
``config.yaml``, with nothing else to remember.

The classes also carry what a function signature could not: ``file_key`` ties
the figure to its config entry, and :class:`FigureSet` groups the figures one
pipeline stage emits so a stage asks for "its" figures rather than naming them
individually.

Figures are **data in, file out**. They never load anything and never compute
a metric, so a figure cannot disagree with the table beside it — both are
passed the same frame.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common import constants as const
from src.common.config import resolve_path

logger = logging.getLogger(__name__)


def pyplot() -> Any:
    """Import pyplot against the non-interactive backend.

    Deferred to call time so importing this module costs nothing on a headless
    machine or when figures are switched off.
    """
    import matplotlib

    matplotlib.use(const.MATPLOTLIB_BACKEND)
    import matplotlib.pyplot as plt

    return plt


# =============================================================================
# Base class
# =============================================================================
class Figure(ABC):
    """One figure: knows its config key, draws itself, saves itself.

    Subclasses implement :meth:`draw` and set :attr:`file_key`. Everything
    else — resolving the output path, creating the directory, applying the
    configured DPI, tightening the layout and closing the figure so a long run
    does not leak Matplotlib state — happens here, once.

    Attributes:
        file_key: Entry in the config ``files`` section holding the filename.
    """

    file_key: str

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.dpi = int(cfg["evaluation"]["figure_dpi"])

    @property
    def path(self) -> Path:
        """Where this figure is written, from ``paths.figures_dir``."""
        return resolve_path(self.cfg, "figures_dir", self.file_key)

    @abstractmethod
    def draw(self, plt: Any) -> Any:
        """Build and return the Matplotlib figure. Implemented by each subclass."""

    def is_drawable(self) -> bool:
        """Whether there is anything to draw.

        The default is yes. A subclass whose input can legitimately be empty
        overrides this so the stage skips it rather than writing a blank
        image, which would look like a result.
        """
        return True

    def render(self) -> Path | None:
        """Draw, save and close. Returns the path, or ``None`` if skipped."""
        if not self.is_drawable():
            logger.info("%s has nothing to draw; skipping", type(self).__name__)
            return None
        plt = pyplot()
        fig = self.draw(plt)
        fig.tight_layout()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.path, dpi=self.dpi)
        plt.close(fig)
        return self.path


class FigureSet:
    """The figures one pipeline stage emits, rendered together.

    A stage holds a set rather than a list of calls, so the
    ``evaluation.figures`` switch is honoured in one place and the stage's
    logging of what it wrote is identical everywhere.
    """

    def __init__(self, figures: list[Figure], cfg: dict[str, Any]) -> None:
        self.figures = figures
        self.cfg = cfg

    def render(self) -> dict[str, Path]:
        """Render every figure; returns ``{file_key: path}`` for those written."""
        if not self.cfg["evaluation"]["figures"]:
            logger.info("evaluation.figures is false: skipping %d figures", len(self.figures))
            return {}
        written: dict[str, Path] = {}
        for figure in self.figures:
            path = figure.render()
            if path is not None:
                written[figure.file_key] = path
                logger.info("Wrote %s -> %s", figure.file_key, path)
        return written


# =============================================================================
# Model-fit figures (Phase 3 / Task 5)
# =============================================================================
class PredictedVsActual(Figure):
    """Scatter out-of-fold prediction against ground truth, one panel per target."""

    file_key = "fig_pred_vs_actual"

    def __init__(self, oof: pd.DataFrame, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.oof = oof

    def draw(self, plt: Any) -> Any:
        """Build the scatter panels."""
        targets = list(const.TARGETS.values())
        fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 5.0))
        for ax, target in zip(np.atleast_1d(axes), targets, strict=True):
            group = self.oof[self.oof[const.COL_TARGET] == target]
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
        return fig


class ResidualDistribution(Figure):
    """Histogram the out-of-fold residuals per target, exposing bias and heavy tails."""

    file_key = "fig_residuals"

    def __init__(self, oof: pd.DataFrame, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.oof = oof

    def draw(self, plt: Any) -> Any:
        """Build the residual histograms."""
        targets = list(const.TARGETS.values())
        fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 4.0))
        for ax, target in zip(np.atleast_1d(axes), targets, strict=True):
            group = self.oof[self.oof[const.COL_TARGET] == target]
            residual = group[const.COL_Y_PRED].to_numpy(dtype=float) - group[
                const.COL_Y_TRUE
            ].to_numpy(dtype=float)
            ax.hist(residual, bins=const.RESIDUAL_HIST_BINS)
            ax.axvline(0.0, linestyle="--", linewidth=1.0, color="grey")
            ax.set_xlabel(f"predicted - observed ({target})")
            ax.set_ylabel("mutations")
            ax.set_title(f"{target}: mean {residual.mean():+.3f}" if residual.size else target)
        fig.suptitle("Out-of-fold residual distribution")
        return fig


class StrategyComparison(Figure):
    """Bar-chart the pooled selection metric per model under every CV strategy.

    This is the visual form of the optimism gap: bars that shrink from left to
    right as the split gets stricter are the signature of leakage-driven
    optimism in the random split.
    """

    file_key = "fig_strategy_comparison"

    def __init__(self, summary: pd.DataFrame, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.summary = summary

    def draw(self, plt: Any) -> Any:
        """Build the grouped bars, one group per model."""
        summary, cfg = self.summary, self.cfg
        metric = f"{const.POOLED_PREFIX}{cfg['models']['selection_metric']}"
        strategies = [
            s for s in cfg["validation"]["strategies"] if s in set(summary[const.CV_COL_STRATEGY])
        ]
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
        return fig


# =============================================================================
# Neural-network figures (Task 4)
# =============================================================================
class LearningCurves(Figure):
    """Training and inner-validation loss per epoch, one panel per fold.

    The figure that shows early stopping working: the dashed line marks the
    epoch whose weights were kept, and it should sit at the minimum of the
    inner-validation curve, well before the run ends. A fold whose best epoch
    is its last never converged and was cut off by ``max_epochs`` instead.

    The validation curve is the **inner** split carved from the training fold,
    not the out-of-fold rows the metrics are computed on, so it says nothing
    about reported performance — only about when to stop.
    """

    def __init__(
        self, history: pd.DataFrame, fold_summary: pd.DataFrame, cfg: dict[str, Any], file_key: str
    ) -> None:
        super().__init__(cfg)
        self.history = history
        self.fold_summary = fold_summary
        self.file_key = file_key

    def is_drawable(self) -> bool:
        """Nothing to plot before the network has been trained."""
        return not self.history.empty

    def draw(self, plt: Any) -> Any:
        """Build the per-fold loss curves."""
        history, fold_summary = self.history, self.fold_summary
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
        return fig


class ModelComparison(Figure):
    """Every configured metric for every model, baselines and network alike.

    One panel per target x metric, because the metrics are on incompatible
    scales — RMSE and MAE are in target units, R2 and the two correlations are
    dimensionless — and shared axes would misrepresent them. Each panel's
    title states the direction, since lower is better for the error metrics
    and higher for the rest.
    """

    def __init__(self, comparison: pd.DataFrame, cfg: dict[str, Any], file_key: str) -> None:
        super().__init__(cfg)
        self.comparison = comparison
        self.file_key = file_key

    def is_drawable(self) -> bool:
        """Needs at least the network's own rows to compare."""
        return not self.comparison.empty

    def draw(self, plt: Any) -> Any:
        """Build the metric x target panel grid."""
        from src.evaluate import configured_metrics

        comparison, cfg = self.comparison, self.cfg
        metrics = configured_metrics(cfg)
        targets = [t for t in const.TARGETS.values() if t in set(comparison[const.COL_TARGET])]
        models = list(dict.fromkeys(comparison[const.COL_MODEL]))
        fig, axes = plt.subplots(
            len(targets), len(metrics), figsize=(3.1 * len(metrics), 3.4 * len(targets)),
            squeeze=False,
        )
        networks = {const.TORCH_MODEL_NAME, const.TORCH_V2_MODEL_NAME}
        for row, target in enumerate(targets):
            rows = comparison[comparison[const.COL_TARGET] == target].set_index(const.COL_MODEL)
            for column, metric in enumerate(metrics):
                ax = axes[row][column]
                name = f"{const.POOLED_PREFIX}{metric.value}"
                heights = [
                    float(rows.loc[model, name]) if model in rows.index else float("nan")
                    for model in models
                ]
                # The networks are the subject of the comparison, so they are
                # filled differently; the baselines are the reference points.
                colors = ["tab:orange" if m in networks else "tab:blue" for m in models]
                ax.bar(np.arange(len(models)), heights, color=colors)
                ax.set_xticks(np.arange(len(models)))
                ax.set_xticklabels(models, rotation=const.FIGURE_XTICK_ROTATION, ha="right")
                direction = (
                    "higher is better"
                    if const.METRIC_GREATER_IS_BETTER[metric]
                    else "lower is better"
                )
                ax.set_title(f"{metric.value} ({direction})", fontsize=9)
                ax.axhline(0.0, linewidth=0.8, color="black")
                if column == 0:
                    ax.set_ylabel(target)
        strategy = comparison[const.CV_COL_STRATEGY].iloc[0] if not comparison.empty else "n/a"
        fig.suptitle(f"Pooled out-of-fold metrics by model ({strategy} cross-validation)")
        return fig


class BaselineVsNetwork(Figure):
    """Overlay the selected baseline and a network on the same out-of-fold rows.

    Top row is predicted against observed with the identity line; bottom row
    the residual distribution. Two models trained on identical folds predict
    the same rows, so the panels are directly comparable: a model that tracks
    the identity line but spreads wider is worse at the same job, whereas one
    whose residual histogram is shifted off zero is biased rather than noisy.
    """

    def __init__(
        self,
        baseline_oof: pd.DataFrame,
        network_oof: pd.DataFrame,
        cfg: dict[str, Any],
        file_key: str,
        model_label: str,
    ) -> None:
        super().__init__(cfg)
        self.baseline_oof = baseline_oof
        self.network_oof = network_oof
        self.file_key = file_key
        self.model_label = model_label

    def is_drawable(self) -> bool:
        """Both series are required; one alone is not a comparison."""
        return not self.baseline_oof.empty and not self.network_oof.empty

    def draw(self, plt: Any) -> Any:
        """Build the scatter and residual overlay."""
        targets = list(const.TARGETS.values())
        fig, axes = plt.subplots(2, len(targets), figsize=(5.4 * len(targets), 8.0), squeeze=False)
        series = [("best baseline", self.baseline_oof), (self.model_label, self.network_oof)]
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
                    y_true, y_pred, s=14, alpha=const.FIGURE_SCATTER_ALPHA,
                    edgecolor="none", label=name,
                )
                hist_ax.hist(
                    y_pred - y_true,
                    bins=const.RESIDUAL_HIST_BINS,
                    alpha=const.FIGURE_OVERLAY_ALPHA,
                    label=f"{name}: mean {float((y_pred - y_true).mean()):+.3f}",
                )
                bounds += [
                    float(y_true.min()), float(y_true.max()),
                    float(y_pred.min()), float(y_pred.max()),
                ]
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
        fig.suptitle("Selected baseline vs. network on identical out-of-fold rows")
        return fig


# =============================================================================
# Ablation figure (Task 6)
# =============================================================================
class AblationPerformanceDrop(Figure):
    """Each feature subset's metric change against the baseline subset.

    One row per headline metric, one column per target. Bars are the signed
    difference from the baseline subset, so the zero line *is* the full
    feature set and a bar's length is exactly what removing (or keeping only)
    that group costs. Colour encodes the direction after accounting for each
    metric's polarity, because the same sign means opposite things for an
    error metric and a correlation.

    Horizontal bars, because subset names are long and reading seven rotated
    labels is worse than reading seven horizontal ones.
    """

    file_key = "fig_ablation_drop"

    def __init__(self, table: pd.DataFrame, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.table = table

    def is_drawable(self) -> bool:
        """Nothing to attribute means no figure, rather than an empty one."""
        return not self.table.empty

    def draw(self, plt: Any) -> Any:
        """Build the signed delta bars."""
        from src.evaluate import configured_metrics

        table, cfg = self.table, self.cfg
        baseline_name = cfg["ablation"]["baseline"]
        metrics = [m for m in const.ABL_HEADLINE_METRICS if m in set(configured_metrics(cfg))]
        targets = [t for t in const.TARGETS.values() if t in set(table[const.COL_TARGET])]
        subsets = [s for s in dict.fromkeys(table[const.ABL_COL_SUBSET]) if s != baseline_name]

        fig, axes = plt.subplots(
            len(metrics), len(targets),
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
                # A drop is a loss whatever the metric's polarity: for
                # r2/spearman that is a negative delta, for rmse/mae positive.
                colors = [
                    const.ABL_COLOR_LOSS if (value < 0) == higher_is_better else const.ABL_COLOR_GAIN
                    for value in values
                ]
                ax.barh(positions, values, color=colors)
                ax.set_yticks(positions)
                ax.set_yticklabels(subsets, fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0.0, linewidth=1.0, color="black")
                ax.set_xlabel(f"delta {metric.value} vs {baseline_name}")
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
        return fig


# =============================================================================
# Attribution figures (Task 7)
# =============================================================================
class _AttributionFigure(Figure):
    """Shared helper: pick the importance kind both attribution figures show.

    Permutation is preferred when present, because impurity importances live
    on post-PCA inputs and on a different scale; drawing both on one axis
    would invite a comparison that is not meaningful.
    """

    def __init__(self, table: pd.DataFrame, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.table = table

    def is_drawable(self) -> bool:
        """No attribution rows, no figure."""
        return not self.table.empty

    @property
    def kind(self) -> str:
        """The importance kind to draw."""
        kinds = list(dict.fromkeys(self.table[const.ATTR_COL_KIND]))
        preferred = const.ImportanceKind.PERMUTATION.value
        return preferred if preferred in kinds else kinds[0]

    @property
    def rows_for_kind(self) -> pd.DataFrame:
        """The table restricted to the chosen kind."""
        return self.table[self.table[const.ATTR_COL_KIND] == self.kind]


class FeatureImportance(_AttributionFigure):
    """Top features per target, coloured by feature group.

    Error bars are the between-fold standard deviation. A bar whose whisker
    crosses zero is not a stable attribution on this dataset, and showing that
    is the point of drawing them.
    """

    file_key = "fig_feature_importance"

    def draw(self, plt: Any) -> Any:
        """Build the top-feature bars with fold error bars."""
        cfg = self.cfg
        top_k = int(cfg["attribution"]["top_k"])
        rows_for_kind = self.rows_for_kind
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
        present = set(rows_for_kind[const.ATTR_COL_GROUP])
        labels = [group for group in const.ATTR_GROUP_COLORS if group in present]
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=const.ATTR_GROUP_COLORS[group]) for group in labels
        ]
        if handles:
            axes[0][0].legend(handles, labels, title="feature group", fontsize=8, loc="lower right")
        fig.suptitle(
            f"Top {top_k} features by {self.kind} importance, averaged over "
            f"{cfg['validation']['primary_strategy']} folds"
        )
        return fig


class GroupImportance(_AttributionFigure):
    """Importance summed within each feature group, per target.

    The figure that answers the ablation question directly: how much of the
    model's realised skill comes from each *source* of information, rather
    than from any one column.
    """

    file_key = "fig_group_importance"

    def draw(self, plt: Any) -> Any:
        """Build the per-group summed bars."""
        cfg = self.cfg
        rows_for_kind = self.rows_for_kind
        targets = [t for t in const.TARGETS.values() if t in set(rows_for_kind[const.COL_TARGET])]
        groups = [
            g for g in const.ATTR_GROUP_COLORS if g in set(rows_for_kind[const.ATTR_COL_GROUP])
        ]

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
        fig.suptitle(
            f"Where the model's skill comes from, by feature group ({self.kind} importance)"
        )
        return fig
