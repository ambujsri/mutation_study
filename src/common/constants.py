"""Project-wide constants: fixed facts about the dataset and code-level conventions.

What belongs here (see docs/CODING_GUIDELINES.md, section 4):
    * Dataset schema facts — column names, embedding layout, categorical codes.
    * Data contracts — documented value ranges and expected table sizes.
    * Naming conventions for engineered columns.
    * Closed sets of allowed choices (enums) that config values are checked against.

What does NOT belong here:
    * Anything a scientist may change between runs (seed, paths, hyperparameters,
      encoding/imputation choice, ranking weights). Those live in config/config.yaml.

All collections are immutable (tuples / MappingProxyType) so no module can
mutate a shared constant at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final

# =============================================================================
# Dataset table keys (logical names for the three raw inputs)
# =============================================================================
TABLE_LABELED: Final = "labeled"
TABLE_CANDIDATES: Final = "candidates"
TABLE_EMBEDDINGS: Final = "embeddings"

# =============================================================================
# Column names — identifiers and grouping
# =============================================================================
COL_MUTATION_ID: Final = "mutation_id"
COL_PROTEIN_ID: Final = "protein_id"
COL_PROTEIN_FAMILY: Final = "protein_family"
COL_PROTEIN_LENGTH: Final = "protein_length"

JOIN_KEY: Final = COL_PROTEIN_ID
GROUP_COLUMNS: Final = MappingProxyType({"protein": COL_PROTEIN_ID, "family": COL_PROTEIN_FAMILY})

# Present in both mutation files and the embedding file; checked for agreement
# on merge and kept once (from the mutation side).
SHARED_PROTEIN_COLUMNS: Final = (COL_PROTEIN_FAMILY, COL_PROTEIN_LENGTH)

# =============================================================================
# Column names — residue and structural features
# =============================================================================
COL_WT_AA: Final = "wt_amino_acid"
COL_MUTANT_AA: Final = "mutant_amino_acid"
COL_POSITION: Final = "position"
COL_SECONDARY_STRUCTURE: Final = "secondary_structure"

COL_CONSERVATION: Final = "conservation_score"
COL_SOLVENT_ACCESSIBILITY: Final = "solvent_accessibility"
COL_DIST_ACTIVE_SITE: Final = "distance_to_active_site"
COL_LOCAL_CONFIDENCE: Final = "local_confidence_score"

COL_WT_HYDROPHOBICITY: Final = "wt_hydrophobicity"
COL_MUT_HYDROPHOBICITY: Final = "mutant_hydrophobicity"
COL_WT_CHARGE: Final = "wt_charge"
COL_MUT_CHARGE: Final = "mutant_charge"
COL_WT_VOLUME: Final = "wt_volume"
COL_MUT_VOLUME: Final = "mutant_volume"

RESIDUE_COLUMNS: Final = (COL_WT_AA, COL_MUTANT_AA, COL_POSITION)

NUMERIC_FEATURE_COLUMNS: Final = (
    COL_PROTEIN_LENGTH,
    COL_POSITION,
    COL_CONSERVATION,
    COL_SOLVENT_ACCESSIBILITY,
    COL_DIST_ACTIVE_SITE,
    COL_LOCAL_CONFIDENCE,
    COL_WT_HYDROPHOBICITY,
    COL_MUT_HYDROPHOBICITY,
    COL_WT_CHARGE,
    COL_MUT_CHARGE,
    COL_WT_VOLUME,
    COL_MUT_VOLUME,
)

# =============================================================================
# Targets
# =============================================================================
TARGET_STABILITY: Final = "stability_ddG"  # negative = stabilizing
TARGET_FUNCTION: Final = "functional_score"  # 0 = severe loss, 1 = wild-type-like
TARGETS: Final = MappingProxyType({"stability": TARGET_STABILITY, "function": TARGET_FUNCTION})

# =============================================================================
# Protein embeddings
# =============================================================================
EMBEDDING_PREFIX: Final = "protein_emb_"
EMBEDDING_DIM: Final = 256
# protein_emb_001 … protein_emb_256 (index zero-padded to the width of EMBEDDING_DIM)
EMBEDDING_COLUMNS: Final = tuple(
    f"{EMBEDDING_PREFIX}{i:0{len(str(EMBEDDING_DIM))}d}" for i in range(1, EMBEDDING_DIM + 1)
)

# =============================================================================
# Secondary-structure encoding
# =============================================================================
SS_CATEGORIES: Final = ("H", "E", "C")  # helix, beta strand, coil/loop
SS_ONEHOT_PREFIX: Final = "ss"  # -> ss_H, ss_E, ss_C
SS_ORDINAL_COLUMN: Final = "ss_ordinal"
# Ranks regular structure above coil (H > E > C); a modelling assumption, used
# only when the ordinal encoding is selected in config.
SS_ORDINAL_MAP: Final = MappingProxyType({"H": 2, "E": 1, "C": 0})

# =============================================================================
# Engineered-feature definitions and naming
# =============================================================================
# delta_<name> = mutant column − wild-type column
DELTA_PAIRS: Final = MappingProxyType(
    {
        "hydrophobicity": (COL_WT_HYDROPHOBICITY, COL_MUT_HYDROPHOBICITY),
        "charge": (COL_WT_CHARGE, COL_MUT_CHARGE),
        "volume": (COL_WT_VOLUME, COL_MUT_VOLUME),
    }
)
DELTA_PREFIX: Final = "delta_"
ABS_DELTA_PREFIX: Final = "abs_delta_"
COL_RELATIVE_POSITION: Final = "relative_position"

# =============================================================================
# Data contracts (from README_DATASET.txt)
# =============================================================================
# (lower, upper) bounds; None = unbounded. Violations are reported, not clipped.
VALUE_RANGES: Final = MappingProxyType(
    {
        COL_CONSERVATION: (0.0, 1.0),
        COL_SOLVENT_ACCESSIBILITY: (0.0, 1.0),
        COL_LOCAL_CONFIDENCE: (0.0, 100.0),
        COL_DIST_ACTIVE_SITE: (0.0, None),
        COL_PROTEIN_LENGTH: (1, None),
        COL_POSITION: (1, None),
        TARGET_FUNCTION: (0.0, 1.0),
    }
)
EXPECTED_COUNTS: Final = MappingProxyType(
    {"labeled_rows": 840, "candidate_rows": 180, "proteins": 36, "families": 9}
)


# =============================================================================
# Allowed choices for config values (validated in src.common.config)
# =============================================================================
class Encoding(str, Enum):
    """Secondary-structure encoding methods."""

    ONEHOT = "onehot"
    ORDINAL = "ordinal"


class Imputation(str, Enum):
    """Numeric imputation strategies (names match pandas reductions)."""

    MEDIAN = "median"
    MEAN = "mean"


class SplitStrategy(str, Enum):
    """Cross-validation splitting strategies, from most to least optimistic."""

    RANDOM = "random"  # mutation-level K-fold; leaks protein/family identity
    PROTEIN_GROUP = "protein_group"  # no protein in both train and validation
    FAMILY_GROUP = "family_group"  # no family in both train and validation (OOD proxy)


# Grouping column per strategy; RANDOM has none.
SPLIT_GROUP_COLUMN: Final = MappingProxyType(
    {
        SplitStrategy.RANDOM: None,
        SplitStrategy.PROTEIN_GROUP: COL_PROTEIN_ID,
        SplitStrategy.FAMILY_GROUP: COL_PROTEIN_FAMILY,
    }
)


class ModelName(str, Enum):
    """Regressors available to ``models.candidates`` (registry in src.train)."""

    RIDGE = "ridge"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class Scaler(str, Enum):
    """Feature scaling applied inside each training fold."""

    STANDARD = "standard"
    ROBUST = "robust"
    NONE = "none"


class ReductionMethod(str, Enum):
    """Embedding dimensionality-reduction methods (fit inside each training fold)."""

    PCA = "pca"


class Metric(str, Enum):
    """Regression metrics computed on validation folds and pooled out-of-fold predictions."""

    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    SPEARMAN = "spearman"


# Direction used for model selection: True if a larger value is better.
METRIC_GREATER_IS_BETTER: Final = MappingProxyType(
    {Metric.RMSE: False, Metric.MAE: False, Metric.R2: True, Metric.SPEARMAN: True}
)

LOG_LEVELS: Final = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
SUPPORTED_TABLE_FORMATS: Final = (".parquet", ".csv")


@dataclass(frozen=True)
class Bound:
    """Allowed numeric range for a config value.

    Attributes:
        low: Lower bound, or ``None`` for unbounded.
        high: Upper bound (inclusive), or ``None`` for unbounded.
        integer: Whether the value must be an ``int`` (``bool`` is rejected).
        low_open: Whether ``low`` itself is excluded (e.g. learning rate > 0).
    """

    low: float | None = None
    high: float | None = None
    integer: bool = False
    low_open: bool = False


# Dotted config key -> allowed range. Checked at load time by src.common.config.
CONFIG_BOUNDS: Final = MappingProxyType(
    {
        "project.random_seed": Bound(0, 2**32 - 1, integer=True),
        "features.embedding_reduction.n_components": Bound(1, EMBEDDING_DIM, integer=True),
        "validation.n_splits": Bound(2, None, integer=True),
        "evaluation.figure_dpi": Bound(50, 600, integer=True),
        "models.candidates.ridge.alpha": Bound(0, None, low_open=True),
        "models.candidates.random_forest.n_estimators": Bound(1, None, integer=True),
        "models.candidates.random_forest.min_samples_leaf": Bound(1, None, integer=True),
        "models.candidates.random_forest.max_features": Bound(0, 1, low_open=True),
        "models.candidates.gradient_boosting.n_estimators": Bound(1, None, integer=True),
        "models.candidates.gradient_boosting.learning_rate": Bound(0, 1, low_open=True),
        "models.candidates.gradient_boosting.max_depth": Bound(1, None, integer=True),
        "models.candidates.gradient_boosting.subsample": Bound(0, 1, low_open=True),
        "prioritization.top_k": Bound(1, None, integer=True),
        "prioritization.min_fold_models": Bound(1, None, integer=True),
        "prioritization.weights.stability": Bound(0, None),
        "prioritization.weights.function": Bound(0, None),
        "prioritization.uncertainty_penalty": Bound(0, None),
    }
)

# =============================================================================
# Cross-validation report schema
# =============================================================================
CV_COL_STRATEGY: Final = "strategy"
CV_COL_FOLD: Final = "fold"
CV_COL_N_TRAIN: Final = "n_train"
CV_COL_N_VAL: Final = "n_val"
CV_COL_VAL_FRACTION: Final = "val_fraction"
CV_COL_LEAKED_PROTEINS: Final = "leaked_proteins"  # proteins in both train and validation
CV_COL_LEAKED_FAMILIES: Final = "leaked_families"  # families in both train and validation
CV_PREFIX_N_VAL_GROUPS: Final = "n_val_"  # -> n_val_protein, n_val_family
CV_PREFIX_VAL_MEAN: Final = "val_mean_"  # -> val_mean_stability_ddG, ...

# =============================================================================
# Training outputs: prediction / metric table schema and artifact naming
# =============================================================================
COL_MODEL: Final = "model"
COL_TARGET: Final = "target"
COL_Y_TRUE: Final = "y_true"
COL_Y_PRED: Final = "y_pred"
COL_N_SAMPLES: Final = "n_samples"
POOLED_PREFIX: Final = "pooled_"  # metric on all out-of-fold predictions at once
FOLD_MEAN_PREFIX: Final = "fold_mean_"  # mean of per-fold metrics
FOLD_STD_PREFIX: Final = "fold_std_"  # spread of per-fold metrics

# Pipeline step / transformer names (stable so saved pipelines can be inspected).
STEP_PREPROCESS: Final = "preprocess"
STEP_MODEL: Final = "model"
BLOCK_TABULAR: Final = "tabular"
BLOCK_EMBEDDING: Final = "embedding"
STEP_IMPUTE: Final = "impute"
STEP_SCALE: Final = "scale"
STEP_REDUCE: Final = "reduce"

FINAL_MODEL_TEMPLATE: Final = "final__{target}__{model}.joblib"
FOLD_MODEL_TEMPLATE: Final = "{strategy}__fold{fold}__{target}__{model}.joblib"
FOLD_MODELS_SUBDIR: Final = "fold_pipelines"
JOBLIB_COMPRESSION: Final = 3


# =============================================================================
# Phase 4: inference, uncertainty and prioritization
# =============================================================================
class PointEstimate(str, Enum):
    """Which prediction is carried forward as each candidate's point estimate."""

    ENSEMBLE_MEAN = "ensemble_mean"  # mean over the CV fold pipelines
    FINAL_MODEL = "final_model"  # the model refit on all labeled data


class ScoreNormalization(str, Enum):
    """How objectives are put on a common scale before weighting."""

    RANK = "rank"  # percentile rank within the candidate pool (robust to outliers)
    MINMAX = "minmax"  # linear rescaling to [0, 1] (keeps relative spacing)


# Prediction column naming: <prefix><target>, e.g. pred_stability_ddG.
PRED_PREFIX: Final = "pred_"
PRED_STD_PREFIX: Final = "pred_std_"  # spread across fold models
PRED_FINAL_PREFIX: Final = "pred_final_"  # refit-on-all-data model, for comparison
NORM_PREFIX: Final = "norm_"  # normalized objective contribution

COL_PRIORITY_SCORE: Final = "priority_score"
COL_PRIORITY_RANK: Final = "priority_rank"
COL_TOTAL_UNCERTAINTY: Final = "total_uncertainty"
COL_NORM_UNCERTAINTY: Final = "norm_uncertainty"
COL_UNSEEN_PROTEIN: Final = "is_unseen_protein"
COL_UNSEEN_FAMILY: Final = "is_unseen_family"
COL_OOD_FLAG: Final = "ood_flag"
COL_N_FOLD_MODELS: Final = "n_fold_models"
COL_JUSTIFICATION: Final = "justification"

# Values of COL_OOD_FLAG, from most to least confident.
OOD_IN_DISTRIBUTION: Final = "in_distribution"
OOD_UNSEEN_PROTEIN: Final = "unseen_protein"
OOD_UNSEEN_FAMILY: Final = "unseen_family"

# Columns of the Top-K report, in order. Embeddings are never reported.
TOP_K_REPORT_COLUMNS: Final = (
    COL_PRIORITY_RANK,
    COL_MUTATION_ID,
    COL_PROTEIN_ID,
    COL_PROTEIN_FAMILY,
    COL_WT_AA,
    COL_POSITION,
    COL_MUTANT_AA,
    COL_SECONDARY_STRUCTURE,
    COL_CONSERVATION,
    COL_SOLVENT_ACCESSIBILITY,
    COL_DIST_ACTIVE_SITE,
    f"{PRED_PREFIX}{TARGET_STABILITY}",
    f"{PRED_STD_PREFIX}{TARGET_STABILITY}",
    f"{PRED_PREFIX}{TARGET_FUNCTION}",
    f"{PRED_STD_PREFIX}{TARGET_FUNCTION}",
    COL_TOTAL_UNCERTAINTY,
    COL_PRIORITY_SCORE,
    COL_OOD_FLAG,
    COL_JUSTIFICATION,
)

MARKDOWN_FLOAT_FORMAT: Final = ".3f"

# =============================================================================
# Ablation study: named feature groups
# =============================================================================
# Raw per-residue physicochemical properties of the wild-type and mutant residue.
GROUP_PHYSICOCHEMICAL: Final = "physicochemical"
# Structural microenvironment of the mutated position.
GROUP_STRUCTURAL: Final = "structural"
# Where the mutation sits in the sequence.
GROUP_SEQUENCE: Final = "sequence"
# Engineered mutant-minus-wild-type shifts (Phase 1 feature engineering).
GROUP_DELTAS: Final = "deltas"
# Pretrained protein embeddings (reduced by PCA inside each training fold).
GROUP_EMBEDDINGS: Final = "embeddings"

# Group -> the columns it contributes. Tabular groups are resolved against the
# Phase 1 feature manifest; the embedding group maps to EMBEDDING_COLUMNS.
FEATURE_GROUPS: Final = MappingProxyType(
    {
        GROUP_PHYSICOCHEMICAL: (
            COL_WT_HYDROPHOBICITY,
            COL_MUT_HYDROPHOBICITY,
            COL_WT_CHARGE,
            COL_MUT_CHARGE,
            COL_WT_VOLUME,
            COL_MUT_VOLUME,
        ),
        GROUP_STRUCTURAL: (
            COL_CONSERVATION,
            COL_SOLVENT_ACCESSIBILITY,
            COL_DIST_ACTIVE_SITE,
            COL_LOCAL_CONFIDENCE,
            *(f"{SS_ONEHOT_PREFIX}_{code}" for code in SS_CATEGORIES),
            SS_ORDINAL_COLUMN,
        ),
        GROUP_SEQUENCE: (COL_PROTEIN_LENGTH, COL_POSITION, COL_RELATIVE_POSITION),
        GROUP_DELTAS: (
            *(f"{DELTA_PREFIX}{name}" for name in DELTA_PAIRS),
            *(f"{ABS_DELTA_PREFIX}{name}" for name in DELTA_PAIRS),
        ),
        GROUP_EMBEDDINGS: EMBEDDING_COLUMNS,
    }
)

# =============================================================================
# Evaluation report and figures
# =============================================================================
EVAL_COL_METRIC: Final = "metric"
EVAL_COL_OPTIMISM_GAP: Final = "optimism_gap"  # random-split score minus grouped-split score

# Figure artifacts, keyed by their entry in the config `files` section. Adding a
# figure means adding its key here and a filename in config.yaml.
FIGURE_FILE_KEYS: Final = ("fig_pred_vs_actual", "fig_residuals", "fig_strategy_comparison")
# Non-interactive backend: figures are written to disk, never displayed, so the
# pipeline runs identically on a headless machine.
MATPLOTLIB_BACKEND: Final = "Agg"
RESIDUAL_HIST_BINS: Final = 30
FIGURE_XTICK_ROTATION: Final = 20

# Ablation report schema.
ABL_COL_SUBSET: Final = "subset"
ABL_COL_GROUPS: Final = "groups"
ABL_COL_N_TABULAR: Final = "n_tabular"
ABL_COL_N_EMBEDDING: Final = "n_embedding"
ABL_DELTA_PREFIX: Final = "delta_vs_baseline_"

# =============================================================================
# Infrastructure
# =============================================================================
LOG_FORMAT: Final = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
CONFIG_ENCODING: Final = "utf-8"
REQUIRED_CONFIG_SECTIONS: Final = (
    "project",
    "paths",
    "files",
    "features",
    "quality",
    "validation",
    "models",
    "evaluation",
    "ablation",
    "interpretation",
    "prioritization",
    "logging",
)
