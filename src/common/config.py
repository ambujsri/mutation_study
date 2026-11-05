"""Configuration loading, validation, path resolution and logging setup.

Shared by every pipeline stage so that no stage re-implements config handling.
Validation is fail-fast: every problem in the file is collected and reported in
a single ``ConfigError`` so a user can fix them all in one pass.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from src.common.constants import (
    CONFIG_BOUNDS,
    CONFIG_ENCODING,
    FEATURE_GROUPS,
    LOG_FORMAT,
    LOG_LEVELS,
    REQUIRED_CONFIG_SECTIONS,
    Activation,
    Bound,
    Encoding,
    ImportanceKind,
    Imputation,
    Metric,
    ModelName,
    PointEstimate,
    ReductionMethod,
    Scaler,
    ScoreNormalization,
    SplitStrategy,
    TorchDevice,
)

# Keys that must be YAML booleans (true / false).
_BOOLEAN_KEYS = (
    "features.add_abs_deltas",
    "features.add_relative_position",
    "features.embedding_reduction.enabled",
    "quality.replace_inf_with_nan",
    "quality.drop_rows_with_missing_target",
    "quality.fail_on_row_loss_in_merge",
    "quality.fail_on_duplicate_ids",
    "validation.shuffle",
    "evaluation.figures",
    "torch.batch_norm",
    "torch.standardize_stability_target",
    "torch.save_fold_models",
    "torch_v2.batch_norm",
    "torch_v2.standardize_stability_target",
    "torch_v2.save_fold_models",
    "torch_v2.embedding_pca",
    "attribution.group_embeddings_as_block",
    "prioritization.ood_flag_unseen_family",
    "prioritization.viability.enabled",
    "prioritization.viability.exclude_non_viable",
    "models.refit_on_full_data",
    "models.save_fold_pipelines",
)
_MODEL_PARAMS_PREFIX = "models.candidates."

# Project root = parent of src/. Used only to resolve relative paths in the config.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_MISSING = object()  # sentinel distinguishing "key absent" from an explicit null


class ConfigError(ValueError):
    """Raised when config.yaml cannot be parsed or holds missing / unsupported values."""


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the YAML configuration file.

    Args:
        config_path: Path to ``config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ConfigError: If the file is not valid YAML or fails validation.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with config_path.open("r", encoding=CONFIG_ENCODING) as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {config_path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(
            f"Config {config_path} must be a YAML mapping of sections, got {type(cfg).__name__}"
        )
    validate_config(cfg)
    return cfg


def get_by_path(cfg: dict[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """Return a nested config value addressed as ``"section.sub.key"``.

    Args:
        cfg: Configuration dictionary.
        dotted_key: Dot-separated path into nested mappings.
        default: Value returned if any component is absent; if omitted, raise.

    Raises:
        KeyError: If the path does not exist and no default was given.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(dotted_key)
            return default
        node = node[part]
    return node


def _check_choice(cfg: dict[str, Any], dotted_key: str, allowed: list[str]) -> list[str]:
    """Return an error if the value at ``dotted_key`` is not one of ``allowed``."""
    value = get_by_path(cfg, dotted_key, None)
    return [] if value in allowed else [f"{dotted_key}={value!r} not in {allowed}"]


def _check_bound(cfg: dict[str, Any], dotted_key: str, bound: Bound) -> list[str]:
    """Return errors if the value at ``dotted_key`` is missing, mistyped or out of range."""
    try:
        value = get_by_path(cfg, dotted_key)
    except KeyError:
        return [f"{dotted_key} is missing"]
    expected = "an integer" if bound.integer else "a number"
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    if not is_number or (bound.integer and not isinstance(value, int)) or not math.isfinite(value):
        return [f"{dotted_key}={value!r} must be {expected}"]
    too_low = bound.low is not None and (value <= bound.low if bound.low_open else value < bound.low)
    too_high = bound.high is not None and value > bound.high
    if too_low or too_high:
        low = "-inf" if bound.low is None else bound.low
        high = "inf" if bound.high is None else bound.high
        interval = f"{'(' if bound.low_open else '['}{low}, {high}]"
        return [f"{dotted_key}={value!r} outside allowed range {interval}"]
    return []


def _check_subset(cfg: dict[str, Any], dotted_key: str, allowed: list[str]) -> list[str]:
    """Return errors unless ``dotted_key`` holds a non-empty, duplicate-free subset of ``allowed``."""
    values = get_by_path(cfg, dotted_key, None)
    if not isinstance(values, list) or not values:
        return [f"{dotted_key} must be a non-empty list drawn from {allowed}"]
    errors = []
    unknown = [v for v in values if v not in allowed]
    if unknown:
        errors.append(f"{dotted_key} has unsupported {unknown}; allowed {allowed}")
    if len(set(map(str, values))) != len(values):
        errors.append(f"{dotted_key} has duplicates: {values}")
    return errors


def _check_layer_widths(cfg: dict[str, Any], dotted_key: str) -> list[str]:
    """Return errors unless ``dotted_key`` holds a non-empty list of positive ints."""
    dims = get_by_path(cfg, dotted_key, None)
    if not isinstance(dims, list) or not dims:
        return [f"{dotted_key} must be a non-empty list of layer widths"]
    if any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in dims):
        return [f"{dotted_key} must all be positive integers, got {dims}"]
    return []


def _check_head_width(cfg: dict[str, Any], dotted_key: str) -> list[str]:
    """Return errors unless ``dotted_key`` is a positive int or ``null``."""
    head = get_by_path(cfg, dotted_key, None)
    if head is not None and (not isinstance(head, int) or isinstance(head, bool) or head < 1):
        return [f"{dotted_key} must be a positive integer or null, got {head!r}"]
    return []


def _check_torch_layers(cfg: dict[str, Any]) -> list[str]:
    """Return errors unless both networks' layer widths are positive integers.

    ``hidden_dims`` defines the v1 shared backbone and the v2 branches, and
    each must hold at least one layer; a ``head_hidden_dim`` of ``null`` makes
    each task head a single ``Linear`` projection.
    """
    errors: list[str] = []
    for key in (
        "torch.hidden_dims",
        "torch_v2.tabular_branch.hidden_dims",
        "torch_v2.embedding_branch.hidden_dims",
    ):
        errors.extend(_check_layer_widths(cfg, key))
    for key in ("torch.head_hidden_dim", "torch_v2.head_hidden_dim"):
        errors.extend(_check_head_width(cfg, key))
    return errors


def _enum_values(enum_cls: type[Enum]) -> list[str]:
    """List the string values of an ``Enum`` (the allowed config choices)."""
    return [member.value for member in enum_cls]


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail fast on missing sections, unsupported choices or out-of-range numbers.

    Checks, in order:
        1. All ``REQUIRED_CONFIG_SECTIONS`` are present (stops here if not).
        2. Closed-set choices and lists match their ``Enum`` / allowed values;
           flags are booleans; ``models.candidates`` names only known models.
        3. Every key in ``CONFIG_BOUNDS`` is a finite number of the right type
           inside its range.
        4. Cross-field rules (primary CV strategy is one that is run; the
           selection metric is one that is computed; hyperparameter bounds only
           for configured models; ranking weights are not all zero).

    Raises:
        ConfigError: Listing every problem found, with the allowed values.
    """
    missing = [s for s in REQUIRED_CONFIG_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"Invalid config: missing sections {missing}")

    strategies_allowed = _enum_values(SplitStrategy)
    errors = [
        *_check_choice(cfg, "features.secondary_structure_encoding", _enum_values(Encoding)),
        *_check_choice(cfg, "quality.numeric_imputation", _enum_values(Imputation)),
        *_check_choice(cfg, "validation.primary_strategy", strategies_allowed),
        *_check_choice(cfg, "logging.level", list(LOG_LEVELS)),
    ]

    errors.extend(_check_subset(cfg, "validation.strategies", strategies_allowed))
    strategies = get_by_path(cfg, "validation.strategies", None)
    primary = get_by_path(cfg, "validation.primary_strategy", None)
    if isinstance(strategies, list) and primary in strategies_allowed and primary not in strategies:
        errors.append(f"validation.primary_strategy={primary!r} is not in validation.strategies")

    errors.extend(_check_subset(cfg, "validation.metrics", _enum_values(Metric)))
    metrics = get_by_path(cfg, "validation.metrics", None)
    selection = get_by_path(cfg, "models.selection_metric", None)
    errors.extend(_check_choice(cfg, "models.selection_metric", _enum_values(Metric)))
    if isinstance(metrics, list) and selection in _enum_values(Metric) and selection not in metrics:
        errors.append(f"models.selection_metric={selection!r} is not in validation.metrics")

    models_allowed_names = _enum_values(ModelName)
    errors.extend(_check_choice(cfg, "models.scaler", _enum_values(Scaler)))
    errors.extend(_check_choice(cfg, "prioritization.point_estimate", _enum_values(PointEstimate)))

    # Ablation: a null strategy/model means "inherit"; otherwise validate the choice.
    ablation_strategy = get_by_path(cfg, "ablation.strategy", None)
    if ablation_strategy is not None and ablation_strategy not in strategies_allowed:
        errors.append(f"ablation.strategy={ablation_strategy!r} not in {strategies_allowed} (or null)")
    ablation_model = get_by_path(cfg, "ablation.model", None)
    if ablation_model is not None and ablation_model not in models_allowed_names:
        errors.append(f"ablation.model={ablation_model!r} not in {models_allowed_names} (or null)")

    subsets = get_by_path(cfg, "ablation.subsets", None)
    known_groups = sorted(FEATURE_GROUPS)
    if not isinstance(subsets, dict) or not subsets:
        errors.append("ablation.subsets must be a non-empty mapping of subset name to group list")
        subsets = {}
    else:
        for name, groups in subsets.items():
            if not isinstance(groups, list) or not groups:
                errors.append(f"ablation.subsets.{name} must be a non-empty list from {known_groups}")
                continue
            unknown_groups = [g for g in groups if g not in FEATURE_GROUPS]
            if unknown_groups:
                errors.append(
                    f"ablation.subsets.{name} has unknown groups {unknown_groups}; allowed {known_groups}"
                )
    baseline = get_by_path(cfg, "ablation.baseline", None)
    if subsets and baseline not in subsets:
        errors.append(f"ablation.baseline={baseline!r} is not one of ablation.subsets {sorted(subsets)}")

    report_features = get_by_path(cfg, "interpretation.report_features", None)
    if not isinstance(report_features, list) or not report_features:
        errors.append("interpretation.report_features must be a non-empty list of column names")
    errors.extend(_check_choice(cfg, "prioritization.score_normalization", _enum_values(ScoreNormalization)))
    errors.extend(_check_choice(cfg, "features.embedding_reduction.method", _enum_values(ReductionMethod)))
    errors.extend(_check_choice(cfg, "torch.device", _enum_values(TorchDevice)))
    errors.extend(_check_choice(cfg, "torch.activation", _enum_values(Activation)))
    errors.extend(_check_choice(cfg, "torch_v2.device", _enum_values(TorchDevice)))
    errors.extend(_check_choice(cfg, "torch_v2.activation", _enum_values(Activation)))
    errors.extend(_check_choice(cfg, "attribution.scoring_metric", _enum_values(Metric)))
    errors.extend(_check_subset(cfg, "attribution.kinds", _enum_values(ImportanceKind)))
    errors.extend(_check_torch_layers(cfg))

    candidates = get_by_path(cfg, "models.candidates", None)
    models_allowed = _enum_values(ModelName)
    if not isinstance(candidates, dict) or not candidates:
        errors.append(f"models.candidates must be a non-empty mapping with keys from {models_allowed}")
        candidates = {}
    else:
        unknown_models = [m for m in candidates if m not in models_allowed]
        if unknown_models:
            errors.append(f"models.candidates has unsupported {unknown_models}; allowed {models_allowed}")
        bad_params = [m for m, params in candidates.items() if not isinstance(params, dict)]
        if bad_params:
            errors.append(f"models.candidates entries {bad_params} must map to a set of hyperparameters")

    for dotted_key in _BOOLEAN_KEYS:
        if not isinstance(get_by_path(cfg, dotted_key, None), bool):
            errors.append(f"{dotted_key} must be true or false")

    for dotted_key, bound in CONFIG_BOUNDS.items():
        # Hyperparameter bounds apply only to models that are actually configured.
        if dotted_key.startswith(_MODEL_PARAMS_PREFIX):
            model = dotted_key.removeprefix(_MODEL_PARAMS_PREFIX).split(".")[0]
            if not isinstance(candidates.get(model), dict):
                continue
        errors.extend(_check_bound(cfg, dotted_key, bound))

    weights = get_by_path(cfg, "prioritization.weights", {})
    if (
        isinstance(weights, dict)
        and weights
        and not any(isinstance(w, int | float) and w > 0 for w in weights.values())
    ):
        errors.append("prioritization.weights must contain at least one positive weight")

    if errors:
        raise ConfigError("Invalid config:\n  - " + "\n  - ".join(errors))


def setup_logging(cfg: dict[str, Any]) -> None:
    """Configure root logging; the level comes from config, the format is a constant."""
    logging.basicConfig(level=cfg["logging"]["level"], format=LOG_FORMAT, force=True)


def resolve_path(cfg: dict[str, Any], dir_key: str, file_key: str | None = None) -> Path:
    """Build an absolute path from configured directory / file keys.

    Absolute paths in the config are used as-is; relative ones are anchored at
    the project root.

    Args:
        cfg: Configuration dictionary.
        dir_key: Key under ``cfg["paths"]`` (e.g. ``"raw_dir"``).
        file_key: Optional key under ``cfg["files"]`` (e.g. ``"labeled"``).

    Returns:
        Absolute path.
    """
    path = PROJECT_ROOT / cfg["paths"][dir_key]
    return path / cfg["files"][file_key] if file_key else path
