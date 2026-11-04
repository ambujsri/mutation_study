"""Unit tests for ``src.common.config``: loading, validation, paths and logging.

Every config file under test is written to ``tmp_path``; the real
``config/config.yaml`` is only read (via the ``base_config`` fixture) as the
known-good starting point that each test perturbs.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.common import constants as const
from src.common.config import (
    PROJECT_ROOT,
    ConfigError,
    get_by_path,
    load_config,
    resolve_path,
    setup_logging,
    validate_config,
)

WriteConfig = Callable[[Any], Path]


# =============================================================================
# Helpers & fixtures
# =============================================================================
def set_by_path(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value addressed as ``"section.sub.key"``."""
    *parents, leaf = dotted_key.split(".")
    node = cfg
    for part in parents:
        node = node[part]
    node[leaf] = value


def delete_by_path(cfg: dict[str, Any], dotted_key: str) -> None:
    """Delete a nested config value addressed as ``"section.sub.key"``."""
    *parents, leaf = dotted_key.split(".")
    node = cfg
    for part in parents:
        node = node[part]
    del node[leaf]


@pytest.fixture()
def good(base_config: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of the project config that a test may freely modify."""
    return copy.deepcopy(base_config)


@pytest.fixture()
def write_config(tmp_path: Path) -> WriteConfig:
    """Return a function that dumps a config object to a YAML file in ``tmp_path``."""

    def _write(content: Any) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(content, sort_keys=False), encoding=const.CONFIG_ENCODING)
        return path

    return _write


@pytest.fixture()
def restore_root_logger() -> Iterator[None]:
    """Restore the root logger's handlers and level after a test reconfigures logging."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def bound_violations() -> list[tuple[str, Any]]:
    """Generate one out-of-range or mistyped value per edge of every ``CONFIG_BOUNDS`` entry."""
    cases: list[tuple[str, Any]] = []
    for key, bound in const.CONFIG_BOUNDS.items():
        if bound.low is not None:
            below = bound.low if bound.low_open else bound.low - 1
            cases.append((key, below))
        if bound.high is not None:
            cases.append((key, bound.high + 1))
        cases.append((key, "not-a-number"))
        cases.append((key, True))  # bool is an int subclass and must still be rejected
        cases.append((key, math.nan))
        if bound.integer:
            low = 0 if bound.low is None else bound.low
            cases.append((key, float(low) + 0.5))
    return cases


def bound_edges() -> list[tuple[str, Any]]:
    """Generate the inclusive edge values of every bound, which must be accepted."""
    cases: list[tuple[str, Any]] = []
    for key, bound in const.CONFIG_BOUNDS.items():
        if bound.low is not None and not bound.low_open:
            cases.append((key, int(bound.low) if bound.integer else bound.low))
        if bound.high is not None:
            cases.append((key, int(bound.high) if bound.integer else bound.high))
    return cases


# =============================================================================
# Loading
# =============================================================================
def test_project_config_is_valid(base_config: dict[str, Any]) -> None:
    """The committed config.yaml passes validation and carries the dataset seed."""
    validate_config(base_config)
    assert base_config["project"]["random_seed"] == 20261102


def test_load_config_round_trips_valid_file(good: dict[str, Any], write_config: WriteConfig) -> None:
    """A valid YAML file loads into exactly the mapping that was written."""
    assert load_config(write_config(good)) == good


def test_load_config_accepts_str_path(good: dict[str, Any], write_config: WriteConfig) -> None:
    """Paths may be given as plain strings as well as ``Path`` objects."""
    assert load_config(str(write_config(good))) == good


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    """A missing config file raises ``FileNotFoundError`` naming the path."""
    missing = tmp_path / "absent.yaml"

    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        load_config(missing)


@pytest.mark.parametrize(
    "text",
    [
        "project: {random_seed: 1\n",  # unclosed flow mapping
        "project:\n  seed: 1\n bad_indent: 2\n",  # inconsistent indentation
        "key: [unterminated\n",  # unclosed list
        "a: b: c\n",  # mapping value where not allowed
    ],
)
def test_load_config_malformed_yaml_raises_config_error(tmp_path: Path, text: str) -> None:
    """Unparseable YAML becomes a ``ConfigError`` naming the file, chained to the parser error."""
    path = tmp_path / "broken.yaml"
    path.write_text(text, encoding=const.CONFIG_ENCODING)

    with pytest.raises(ConfigError, match=r"Malformed YAML in .*broken\.yaml") as exc_info:
        load_config(path)
    assert isinstance(exc_info.value.__cause__, yaml.YAMLError)


@pytest.mark.parametrize("content", [None, [], ["project"], "just a string", 42])
def test_load_config_non_mapping_raises(write_config: WriteConfig, content: Any) -> None:
    """Empty files and top-level lists / scalars are rejected with the type that was found."""
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_config(write_config(content))


def test_config_error_is_value_error() -> None:
    """``ConfigError`` subclasses ``ValueError`` so generic callers can still catch it."""
    assert issubclass(ConfigError, ValueError)


# =============================================================================
# Validation: sections and closed-set choices
# =============================================================================
@pytest.mark.parametrize("section", list(const.REQUIRED_CONFIG_SECTIONS))
def test_missing_section_raises(good: dict[str, Any], section: str) -> None:
    """Each required section is enforced, and its name appears in the error."""
    del good[section]

    with pytest.raises(ConfigError, match=f"missing sections.*{section}"):
        validate_config(good)


@pytest.mark.parametrize(
    ("key", "allowed"),
    [
        ("features.secondary_structure_encoding", [e.value for e in const.Encoding]),
        ("quality.numeric_imputation", [i.value for i in const.Imputation]),
        ("validation.primary_strategy", [s.value for s in const.SplitStrategy]),
        ("logging.level", list(const.LOG_LEVELS)),
    ],
)
@pytest.mark.parametrize("bad_value", ["binary", "", None, 3])
def test_unsupported_choice_lists_allowed_values(
    good: dict[str, Any], key: str, allowed: list[str], bad_value: Any
) -> None:
    """An unsupported choice is rejected and the message lists every allowed value."""
    set_by_path(good, key, bad_value)

    with pytest.raises(ConfigError) as exc_info:
        validate_config(good)
    message = str(exc_info.value)
    assert key in message
    assert str(allowed) in message


@pytest.mark.parametrize(
    "key", ["features.secondary_structure_encoding", "quality.numeric_imputation", "logging.level"]
)
def test_missing_choice_key_raises(good: dict[str, Any], key: str) -> None:
    """A choice key that is absent is reported, not silently defaulted."""
    delete_by_path(good, key)

    with pytest.raises(ConfigError, match=key):
        validate_config(good)


@pytest.mark.parametrize("value", [s.value for s in const.Encoding])
def test_every_encoding_choice_is_accepted(good: dict[str, Any], value: str) -> None:
    """All values of the ``Encoding`` enum are valid config choices."""
    set_by_path(good, "features.secondary_structure_encoding", value)
    validate_config(good)


# =============================================================================
# Validation: cross-validation settings
# =============================================================================
@pytest.mark.parametrize(
    ("strategies", "pattern"),
    [
        (["random", "kfold"], "unsupported.*kfold"),
        (["random", "random", "family_group"], "duplicates"),
        ([], "non-empty list"),
        ("family_group", "non-empty list"),
        (None, "non-empty list"),
    ],
)
def test_invalid_strategy_list_raises(good: dict[str, Any], strategies: Any, pattern: str) -> None:
    """The strategy list must be a non-empty, duplicate-free subset of ``SplitStrategy``."""
    good["validation"]["strategies"] = strategies
    good["validation"]["primary_strategy"] = const.SplitStrategy.RANDOM.value

    with pytest.raises(ConfigError, match=pattern):
        validate_config(good)


def test_primary_strategy_must_be_run(good: dict[str, Any]) -> None:
    """The primary strategy has to be one of the strategies that are actually executed."""
    good["validation"]["strategies"] = [const.SplitStrategy.RANDOM.value]
    good["validation"]["primary_strategy"] = const.SplitStrategy.FAMILY_GROUP.value

    with pytest.raises(ConfigError, match="primary_strategy.*not in validation.strategies"):
        validate_config(good)


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_shuffle_must_be_boolean(good: dict[str, Any], value: Any) -> None:
    """``validation.shuffle`` accepts only true/false (YAML ``yes`` would be read as a string here)."""
    good["validation"]["shuffle"] = value

    with pytest.raises(ConfigError, match="validation.shuffle"):
        validate_config(good)


# =============================================================================
# Validation: numeric bounds
# =============================================================================
@pytest.mark.parametrize(("key", "value"), bound_violations())
def test_out_of_range_or_mistyped_number_raises(good: dict[str, Any], key: str, value: Any) -> None:
    """Every bounded key rejects values outside its range, non-numbers, bools, NaN and non-integers."""
    set_by_path(good, key, value)

    with pytest.raises(ConfigError, match=rf"{key}=.*(outside allowed range|must be)"):
        validate_config(good)


@pytest.mark.parametrize(("key", "value"), bound_edges())
def test_inclusive_bound_edges_are_accepted(good: dict[str, Any], key: str, value: Any) -> None:
    """Inclusive edges of each range are valid (guards against off-by-one comparisons)."""
    set_by_path(good, key, value)
    validate_config(good)


@pytest.mark.parametrize("key", list(const.CONFIG_BOUNDS))
def test_missing_bounded_key_raises(good: dict[str, Any], key: str) -> None:
    """A bounded key that is absent is reported as missing."""
    delete_by_path(good, key)

    with pytest.raises(ConfigError, match=f"{key} is missing"):
        validate_config(good)


def test_range_message_shows_interval(good: dict[str, Any]) -> None:
    """The error states the allowed interval, using ``(`` for an exclusive lower bound."""
    set_by_path(good, "models.candidates.gradient_boosting.learning_rate", 0)

    with pytest.raises(ConfigError, match=r"outside allowed range \(0, 1\]"):
        validate_config(good)


def test_all_zero_ranking_weights_raise(good: dict[str, Any]) -> None:
    """At least one prioritization weight must be positive, or every candidate scores zero."""
    for objective in good["prioritization"]["weights"]:
        good["prioritization"]["weights"][objective] = 0

    with pytest.raises(ConfigError, match="at least one positive weight"):
        validate_config(good)


def test_all_errors_reported_together(good: dict[str, Any]) -> None:
    """Several problems are listed in one error so the user can fix them in a single pass."""
    set_by_path(good, "features.secondary_structure_encoding", "binary")
    set_by_path(good, "validation.n_splits", 1)
    set_by_path(good, "prioritization.top_k", 0)

    with pytest.raises(ConfigError) as exc_info:
        validate_config(good)
    message = str(exc_info.value)
    for key in ("secondary_structure_encoding", "validation.n_splits", "prioritization.top_k"):
        assert key in message


def test_load_config_rejects_invalid_file(good: dict[str, Any], write_config: WriteConfig) -> None:
    """``load_config`` runs validation, so an invalid file never reaches the pipeline."""
    set_by_path(good, "validation.n_splits", 1)

    with pytest.raises(ConfigError, match="validation.n_splits"):
        load_config(write_config(good))


# =============================================================================
# get_by_path, resolve_path, setup_logging
# =============================================================================
def test_get_by_path_reads_nested_value(good: dict[str, Any]) -> None:
    """Dotted keys address nested values."""
    assert get_by_path(good, "project.random_seed") == good["project"]["random_seed"]


def test_get_by_path_missing_raises_or_defaults(good: dict[str, Any]) -> None:
    """Absent paths raise ``KeyError`` unless a default is supplied (including ``None``)."""
    with pytest.raises(KeyError):
        get_by_path(good, "project.nope")
    assert get_by_path(good, "project.nope", None) is None
    assert get_by_path(good, "project.random_seed.deeper", "fallback") == "fallback"


def test_resolve_path_anchors_relative_paths_at_project_root(good: dict[str, Any]) -> None:
    """Relative configured paths resolve under the project root."""
    path = resolve_path(good, "raw_dir", "labeled")

    assert path == PROJECT_ROOT / good["paths"]["raw_dir"] / good["files"]["labeled"]
    assert path.is_absolute()


def test_resolve_path_keeps_absolute_paths(good: dict[str, Any], tmp_path: Path) -> None:
    """Absolute configured paths (e.g. a shared drive) are used unchanged."""
    good["paths"]["metrics_dir"] = str(tmp_path)

    assert resolve_path(good, "metrics_dir") == tmp_path
    assert resolve_path(good, "metrics_dir", "cv_fold_summary") == tmp_path / good["files"]["cv_fold_summary"]


@pytest.mark.usefixtures("restore_root_logger")
@pytest.mark.parametrize("level", list(const.LOG_LEVELS))
def test_setup_logging_applies_configured_level(good: dict[str, Any], level: str) -> None:
    """The root logger level follows ``logging.level`` from config."""
    good["logging"]["level"] = level

    setup_logging(good)

    assert logging.getLogger().level == logging.getLevelName(level)


# =============================================================================
# Validation: model, metric and preprocessing settings (Phase 3)
# =============================================================================
@pytest.mark.parametrize(
    ("key", "allowed"),
    [
        ("models.scaler", [s.value for s in const.Scaler]),
        ("models.selection_metric", [m.value for m in const.Metric]),
        ("features.embedding_reduction.method", [r.value for r in const.ReductionMethod]),
    ],
)
def test_unsupported_model_setting_lists_allowed_values(
    good: dict[str, Any], key: str, allowed: list[str]
) -> None:
    """Scaler, selection metric and reduction method reject unknown values and list the allowed ones."""
    set_by_path(good, key, "unsupported")

    with pytest.raises(ConfigError) as exc_info:
        validate_config(good)
    assert key in str(exc_info.value)
    assert str(allowed) in str(exc_info.value)


def test_unknown_model_rejected(good: dict[str, Any]) -> None:
    """A model name with no registered estimator is rejected, listing the available ones."""
    good["models"]["candidates"]["lightgbm"] = {"n_estimators": 100}

    with pytest.raises(ConfigError, match=r"unsupported \['lightgbm'\].*ridge"):
        validate_config(good)


@pytest.mark.parametrize("candidates", [{}, None, ["ridge"]])
def test_candidates_must_be_non_empty_mapping(good: dict[str, Any], candidates: Any) -> None:
    """At least one model must be configured, as a mapping of name to hyperparameters."""
    good["models"]["candidates"] = candidates

    with pytest.raises(ConfigError, match="models.candidates must be a non-empty mapping"):
        validate_config(good)


def test_model_hyperparameters_must_be_mapping(good: dict[str, Any]) -> None:
    """Each model entry must hold a set of hyperparameters."""
    good["models"]["candidates"]["ridge"] = 1.0

    with pytest.raises(ConfigError, match=r"\['ridge'\] must map to a set of hyperparameters"):
        validate_config(good)


def test_bounds_only_checked_for_configured_models(good: dict[str, Any]) -> None:
    """Removing a model from the candidates does not trigger 'missing hyperparameter' errors."""
    del good["models"]["candidates"][const.ModelName.RIDGE.value]

    validate_config(good)


@pytest.mark.parametrize(
    ("metrics", "pattern"),
    [
        (["rmse", "auc"], r"unsupported \['auc'\]"),
        (["rmse", "rmse"], "duplicates"),
        ([], "non-empty list"),
    ],
)
def test_invalid_metric_list_raises(good: dict[str, Any], metrics: list[str], pattern: str) -> None:
    """Metrics must be a non-empty, duplicate-free subset of ``Metric``."""
    good["validation"]["metrics"] = metrics
    good["models"]["selection_metric"] = const.Metric.RMSE.value

    with pytest.raises(ConfigError, match=pattern):
        validate_config(good)


def test_selection_metric_must_be_computed(good: dict[str, Any]) -> None:
    """Models cannot be selected on a metric that cross-validation does not compute."""
    good["validation"]["metrics"] = [const.Metric.RMSE.value]
    good["models"]["selection_metric"] = const.Metric.SPEARMAN.value

    with pytest.raises(ConfigError, match="selection_metric='spearman' is not in validation.metrics"):
        validate_config(good)


@pytest.mark.parametrize(
    "key",
    [
        "features.embedding_reduction.enabled",
        "models.refit_on_full_data",
        "models.save_fold_pipelines",
        "quality.fail_on_duplicate_ids",
    ],
)
@pytest.mark.parametrize("value", ["true", 1, None])
def test_flags_must_be_boolean(good: dict[str, Any], key: str, value: Any) -> None:
    """On/off settings accept only YAML true/false, never strings or numbers."""
    set_by_path(good, key, value)

    with pytest.raises(ConfigError, match=f"{key} must be true or false"):
        validate_config(good)
