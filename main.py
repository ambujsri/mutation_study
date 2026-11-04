"""End-to-end pipeline orchestrator.

Runs the stages in dependency order, each of which is also runnable on its own
as ``python -m src.<stage>``:

    data_processing -> validation -> train -> evaluate -> predict -> ablation -> interpret

Every stage reads its inputs from, and writes its outputs to, the locations in
``config/config.yaml``; this module adds no behaviour of its own beyond
ordering, stage selection and a single log line per stage. That keeps the
orchestrator free of scientific decisions — it cannot change a result, only the
set of results produced.

Usage::

    python main.py --config config/config.yaml           # the whole pipeline
    python main.py --stages train evaluate               # just these, in canonical order
    python main.py --from evaluate                       # this stage and everything after
    python main.py --list                                # show the stage names

``--from`` is the one to reach for after changing a model: it skips the Phase 1
work that has not changed while still refreshing everything downstream.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.ablation import export_artifacts as export_ablation
from src.ablation import run_ablation
from src.common.config import DEFAULT_CONFIG_PATH, load_config, setup_logging
from src.data_processing import run_preprocessing
from src.evaluate import run_evaluation
from src.interpret import run_interpretation
from src.predict import run_prediction
from src.train import run_training
from src.validation import run_validation_harness

logger = logging.getLogger(__name__)


def _ablation_stage(cfg: dict[str, Any]) -> None:
    """Run the ablation study and write its table (the module keeps the two separate)."""
    export_ablation(run_ablation(cfg), cfg)


# Stage name -> entry point. Order is the dependency order: a stage may rely on
# the artifacts of any stage above it, never below.
STAGES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "data_processing": run_preprocessing,
    "validation": run_validation_harness,
    "train": run_training,
    "evaluate": run_evaluation,
    "predict": run_prediction,
    "ablation": _ablation_stage,
    "interpret": run_interpretation,
}


def resolve_stages(
    requested: list[str] | None, start_from: str | None, available: dict[str, Any] = STAGES
) -> list[str]:
    """Return the stages to run, always in canonical dependency order.

    Args:
        requested: Explicit stage names, or ``None`` for all of them.
        start_from: Run this stage and every stage after it.
        available: Stage registry, injected for testability.

    Raises:
        ValueError: If a name is unknown, or both selectors are given.
    """
    if requested and start_from:
        raise ValueError("--stages and --from are mutually exclusive")
    names = list(available)
    if start_from is not None:
        if start_from not in available:
            raise ValueError(f"Unknown stage '{start_from}'; choose from {names}")
        return names[names.index(start_from) :]
    if not requested:
        return names
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; choose from {names}")
    return [name for name in names if name in set(requested)]


def run_pipeline(cfg: dict[str, Any], stages: list[str]) -> dict[str, float]:
    """Execute ``stages`` in order, timing each one.

    Returns:
        Mapping of stage name to wall-clock seconds.
    """
    timings: dict[str, float] = {}
    for position, name in enumerate(stages, start=1):
        logger.info("=== stage %d/%d: %s ===", position, len(stages), name)
        started = time.perf_counter()
        STAGES[name](cfg)
        timings[name] = time.perf_counter() - started
        logger.info("--- stage %s finished in %.1fs ---", name, timings[name])
    logger.info(
        "Pipeline complete in %.1fs: %s",
        sum(timings.values()),
        ", ".join(f"{name}={seconds:.1f}s" for name, seconds in timings.items()),
    )
    return timings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--stages", nargs="+", metavar="STAGE", help="Run only these stages")
    parser.add_argument("--from", dest="start_from", metavar="STAGE", help="Run this stage onwards")
    parser.add_argument("--list", action="store_true", help="List the stage names and exit")
    return parser.parse_args()


def main() -> None:
    """Load the config and run the selected stages."""
    args = parse_args()
    if args.list:
        print("\n".join(STAGES))  # noqa: T201 - a CLI listing, not diagnostic logging
        return
    config = load_config(args.config)
    setup_logging(config)
    run_pipeline(config, resolve_stages(args.stages, args.start_from))


if __name__ == "__main__":
    main()
