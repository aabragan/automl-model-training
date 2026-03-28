"""Shared configuration defaults and logging setup."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_LABEL = "target"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_TIME_LIMIT = None  # no limit — train until all models are complete
# aliases: best_quality. Also: extreme, best_v150, high, high_v150, good, medium
DEFAULT_PRESET = "best"
DEFAULT_EVAL_METRIC = None  # auto-detect based on problem type
DEFAULT_PROBLEM_TYPE = None  # auto-detect from label column
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_PREDICTIONS_DIR = "predictions_output"

# Features to drop before training (edit as needed)
FEATURES_TO_DROP: list[str] = [
    # "feature_a",
    # "feature_b",
]

# Package-level logger name
PACKAGE_LOGGER = "automl_model_training"

logger = logging.getLogger(PACKAGE_LOGGER)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging for the package.

    - Default: INFO level, human-readable format
    - ``--verbose``: DEBUG level
    - ``--quiet``: WARNING level (errors and warnings only)
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root = logging.getLogger(PACKAGE_LOGGER)
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def make_run_dir(base_dir: str, prefix: str = "run") -> str:
    """Create and return a timestamped subdirectory under *base_dir*.

    Example: ``output/run_20260321_120530``
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory → %s", run_dir)
    return str(run_dir)
