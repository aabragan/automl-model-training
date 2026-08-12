"""Shared configuration defaults and logging setup."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# setdefault so users can override via shell if needed
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

DEFAULT_LABEL = "target"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_TIME_LIMIT = None  # no limit — train until all models are complete
# alias: best_quality. Also: extreme and noncommercial (GPU foundation models,
# require the matching extra), best_v150, high, high_v150, good, medium, extreme_v150
DEFAULT_PRESET = "best"
DEFAULT_EVAL_METRIC = None  # auto-detect based on problem type
DEFAULT_PROBLEM_TYPE = None  # auto-detect from label column
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_PREDICTIONS_DIR = "predictions_output"

# Analysis thresholds
CLASSIFICATION_CARDINALITY_THRESHOLD = 20  # max unique values to treat label as classification
CORRELATION_THRESHOLD = 0.90  # |r| above which feature pairs are flagged
LOW_IMPORTANCE_THRESHOLD = 0.001  # permutation importance at or below this is "near-zero"
OVERFITTING_SEVERE_GAP_PCT = 10.0  # val/test gap % triggering a strong warning
OVERFITTING_MODERATE_GAP_PCT = 5.0  # val/test gap % triggering a mild warning

# Regression diagnostics thresholds (used by evaluate.analyze)
# |t| = |mean residual| / (std residual / sqrt(n)) above this = systematic bias.
# 2.0 ≈ the two-sided 5% critical value of the t distribution at large n.
RESIDUAL_BIAS_T_THRESHOLD = 2.0
HETEROSCEDASTICITY_CORR_THRESHOLD = 0.3  # |corr(actual, |residual|)| above this
REGRESSION_LOW_R2_THRESHOLD = 0.3  # R² below this triggers a weak-fit warning
TARGET_SKEW_THRESHOLD = 2.0  # |skew| of the training target above this suggests log transform


@dataclass(frozen=True)
class RegressionThresholds:
    """Per-run overrides for the regression diagnostics in ``evaluate.analyze``.

    Defaults come from the module constants above. Override per run when the
    package-wide defaults are miscalibrated for a specific target — e.g. a
    hard-ceiling target where the achievable R² is inherently low (market
    residuals, noisy human behavior), so the default weak-fit warning would
    point the agent at feature engineering that cannot pay off.
    """

    residual_bias_t_threshold: float = RESIDUAL_BIAS_T_THRESHOLD
    heteroscedasticity_corr_threshold: float = HETEROSCEDASTICITY_CORR_THRESHOLD
    low_r2_threshold: float = REGRESSION_LOW_R2_THRESHOLD
    target_skew_threshold: float = TARGET_SKEW_THRESHOLD


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

    # Scope to package logger so we don't affect third-party library output
    root = logging.getLogger(PACKAGE_LOGGER)
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def make_run_dir(base_dir: str, prefix: str = "run") -> str:
    """Create and return a timestamped subdirectory under *base_dir*.

    Example: ``output/run_20260321_120530_123456``
    """
    # Microsecond precision prevents collisions when two runs start within the same second
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory → %s", run_dir)
    return str(run_dir)
