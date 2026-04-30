from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def tool_threshold_sweep(
    run_dir: str,
    n_thresholds: int = 99,
    metrics: list[str] | None = None,
) -> dict:
    """Sweep the binary-classification decision threshold and report metric curves.

    Loads ``test_predictions.csv`` from a training run (always present after a
    binary run) and computes precision, recall, F1, MCC, and balanced accuracy
    at a grid of thresholds. Returns the full curves plus the argmax threshold
    for each metric, so an LLM agent can see the trade-off shape rather than a
    single calibrated number.

    Uses test-set predictions (not OOF probabilities) to stay compatible with
    refit-best predictors, which cannot produce OOF scores. The test set is
    held out from training, so this is still an unbiased threshold estimate.

    Parameters
    ----------
    run_dir : str
        Training run directory containing ``test_predictions.csv``.
    n_thresholds : int
        Number of thresholds to evaluate, spaced evenly in (0, 1). Default 99
        gives a 0.01 grid from 0.01 to 0.99.
    metrics : list[str] or None
        Which metrics to compute. Default: all of
        ``["f1", "precision", "recall", "mcc", "balanced_accuracy"]``.

    Returns
    -------
    dict with keys:
        thresholds : [float]                — grid points
        curves     : {metric: [float]}      — values per threshold
        best       : {metric: {"threshold", "value"}}
        hints      : [str]                  — observations (e.g., "F1-optimal
                                              threshold is within 0.01 of 0.5,
                                              calibration unlikely to help")

    Raises
    ------
    FileNotFoundError : if test_predictions.csv is missing.
    ValueError        : if the run is not binary classification (detected by
                        the prob_<class> columns present).
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    run_path = Path(run_dir)
    preds_path = run_path / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"tool_threshold_sweep: missing {preds_path}")

    preds = pd.read_csv(preds_path)
    prob_cols = sorted(c for c in preds.columns if c.startswith("prob_"))
    if len(prob_cols) != 2:
        raise ValueError(
            f"tool_threshold_sweep requires binary classification; found "
            f"{len(prob_cols)} prob_ columns in {preds_path.name}. Use "
            f"tool_read_analysis for non-binary runs."
        )
    if "actual" not in preds.columns:
        raise ValueError(
            f"tool_threshold_sweep: {preds_path.name} is missing 'actual' column; "
            "predictions were produced without ground truth."
        )

    # Positive class = highest-sorted label (same convention as
    # evaluate/classification.py)
    pos_prob_col = prob_cols[-1]  # e.g., "prob_1"
    pos_label_str = pos_prob_col.removeprefix("prob_")
    # Coerce the actual column to match the inferred positive label's dtype
    actual_raw = preds["actual"]
    try:
        pos_label: object = type(actual_raw.iloc[0])(pos_label_str)
    except (ValueError, TypeError):
        pos_label = pos_label_str
    y_true = (actual_raw == pos_label).astype(int).to_numpy()
    y_prob = preds[pos_prob_col].to_numpy(dtype=float)

    if metrics is None:
        metrics = ["f1", "precision", "recall", "mcc", "balanced_accuracy"]
    valid_metrics = {"f1", "precision", "recall", "mcc", "balanced_accuracy"}
    invalid = set(metrics) - valid_metrics
    if invalid:
        raise ValueError(f"Unknown metrics {sorted(invalid)}; valid: {sorted(valid_metrics)}")

    # Build threshold grid — exclude 0 and 1 because at those points many
    # metrics are undefined (division-by-zero) and the curves are uninformative
    thresholds = np.linspace(0, 1, n_thresholds + 2)[1:-1]

    curves: dict[str, list[float]] = {m: [] for m in metrics}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if "f1" in metrics:
            # F1 derived from precision and recall to avoid recomputing them twice
            # when both metrics are requested. Guarded for the degenerate case
            # where the predicted set is empty or all positives.
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            curves["f1"].append(float(f1))
            if "precision" in metrics:
                curves["precision"].append(float(p))
            if "recall" in metrics:
                curves["recall"].append(float(r))
        else:
            if "precision" in metrics:
                curves["precision"].append(float(precision_score(y_true, y_pred, zero_division=0)))
            if "recall" in metrics:
                curves["recall"].append(float(recall_score(y_true, y_pred, zero_division=0)))
        if "mcc" in metrics:
            curves["mcc"].append(float(matthews_corrcoef(y_true, y_pred)))
        if "balanced_accuracy" in metrics:
            curves["balanced_accuracy"].append(float(balanced_accuracy_score(y_true, y_pred)))

    # Argmax threshold per metric
    best = {}
    for m in metrics:
        vals = np.asarray(curves[m])
        idx = int(np.argmax(vals))
        best[m] = {
            "threshold": round(float(thresholds[idx]), 4),
            "value": round(float(vals[idx]), 6),
        }

    # Hints: call out when the optimum is near 0.5 (calibration is likely
    # a wash) or at the edges (class imbalance / probability miscalibration)
    hints: list[str] = []
    if "f1" in best:
        f1_thresh = best["f1"]["threshold"]
        if abs(f1_thresh - 0.5) < 0.02:
            hints.append(
                f"F1-optimal threshold is {f1_thresh:.2f}, within 0.02 of 0.5 — "
                "threshold calibration is unlikely to materially help; "
                "focus on data/feature work instead."
            )
        elif f1_thresh < 0.2 or f1_thresh > 0.8:
            hints.append(
                f"F1-optimal threshold is {f1_thresh:.2f}, far from 0.5 — "
                "probabilities are poorly calibrated or classes are highly imbalanced. "
                "Consider tool_calibration_curve for a calibration diagnostic."
            )
    # Per-metric disagreement check: precision-optimal vs recall-optimal should
    # rarely be at the same threshold for non-trivial problems; if they are,
    # the model is likely under-fitting.
    if "precision" in best and "recall" in best:
        gap = abs(best["precision"]["threshold"] - best["recall"]["threshold"])
        if gap < 0.05:
            hints.append(
                "Precision and recall peak at nearly the same threshold — the "
                "model has little threshold sensitivity (flat probabilities); "
                "calibration or a larger model may help."
            )

    return {
        "run_dir": run_dir,
        "thresholds": [round(float(t), 4) for t in thresholds],
        "curves": {m: [round(v, 6) for v in vals] for m, vals in curves.items()},
        "best": best,
        "hints": hints,
    }


def tool_calibration_curve(
    run_dir: str,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict:
    """Compute a reliability diagram for a binary classifier's predicted probabilities.

    Answers "when the model says 0.8, is it right 80% of the time?" by binning
    predictions and comparing the mean predicted probability in each bin to the
    observed positive rate. Reports the Expected Calibration Error (ECE) and
    the dominant miscalibration direction.

    Uses ``test_predictions.csv`` (held-out test probabilities) to stay
    compatible with refit-best predictors, which cannot produce OOF scores.
    The test set is unseen during training, so this is a valid calibration
    diagnostic.

    Parameters
    ----------
    run_dir : str
        Training run directory containing ``test_predictions.csv``.
    n_bins : int
        Number of bins for the reliability diagram (default 10).
    strategy : str
        "quantile" (default) — equal-frequency bins, one decile per bin.
        "uniform" — equal-width bins across [0, 1]. Quantile is preferred
        when probabilities cluster (e.g., most near 0 for rare events).

    Returns
    -------
    dict with keys:
        bins : [{
            bin_index       : int
            prob_range      : [float, float]  — bin edges
            n_samples       : int
            mean_predicted  : float           — average prob_1 in bin
            actual_positive_rate : float       — fraction of bin that is actually positive
            gap             : float           — mean_predicted - actual_positive_rate
        }]
        ece : float                           — sum(|gap_i| * n_i) / n
        max_gap : float                       — largest |gap| across bins
        direction : str                       — "over_confident", "under_confident",
                                                "well_calibrated", or "mixed"
        hints : [str]

    Raises
    ------
    FileNotFoundError : missing test_predictions.csv
    ValueError        : not binary, invalid strategy, or missing 'actual' column
    """
    run_path = Path(run_dir)
    preds_path = run_path / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"tool_calibration_curve: missing {preds_path}")
    if strategy not in {"quantile", "uniform"}:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")

    preds = pd.read_csv(preds_path)
    prob_cols = sorted(c for c in preds.columns if c.startswith("prob_"))
    if len(prob_cols) != 2:
        raise ValueError(
            f"tool_calibration_curve requires binary classification; found "
            f"{len(prob_cols)} prob_ columns in {preds_path.name}."
        )
    if "actual" not in preds.columns:
        raise ValueError(f"tool_calibration_curve: {preds_path.name} is missing 'actual' column.")

    pos_prob_col = prob_cols[-1]
    pos_label_str = pos_prob_col.removeprefix("prob_")
    actual_raw = preds["actual"]
    try:
        pos_label: object = type(actual_raw.iloc[0])(pos_label_str)
    except (ValueError, TypeError):
        pos_label = pos_label_str
    y_true = (actual_raw == pos_label).astype(int).to_numpy()
    y_prob = preds[pos_prob_col].to_numpy(dtype=float)
    n = len(y_true)

    # Bin edges
    if strategy == "quantile":
        # np.quantile with 0..1 gives n_bins+1 edges; dedupe (can collapse on
        # heavily-clustered probs)
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(y_prob, quantiles))
        # Ensure the first/last edges cover the full probability range so no
        # sample falls outside
        edges[0] = 0.0
        edges[-1] = 1.0
    else:  # uniform
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Assign each sample to a bin. np.digitize is right-exclusive by default;
    # clip indices into valid range (last bin absorbs samples at edge = 1.0).
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, len(edges) - 2)

    bins_out: list[dict] = []
    ece_weighted_sum = 0.0
    max_gap = 0.0
    # Proper definition: over-confidence means predicted probabilities are MORE
    # EXTREME (farther from 0.5) than the actual positive rates warrant. This
    # is separate from gap sign, which flips around 0.5. We track signed
    # "extremeness_error" per bin = (predicted - 0.5)² - (actual - 0.5)² so:
    #   > 0 : predicted is more extreme than actual (over-confident in this bin)
    #   < 0 : predicted is closer to 0.5 than actual (under-confident in this bin)
    extremeness_errors: list[float] = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            bins_out.append(
                {
                    "bin_index": b,
                    "prob_range": [round(float(edges[b]), 4), round(float(edges[b + 1]), 4)],
                    "n_samples": 0,
                    "mean_predicted": None,
                    "actual_positive_rate": None,
                    "gap": None,
                }
            )
            continue
        mean_pred = float(y_prob[mask].mean())
        actual_rate = float(y_true[mask].mean())
        gap = mean_pred - actual_rate
        ece_weighted_sum += abs(gap) * n_bin
        if abs(gap) > max_gap:
            max_gap = abs(gap)
        if n_bin >= 5 and abs(gap) > 0.02:  # Ignore near-zero gaps and tiny bins
            # Extremeness comparison: how far predicted vs actual are from 0.5
            extremeness_errors.append((mean_pred - 0.5) ** 2 - (actual_rate - 0.5) ** 2)
        bins_out.append(
            {
                "bin_index": b,
                "prob_range": [round(float(edges[b]), 4), round(float(edges[b + 1]), 4)],
                "n_samples": n_bin,
                "mean_predicted": round(mean_pred, 6),
                "actual_positive_rate": round(actual_rate, 6),
                "gap": round(gap, 6),
            }
        )

    ece = ece_weighted_sum / n

    # Direction classification: based on extremeness error, not gap sign.
    #   over_confident   : predicted probs are more extreme than actual rates
    #                      in most bins (e.g., 0.9 when truly 0.7, OR 0.1 when
    #                      truly 0.3 — both are over-confident)
    #   under_confident  : predicted probs are compressed toward 0.5 vs actuals
    #   well_calibrated  : gap small everywhere
    #   mixed            : no dominant direction across bins
    direction: str
    if max_gap < 0.05 or not extremeness_errors:
        direction = "well_calibrated"
    else:
        # Use the fraction of bins with same-sign extremeness error as the vote
        pos = sum(1 for e in extremeness_errors if e > 0)
        neg = sum(1 for e in extremeness_errors if e < 0)
        total = pos + neg
        if total == 0:
            direction = "well_calibrated"
        elif pos / total >= 0.7:
            direction = "over_confident"
        elif neg / total >= 0.7:
            direction = "under_confident"
        else:
            direction = "mixed"

    # Hints: prioritize the actionable interpretation
    hints: list[str] = []
    if direction == "over_confident":
        hints.append(
            f"Model is systematically over-confident (ECE={ece:.3f}). "
            "Probabilities above 0.5 are higher than actual positive rates. "
            "Consider temperature scaling or isotonic regression post-hoc."
        )
    elif direction == "under_confident":
        hints.append(
            f"Model is systematically under-confident (ECE={ece:.3f}). "
            "Probabilities are closer to 0.5 than they should be. "
            "Common for over-regularized models."
        )
    elif direction == "mixed":
        hints.append(
            f"Model has mixed miscalibration (ECE={ece:.3f}) — over-confident in "
            "some probability ranges and under-confident in others. Isotonic "
            "regression handles this better than temperature scaling."
        )
    elif direction == "well_calibrated":
        hints.append(
            f"Probabilities are well calibrated (ECE={ece:.3f}, max gap={max_gap:.3f}). "
            "Threshold calibration will not help; focus on score improvement."
        )

    # Flag extreme bins where model says very-high or very-low and is wrong
    for bin_info in bins_out:
        gap = bin_info["gap"]
        if gap is None:
            continue
        mean_p = bin_info["mean_predicted"]
        assert mean_p is not None  # narrowed by gap check above
        if mean_p > 0.9 and gap > 0.15:
            hints.append(
                f"High-confidence bucket (mean prob {mean_p:.2f}) has actual rate "
                f"{bin_info['actual_positive_rate']:.2f} — the model's most confident "
                "positives are {frac:.0%} wrong. Check for leakage or label noise.".format(frac=gap)
            )
        if mean_p < 0.1 and gap < -0.15:
            hints.append(
                f"Low-confidence bucket (mean prob {mean_p:.2f}) has actual rate "
                f"{bin_info['actual_positive_rate']:.2f} — the model misses more positives "
                "than its probabilities suggest in this range."
            )

    return {
        "run_dir": run_dir,
        "n_samples": int(n),
        "n_bins_effective": int(sum(1 for b in bins_out if b["n_samples"] > 0)),
        "strategy": strategy,
        "bins": bins_out,
        "ece": round(float(ece), 6),
        "max_gap": round(float(max_gap), 6),
        "direction": direction,
        "hints": hints,
    }


# ---------------------------------------------------------------------------
# tool_optuna_tune helpers — per-family search spaces
# ---------------------------------------------------------------------------


# Metrics where higher is better. The direction drives Optuna's study
# direction (maximize vs minimize) and also governs pruner comparisons.
