from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from automl_model_training.predict import load_predictor


def tool_shap_interactions(run_dir: str, top_k: int = 5) -> dict:
    """Find pairs of features whose SHAP contributions correlate across rows.

    Uses the SHAP values already saved by ``tool_train(..., explain=True)``.
    Does NOT retrain anything.

    A pair of top-K features whose per-row SHAP contributions have high
    correlation suggests they carry redundant or coupled signal about the
    prediction. The LLM can use this to:
    - Drop one of a redundant pair (correlation ~ +1)
    - Engineer a ratio or product feature for a strongly-coupled pair
    - Investigate whether a counter-correlated pair (~ -1) indicates
      a hidden interaction the model is trying to express

    Parameters
    ----------
    run_dir : str
        Training run directory (must contain shap_values.csv — i.e.
        training must have used ``explain=True``).
    top_k : int
        Rank features by mean |SHAP| and analyze the top-k only. Default 5.
        Pairwise table size grows as k*(k-1)/2.

    Returns
    -------
    dict with keys:
        top_features : list of {feature, mean_abs_shap} sorted
        pairs        : list of {feature_a, feature_b, corr, abs_corr}
                       sorted by |corr| desc
        hints        : actionable observations
    """
    path = Path(run_dir)
    shap_path = path / "shap_values.csv"
    summary_path = path / "shap_summary.csv"
    if not shap_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"tool_shap_interactions: missing shap_values.csv or shap_summary.csv in "
            f"{run_dir}. Re-run training with explain=True."
        )

    shap_df = pd.read_csv(shap_path)
    summary = pd.read_csv(summary_path)

    # Top-k features by mean |SHAP|
    summary_sorted = summary.sort_values("mean_abs_shap", ascending=False).head(top_k)
    top_features = summary_sorted.to_dict(orient="records")
    top_feature_names = summary_sorted["feature"].tolist()

    # Restrict SHAP matrix to top-k features that actually exist as columns
    present = [f for f in top_feature_names if f in shap_df.columns]
    if len(present) < 2:
        return {
            "top_features": top_features,
            "pairs": [],
            "hints": ["Fewer than 2 top features found in shap_values.csv — no pairs to analyze"],
        }

    top_shap = shap_df[present]

    pairs = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            col_a = top_shap[a]
            col_b = top_shap[b]
            if col_a.std() == 0 or col_b.std() == 0:
                continue
            corr = float(col_a.corr(col_b))
            pairs.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "corr": round(corr, 4),
                    "abs_corr": round(abs(corr), 4),
                }
            )
    pairs.sort(key=lambda p: p["abs_corr"], reverse=True)

    hints = []
    for p in pairs:
        if p["abs_corr"] > 0.7:
            if p["corr"] > 0:
                hints.append(
                    f"'{p['feature_a']}' and '{p['feature_b']}' SHAP values are "
                    f"highly correlated (r={p['corr']}) — they may carry redundant "
                    "signal. Try dropping one, or engineer their ratio/product."
                )
            else:
                hints.append(
                    f"'{p['feature_a']}' and '{p['feature_b']}' SHAP values are "
                    f"strongly counter-correlated (r={p['corr']}) — consider "
                    "engineering their difference or ratio."
                )
    if not hints and pairs:
        hints.append("No strongly interacting pairs among top features (all |r| ≤ 0.7).")

    return {"top_features": top_features, "pairs": pairs, "hints": hints}


def tool_partial_dependence(
    run_dir: str,
    features: list[str] | None = None,
    n_values: int = 20,
    sample_size: int = 200,
    grid_strategy: str = "quantile",
    return_ice: bool = False,
) -> dict:
    r"""Compute partial-dependence curves for selected features.

    Answers "how does feature X affect predictions across its range?" —
    which SHAP importance cannot (SHAP shows magnitude, PDP shows shape).
    Uses the already-trained AutoGluon predictor from ``run_dir``.

    Math
    ----
    For a feature of interest :math:`x_S` and complement features
    :math:`x_C` (everything else), the partial-dependence function is

    .. math::

        \hat{f}_S(x_S)
            = \frac{1}{n} \sum_{i=1}^{n} \hat{f}(x_S,\; x_C^{(i)})

    i.e., for each grid value of the feature, copy the sample rows,
    overwrite the feature column with that value in every row, ask the
    predictor what each row would look like, and average the predictions.

    Implementation is a single batched predictor call per feature: build a
    DataFrame of shape ``(n_values × sample_size, n_features)`` where the
    first ``sample_size`` rows have the feature set to grid value 0, the
    next ``sample_size`` rows to grid value 1, and so on. One
    ``predict`` / ``predict_proba`` call returns all predictions; reshape
    into ``(n_values, sample_size)`` and average over axis 1.

    For classification, the curve is the mean predicted probability of
    the highest-sorted class (same convention as
    ``evaluate/classification.py``). For multiclass, per-class curves are
    also returned in ``per_class_pdp_values``.

    ICE (Individual Conditional Expectation) curves — per-row PDP before
    averaging — are returned in ``ice_values`` when ``return_ice=True``.
    Averaging can hide Simpson's-paradox-like effects where subgroups
    respond oppositely; ICE makes them visible.

    Parameters
    ----------
    run_dir : str
        Training run directory containing AutogluonModels/.
    features : list[str] or None
        Features to compute PDP for. If None, uses the top 5 from
        feature_importance.csv if available, else the first 5 columns.
    n_values : int
        Grid points across numeric feature range (default 20). Ignored
        for categorical features (all observed categories are used).
    sample_size : int
        Number of rows to average over (default 200). Lower = faster.
    grid_strategy : str
        How to place numeric grid points. "quantile" (default) spaces
        points by data density — more resolution where the data lives.
        "linspace" spaces them evenly across the min/max range. Ignored
        for categorical features.
    return_ice : bool
        If True, include per-row ICE curves in the output as
        ``ice_values`` (shape ``n_values × sample_size``). Default False.

    Returns
    -------
    dict with keys:
        feature_curves : list of dicts, one per feature, with:
            feature              : column name
            is_numeric           : bool
            grid_values          : grid points (floats or strings)
            pdp_values           : averaged predictions per grid point
            pdp_std              : std across sample rows per grid point
            per_class_pdp_values : {class_label: [values]} — multiclass only
            ice_values           : n_values × sample_size matrix — if return_ice
        hints          : observations (monotonicity, threshold effects)
    """
    run_path = Path(run_dir)
    model_dir = run_path / "AutogluonModels"
    test_path = run_path / "test_raw.csv"
    if not model_dir.exists():
        raise FileNotFoundError(f"tool_partial_dependence: missing {model_dir}")
    if not test_path.exists():
        raise FileNotFoundError(f"tool_partial_dependence: missing {test_path}")
    if grid_strategy not in {"quantile", "linspace"}:
        raise ValueError(f"grid_strategy must be 'quantile' or 'linspace', got {grid_strategy!r}")

    predictor = load_predictor(str(model_dir))
    test_data = pd.read_csv(test_path)
    label = predictor.label
    feature_cols = [c for c in test_data.columns if c != label]

    # Pick features to analyze
    if features is None:
        imp_path = run_path / "feature_importance.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path, index_col=0)
            if "importance" in imp.columns:
                ranked = imp.sort_values("importance", ascending=False).index.tolist()
                features = [f for f in ranked if f in feature_cols][:5]
        if not features:
            features = feature_cols[:5]
    missing = [f for f in features if f not in feature_cols]
    if missing:
        raise ValueError(f"Features not in test data: {missing}")

    # Subsample rows (PDP averages predictions → more rows = smoother, slower)
    sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
    sample_x = sample[feature_cols].reset_index(drop=True)
    n_samples = len(sample_x)

    # Memory guard: batched prediction materializes a (n_values * n_samples) DataFrame.
    # Cap at 100k rows and fall back to the per-grid-value loop above that threshold.
    MAX_BATCH_ROWS = 100_000

    problem_type = predictor.problem_type
    is_classification = problem_type in ("binary", "multiclass")

    feature_curves = []
    for feat in features:
        series = test_data[feat]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # --- Grid construction -------------------------------------------------
        if is_numeric:
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= n_values:
                # Feature has few distinct values — use them directly, no interpolation
                grid = np.sort(unique_vals).tolist()
            elif grid_strategy == "quantile":
                quantiles = np.linspace(0, 1, n_values)
                grid_arr = np.quantile(series.dropna(), quantiles)
                # Quantile grid can produce duplicates on features with tied values;
                # dedupe and pad with linspace points to reach n_values when needed
                deduped = np.unique(grid_arr)
                if len(deduped) < n_values:
                    deduped = np.unique(
                        np.concatenate([deduped, np.linspace(series.min(), series.max(), n_values)])
                    )
                grid = deduped.tolist()
            else:  # linspace
                grid = np.linspace(series.min(), series.max(), n_values).tolist()
        else:
            grid = series.value_counts().head(n_values).index.tolist()

        # --- Batched prediction ------------------------------------------------
        # Build the (|grid| * n_samples, n_features) perturbed DataFrame: for each
        # grid value v, copy the sample rows and set the feature to v.
        total_rows = len(grid) * n_samples
        if total_rows <= MAX_BATCH_ROWS:
            # Tile sample rows |grid| times (keeps the original dtypes of the other cols)
            batched = pd.concat([sample_x] * len(grid), ignore_index=True)
            # Build the perturbed feature column with the right dtype. For numeric
            # features, preserve the original int dtype when every grid value is an
            # integer; otherwise let it promote (e.g., int → float for non-integer grids).
            # Extension dtypes (e.g., nullable Int64) are left as object and pandas
            # will coerce on assignment.
            grid_col = np.repeat(np.array(grid, dtype=object), n_samples)
            np_dtype = series.dtype if isinstance(series.dtype, np.dtype) else None
            if (
                is_numeric
                and np_dtype is not None
                and np.issubdtype(np_dtype, np.integer)
                and all(float(g).is_integer() for g in grid)
            ):
                grid_col = grid_col.astype(np_dtype)
            batched[feat] = grid_col

            if is_classification:
                proba = predictor.predict_proba(batched)
                # Reshape to (n_values, n_samples, n_classes). Force float dtype:
                # predict_proba can return object columns, which breaks .std().
                pred_matrix = np.asarray(proba.values, dtype=float).reshape(
                    len(grid), n_samples, -1
                )
                class_labels = list(proba.columns)
            else:
                preds = predictor.predict(batched)
                pred_matrix = np.asarray(preds, dtype=float).reshape(len(grid), n_samples)
                class_labels = None
        else:
            # Fallback: per-grid loop (legacy path). Triggered on very large
            # n_values * sample_size to keep memory bounded.
            if is_classification:
                per_value = []
                for v in grid:
                    p = sample_x.copy()
                    p[feat] = v
                    per_value.append(np.asarray(predictor.predict_proba(p).values, dtype=float))
                pred_matrix = np.stack(per_value, axis=0)  # (n_values, n_samples, n_classes)
                class_labels = list(predictor.predict_proba(sample_x.head(1)).columns)
            else:
                per_value = []
                for v in grid:
                    p = sample_x.copy()
                    p[feat] = v
                    per_value.append(np.asarray(predictor.predict(p), dtype=float))
                pred_matrix = np.stack(per_value, axis=0)  # (n_values, n_samples)
                class_labels = None

        # --- Aggregate into PDP / ICE -----------------------------------------
        curve: dict = {
            "feature": feat,
            "is_numeric": bool(is_numeric),
            "grid_values": [float(g) if is_numeric else str(g) for g in grid],
        }

        if is_classification and class_labels is not None:
            # Overall curve uses the highest-sorted class (consistent with
            # evaluate/classification.py::pos_label = labels[-1])
            pos_label = sorted(class_labels)[-1]
            pos_idx = class_labels.index(pos_label)
            pos_slice = pred_matrix[:, :, pos_idx]  # (n_values, n_samples)
            curve["pdp_values"] = [round(float(v), 6) for v in pos_slice.mean(axis=1)]
            curve["pdp_std"] = [round(float(v), 6) for v in pos_slice.std(axis=1)]
            # Per-class curves for multiclass; for binary this is just the two
            # complementary curves (cheap, still useful for threshold reasoning)
            curve["per_class_pdp_values"] = {
                str(cls): [round(float(v), 6) for v in pred_matrix[:, :, i].mean(axis=1)]
                for i, cls in enumerate(class_labels)
            }
            if return_ice:
                curve["ice_values"] = pos_slice.round(6).tolist()
        else:
            curve["pdp_values"] = [round(float(v), 6) for v in pred_matrix.mean(axis=1)]
            curve["pdp_std"] = [round(float(v), 6) for v in pred_matrix.std(axis=1)]
            if return_ice:
                curve["ice_values"] = pred_matrix.round(6).tolist()

        feature_curves.append(curve)

    # --- Hints ----------------------------------------------------------------
    hints = []
    for curve in feature_curves:
        raw_vals = curve["pdp_values"]
        assert isinstance(raw_vals, list)
        vals: list[float] = [float(v) for v in raw_vals]
        if len(vals) < 3:
            continue
        if curve["is_numeric"]:
            diffs = np.diff(np.array(vals))
            if np.all(diffs >= 0):
                hints.append(f"'{curve['feature']}' PDP is monotonically increasing.")
            elif np.all(diffs <= 0):
                hints.append(f"'{curve['feature']}' PDP is monotonically decreasing.")
            else:
                span = max(vals) - min(vals)
                if span > 0.05:
                    hints.append(
                        f"'{curve['feature']}' PDP is non-monotonic — the model learned "
                        "a threshold or peak effect; consider binning or polynomial terms."
                    )

    return {"feature_curves": feature_curves, "hints": hints}


def tool_partial_dependence_2way(
    run_dir: str,
    feature_a: str,
    feature_b: str,
    n_values_a: int = 10,
    n_values_b: int = 10,
    sample_size: int = 200,
    grid_strategy: str = "quantile",
    max_cells: int = 50_000,
) -> dict:
    """Compute a 2-way partial-dependence surface for two features.

    Extends ``tool_partial_dependence`` to an interaction view: instead of
    showing how predictions shift when ONE feature varies, this shows the
    response surface when TWO features vary jointly. Answers "do these two
    features interact, or do they each act independently?" — which SHAP
    interaction correlation cannot (that gives a magnitude only).

    Implementation mirrors the 1D tool: build a single perturbed
    DataFrame of shape ``(n_values_a * n_values_b * sample_size, n_features)``,
    one ``predict``/``predict_proba`` call, reshape into
    ``(n_values_a, n_values_b, sample_size)`` and average over the sample axis.

    Cost guard: the total prediction rows are capped at ``max_cells``
    (default 50,000). If ``n_values_a * n_values_b * sample_size`` exceeds
    the cap, the function raises ``ValueError`` with a specific reduction
    recommendation — no silent sample-size shrinking.

    Parameters
    ----------
    run_dir : str
        Training run directory containing ``AutogluonModels`` and
        ``test_raw.csv``.
    feature_a, feature_b : str
        Two columns to sweep. Must both be present in the test data and
        must differ from each other.
    n_values_a, n_values_b : int
        Grid resolution per feature. Default 10 x 10.
    sample_size : int
        Number of background rows to average over per cell. Default 200.
    grid_strategy : str
        "quantile" (default) or "linspace" — applies to numeric features
        only. Same semantics as ``tool_partial_dependence``.
    max_cells : int
        Cap on the total (n_values_a * n_values_b * sample_size). Raises
        if exceeded. Default 50,000 which comfortably fits in memory.

    Returns
    -------
    dict with keys:
        feature_a, feature_b            : column names
        is_numeric_a, is_numeric_b      : bool
        grid_a, grid_b                  : grid values (floats or strings)
        surface : n_values_a x n_values_b matrix — averaged predictions
        surface_std : n_values_a x n_values_b — std across sample rows
        interaction_strength : float — how much the surface deviates from
                                additivity (pure additive surface = 0.0;
                                pure multiplicative/threshold = larger).
                                Computed as the std of residuals after
                                fitting best-additive decomposition.
        shape_hint : str — "additive", "synergy", "saddle", "threshold"
        hints : [str]

    Raises
    ------
    FileNotFoundError : missing model or test_raw.csv
    ValueError        : features not in test data, features equal, invalid
                        grid_strategy, cost cap exceeded
    """
    run_path = Path(run_dir)
    model_dir = run_path / "AutogluonModels"
    test_path = run_path / "test_raw.csv"
    if not model_dir.exists():
        raise FileNotFoundError(f"tool_partial_dependence_2way: missing {model_dir}")
    if not test_path.exists():
        raise FileNotFoundError(f"tool_partial_dependence_2way: missing {test_path}")
    if grid_strategy not in {"quantile", "linspace"}:
        raise ValueError(f"grid_strategy must be 'quantile' or 'linspace', got {grid_strategy!r}")
    if feature_a == feature_b:
        raise ValueError("feature_a and feature_b must differ")

    predictor = load_predictor(str(model_dir))
    test_data = pd.read_csv(test_path)
    label = predictor.label
    feature_cols = [c for c in test_data.columns if c != label]

    for feat in (feature_a, feature_b):
        if feat not in test_data.columns:
            raise ValueError(f"Feature {feat!r} not in test data. Available: {feature_cols}")

    sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
    sample_x = sample[feature_cols].reset_index(drop=True)
    n_samples = len(sample_x)

    # Cost cap check — fail fast with a clear message rather than OOM
    total_rows = n_values_a * n_values_b * n_samples
    if total_rows > max_cells:
        recommended_na = min(n_values_a, int(np.sqrt(max_cells / n_samples)))
        recommended_nb = min(n_values_b, int(np.sqrt(max_cells / n_samples)))
        raise ValueError(
            f"2-way PDP would materialize {total_rows:,} prediction rows, exceeding "
            f"the max_cells={max_cells:,} cap. Reduce n_values_a and n_values_b to "
            f"≈{recommended_na}x{recommended_nb}, or lower sample_size from "
            f"{n_samples}, or raise max_cells if memory allows."
        )

    # Build the two grids using the same logic as the 1D tool
    def _build_grid(feat: str, n_values: int) -> tuple[list, bool]:
        series = test_data[feat]
        is_num = pd.api.types.is_numeric_dtype(series)
        if is_num:
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= n_values:
                grid = np.sort(unique_vals).tolist()
            elif grid_strategy == "quantile":
                quantiles = np.linspace(0, 1, n_values)
                grid_arr = np.quantile(series.dropna(), quantiles)
                deduped = np.unique(grid_arr)
                if len(deduped) < n_values:
                    deduped = np.unique(
                        np.concatenate([deduped, np.linspace(series.min(), series.max(), n_values)])
                    )
                grid = deduped.tolist()
            else:  # linspace
                grid = np.linspace(series.min(), series.max(), n_values).tolist()
        else:
            grid = series.value_counts().head(n_values).index.tolist()
        return grid, is_num

    grid_a, is_num_a = _build_grid(feature_a, n_values_a)
    grid_b, is_num_b = _build_grid(feature_b, n_values_b)
    na = len(grid_a)
    nb = len(grid_b)

    # Build the (na * nb * n_samples, n_features) perturbed DataFrame. Row
    # layout: [a0*b0*all_samples, a0*b1*all_samples, ..., a1*b0*all_samples, ...]
    batched = pd.concat([sample_x] * (na * nb), ignore_index=True)
    # np.repeat over a broadcasted outer product gives the column values in the
    # same order. For a with na values, b with nb values, n_samples rows each:
    #   a values repeat nb*n_samples times each
    #   b values repeat n_samples times each (within each a block)
    a_col = np.repeat(np.array(grid_a, dtype=object), nb * n_samples)
    b_col = np.tile(np.repeat(np.array(grid_b, dtype=object), n_samples), na)
    batched[feature_a] = a_col
    batched[feature_b] = b_col

    # Preserve int dtypes when every grid value is integer (same logic as 1D tool)
    for feat, grid_vals in [(feature_a, grid_a), (feature_b, grid_b)]:
        series = test_data[feat]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        np_dtype = series.dtype if isinstance(series.dtype, np.dtype) else None
        if (
            np_dtype is not None
            and np.issubdtype(np_dtype, np.integer)
            and all(float(g).is_integer() for g in grid_vals)
        ):
            batched[feat] = batched[feat].astype(np_dtype)

    problem_type = predictor.problem_type
    is_classification = problem_type in ("binary", "multiclass")
    if is_classification:
        proba = predictor.predict_proba(batched)
        pos_label = sorted(proba.columns)[-1]  # convention matches evaluate/classification.py
        preds = np.asarray(proba[pos_label].values, dtype=float)
    else:
        preds = np.asarray(predictor.predict(batched), dtype=float)

    # Reshape into (na, nb, n_samples) and collapse the sample axis
    pred_cube = preds.reshape(na, nb, n_samples)
    surface = pred_cube.mean(axis=2)
    surface_std = pred_cube.std(axis=2)

    # --- Interaction strength ------------------------------------------------
    # Decompose the surface into an additive baseline:
    #   f_add(a, b) ≈ row_mean(a) + col_mean(b) - grand_mean
    # The residual std measures how much the interaction departs from
    # additivity. 0 = purely additive (no interaction); larger = stronger
    # interaction effect.
    row_means = surface.mean(axis=1, keepdims=True)
    col_means = surface.mean(axis=0, keepdims=True)
    grand_mean = surface.mean()
    additive_approx = row_means + col_means - grand_mean
    residual = surface - additive_approx
    interaction_strength = float(residual.std())

    # Shape classification based on the residual pattern and surface range
    # - "additive" when residual std is negligible vs surface span
    # - "synergy" when residuals are all positive or all negative (monotone
    #   reinforcement/cancellation)
    # - "saddle" when residuals have both signs with similar magnitudes
    # - "threshold" when the surface has a sharp step in one direction
    surface_span = float(surface.max() - surface.min())
    if surface_span == 0.0 or interaction_strength / max(surface_span, 1e-12) < 0.05:
        shape_hint = "additive"
    else:
        pos_frac = float((residual > 0).mean())
        if pos_frac > 0.85 or pos_frac < 0.15:
            shape_hint = "synergy"
        else:
            # Distinguish saddle vs threshold by checking whether the surface
            # has a sharp monotonic jump along either axis
            if is_num_a:
                jumps_a = np.abs(np.diff(surface, axis=0))
                max_jump_a = float(jumps_a.max()) if jumps_a.size else 0.0
            else:
                max_jump_a = 0.0
            if is_num_b:
                jumps_b = np.abs(np.diff(surface, axis=1))
                max_jump_b = float(jumps_b.max()) if jumps_b.size else 0.0
            else:
                max_jump_b = 0.0
            if max(max_jump_a, max_jump_b) > 0.5 * surface_span:
                shape_hint = "threshold"
            else:
                shape_hint = "saddle"

    hints: list[str] = []
    if shape_hint == "additive":
        hints.append(
            f"{feature_a!r} and {feature_b!r} act additively (no interaction). "
            "SHAP importance and 1D PDPs already capture their effects."
        )
    elif shape_hint == "synergy":
        hints.append(
            f"{feature_a!r} and {feature_b!r} interact synergistically — their "
            "joint effect is consistently stronger (or weaker) than the sum of "
            "their individual effects. Consider an explicit interaction term."
        )
    elif shape_hint == "saddle":
        hints.append(
            f"{feature_a!r} and {feature_b!r} interact non-monotonically "
            "(saddle-shaped response). Linear interaction terms won't capture "
            "this; tree-based models or polynomial features may be needed."
        )
    elif shape_hint == "threshold":
        hints.append(
            f"{feature_a!r} or {feature_b!r} has a threshold effect that depends "
            "on the other feature's value. Consider binning the threshold feature."
        )

    return {
        "run_dir": run_dir,
        "feature_a": feature_a,
        "feature_b": feature_b,
        "is_numeric_a": bool(is_num_a),
        "is_numeric_b": bool(is_num_b),
        "grid_a": [float(g) if is_num_a else str(g) for g in grid_a],
        "grid_b": [float(g) if is_num_b else str(g) for g in grid_b],
        "surface": [[round(float(v), 6) for v in row] for row in surface],
        "surface_std": [[round(float(v), 6) for v in row] for row in surface_std],
        "interaction_strength": round(interaction_strength, 6),
        "shape_hint": shape_hint,
        "hints": hints,
    }
