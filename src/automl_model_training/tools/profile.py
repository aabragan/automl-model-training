from __future__ import annotations

import pandas as pd

from automl_model_training.config import CLASSIFICATION_CARDINALITY_THRESHOLD
from automl_model_training.profile import compute_correlation_matrix, recommend_features_to_drop


def tool_profile(csv_path: str, label: str) -> dict:
    """Analyze a dataset before training.

    Always call this first. Returns shape, label distribution, missing value
    percentages, and correlated feature drop recommendations.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV.
    label : str
        Target column name.

    Returns
    -------
    dict with keys:
        shape               : [n_rows, n_cols]
        label_distribution  : {class: count} — use to detect class imbalance
        missing_pct         : {column: pct_missing} — flag columns >20% missing
        drop_recommendations: list of {feature, reason, correlated_with}
        numeric_features    : list of numeric column names
        categorical_features: list of categorical column names
    """
    data = pd.read_csv(csv_path)
    corr = compute_correlation_matrix(data, label)
    recs = recommend_features_to_drop(corr, label)
    numeric = data.select_dtypes(include="number").columns.tolist()
    categorical = data.select_dtypes(exclude="number").columns.tolist()
    return {
        "shape": list(data.shape),
        "label_distribution": data[label].value_counts().to_dict(),
        "missing_pct": (data.isnull().mean() * 100).round(2).to_dict(),
        "drop_recommendations": recs,
        "numeric_features": [c for c in numeric if c != label],
        "categorical_features": [c for c in categorical if c != label],
    }


def tool_detect_leakage(
    csv_path: str,
    label: str,
    threshold: float = 0.95,
    sample_size: int = 5000,
    seed: int = 42,
) -> dict:
    """Detect features that are suspiciously predictive of the target.

    Trains a depth-3 decision tree on each feature individually and scores
    it against the target. Any feature that alone achieves a score above
    ``threshold`` is almost certainly leaking — either a direct copy of the
    target, a derived variant (e.g., log of target), or a proxy computed
    from the target after the fact (e.g., "outcome_category" derived from
    the outcome column).

    Runs in seconds — use this BEFORE any AutoGluon training run to avoid
    wasting time optimizing a leaky model.

    Problem type is auto-detected using the same convention as the training
    pipeline: ≤20 unique label values → classification (tree classifier +
    accuracy), more → regression (tree regressor + R²).

    Parameters
    ----------
    csv_path : str
        Path to the input CSV.
    label : str
        Target column name.
    threshold : float
        Score above which a feature is flagged as leaking (default 0.95).
        Set lower (e.g., 0.85) to catch near-leaks; higher to reduce
        false positives on legitimately strong features.
    sample_size : int
        Number of rows to subsample for the leakage test (default 5000).
        More rows doesn't change the signal — single-feature trees saturate
        quickly.
    seed : int
        Random seed for subsampling and tree training (default 42).

    Returns
    -------
    dict with keys:
        problem_type    : "classification" | "regression"
        label           : target column name
        threshold       : score threshold used
        suspected_leaks : list of {feature, score, reason} sorted worst-first
        all_scores      : list of {feature, score} for every feature
        hints           : list of actionable observations
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    df = pd.read_csv(csv_path)
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not in CSV: {list(df.columns)}")

    # Auto-detect problem type using the same rule as data.load_and_prepare
    is_classification = df[label].nunique() <= CLASSIFICATION_CARDINALITY_THRESHOLD
    problem_type = "classification" if is_classification else "regression"

    # Subsample for speed; single-feature trees saturate long before 5000 rows
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    y = df[label]
    # Drop rows where target is NaN — tree models can't score those
    mask = y.notna()
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != label]
    scores: list[dict] = []

    for col in feature_cols:
        x = df[col]
        # Skip columns that are entirely NaN (no signal to measure)
        if x.isna().all():
            continue

        # Encode non-numeric columns via dense integer codes. Trees don't need
        # one-hot for this test — we're measuring dependence, not accuracy.
        if x.dtype == object or isinstance(x.dtype, pd.CategoricalDtype):
            codes = pd.Categorical(x).codes.astype(float)
            x_encoded: pd.Series = pd.Series(codes).replace(-1, pd.NA)
        else:
            x_encoded = x.astype(float)

        # Drop rows where this feature is NaN
        feat_mask = x_encoded.notna()
        if feat_mask.sum() < 10:
            continue
        x_clean = x_encoded.loc[feat_mask].to_numpy().reshape(-1, 1)
        y_clean = y.loc[feat_mask]

        if is_classification:
            clf = DecisionTreeClassifier(max_depth=3, random_state=seed)
            # Use balanced_accuracy so a feature that merely predicts the
            # majority class (trivially high raw accuracy on imbalanced data)
            # scores near 0.5, not 1.0. Genuine leaks still score near 1.0.
            # 3-fold CV prevents the tree from memorizing the training set.
            cv_scores = cross_val_score(clf, x_clean, y_clean, cv=3, scoring="balanced_accuracy")
            score = float(cv_scores.mean())
        else:
            reg = DecisionTreeRegressor(max_depth=3, random_state=seed)
            cv_scores = cross_val_score(reg, x_clean, y_clean, cv=3, scoring="r2")
            score = float(cv_scores.mean())

        scores.append({"feature": col, "score": round(score, 4)})

    scores.sort(key=lambda s: s["score"], reverse=True)

    suspected = []
    for entry in scores:
        if entry["score"] >= threshold:
            if entry["score"] >= 0.999:
                reason = (
                    "Perfect single-feature predictor — either a direct copy of "
                    "the target, a derived variant (e.g., log(target)), or a "
                    "feature computed after the target was known. If this is real "
                    "data, it's leakage. If this is synthetic/demo data, it may "
                    "be an unrealistically clean signal."
                )
            elif entry["score"] >= 0.98:
                reason = (
                    "Near-perfect single-feature score — likely a derived proxy "
                    "of the target or a post-hoc calculation that shouldn't be "
                    "available at inference time."
                )
            else:
                reason = (
                    "Single-feature score exceeds threshold — investigate how "
                    "this feature was computed and whether it's available at "
                    "inference time."
                )
            suspected.append({**entry, "reason": reason})

    hints = []
    if suspected:
        hints.append(
            f"{len(suspected)} feature(s) individually predict the target above the "
            f"{threshold:.2f} threshold — REMOVE these before training or you will "
            f"train a leaky model."
        )
        # Callers often want to pass the list to tool_train's drop parameter directly
        hints.append(f"Suggested drop list: {[s['feature'] for s in suspected]}")
    elif scores and scores[0]["score"] > 0.85:
        hints.append(
            f"No leaks above {threshold:.2f}, but '{scores[0]['feature']}' scores "
            f"{scores[0]['score']:.2f} on its own — verify it's a legitimate signal "
            f"and not a subtle proxy for the target."
        )

    return {
        "problem_type": problem_type,
        "label": label,
        "threshold": threshold,
        "suspected_leaks": suspected,
        "all_scores": scores,
        "hints": hints,
    }


def tool_deep_profile(csv_path: str, label: str) -> dict:
    """Extended per-feature profiling that maps signals to feature engineering actions.

    Use after ``tool_profile`` (which gives shape/missing/correlations) when
    you want to see per-feature skewness, outliers, and cardinality with
    direct ``tool_engineer_features`` spec suggestions. This is richer than
    ``tool_profile`` — call it when you plan to engineer features.

    Parameters
    ----------
    csv_path : str
    label : str

    Returns
    -------
    dict with keys:
        numeric_features : list of {feature, skew, outlier_pct, recommendation}
        categorical_features : list of {feature, cardinality, top_value, recommendation}
        suggested_transforms : dict ready to pass to tool_engineer_features, e.g.
            {"log": ["price"], "onehot": ["category"], "bin": {}}
        hints : list of summary observations
    """
    from automl_model_training.profile import (
        profile_categorical_features,
        profile_numeric_features,
    )

    df = pd.read_csv(csv_path)
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not in CSV: {list(df.columns)}")

    numeric_stats = profile_numeric_features(df, label)
    categorical_stats = profile_categorical_features(df, label)

    numeric_features = []
    suggested_log: list[str] = []
    for col_idx, row in numeric_stats.iterrows():
        col = str(col_idx)
        if col == label:
            continue
        skew = float(row.get("skew", 0))
        outlier_pct = float(row.get("outlier_pct", 0))
        # Skew > 1 (or < -1) is moderately-to-highly skewed; log1p typically helps
        # when all values are non-negative. Only recommend log for positive skew.
        recommend = []
        if skew > 1.0 and (df[col] >= 0).all():
            recommend.append("log transform (positive skew, non-negative values)")
            suggested_log.append(col)
        elif skew < -1.0:
            recommend.append("negative skew — consider reflecting + log, or keep as-is")
        if outlier_pct > 5.0:
            recommend.append(f"{outlier_pct:.1f}% outliers — tree models handle these fine")
        numeric_features.append(
            {
                "feature": col,
                "skew": round(skew, 3),
                "outlier_pct": round(outlier_pct, 2),
                "recommendation": "; ".join(recommend) if recommend else "no action",
            }
        )

    categorical_features = []
    suggested_onehot: list[str] = []
    for cat_idx, row in categorical_stats.iterrows():
        col = str(cat_idx)
        if col == label:
            continue
        cardinality = int(row.get("nunique", 0))
        top_value = row.get("top_value", "")
        recommend = []
        if cardinality == 2:
            recommend.append("binary categorical — one-hot is safe")
            suggested_onehot.append(str(col))
        elif 2 < cardinality <= 20:
            recommend.append(f"low cardinality ({cardinality}) — one-hot encode")
            suggested_onehot.append(str(col))
        elif 20 < cardinality <= 100:
            recommend.append(
                f"medium cardinality ({cardinality}) — one-hot with _other bucket or target encode"
            )
        else:
            recommend.append(
                f"high cardinality ({cardinality}) — consider target_mean encoding or drop"
            )
        categorical_features.append(
            {
                "feature": col,
                "cardinality": cardinality,
                "top_value": str(top_value),
                "recommendation": "; ".join(recommend),
            }
        )

    suggested_transforms: dict = {}
    if suggested_log:
        suggested_transforms["log"] = suggested_log
    if suggested_onehot:
        suggested_transforms["onehot"] = suggested_onehot

    hints = []
    if suggested_log:
        hints.append(
            f"{len(suggested_log)} numeric feature(s) are right-skewed — "
            f"pass suggested_transforms to tool_engineer_features for log1p transforms."
        )
    if any(int(f["cardinality"]) > 100 for f in categorical_features):  # type: ignore[call-overload]
        hints.append(
            "High-cardinality categorical(s) present — one-hot would explode columns. "
            "Use target_mean encoding or drop instead."
        )

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "suggested_transforms": suggested_transforms,
        "hints": hints,
    }
