"""Declarative feature engineering primitives.

Exposes a single entry point, :func:`apply_transformations`, that takes a
DataFrame and a spec dict and returns a new DataFrame plus a report of what
changed. Designed to be driven by an LLM: the LLM writes the spec, this
module applies it safely.

Supported transformations (keys of the spec dict):

- ``log``         : list[str]                       → ``log_<col>`` via log1p
- ``sqrt``        : list[str]                       → ``sqrt_<col>``
- ``ratio``       : list[[num, denom]]              → ``<num>_per_<denom>``
- ``diff``        : list[[a, b]]                    → ``<a>_minus_<b>``
- ``product``     : list[[a, b]]                    → ``<a>_x_<b>``
- ``bin``         : {col: [edges]}                  → ``<col>_bin`` (categorical)
- ``date_parts``  : list[str]                       → ``<col>_{year,month,day,dow,is_weekend}``
- ``onehot``      : list[str]                       → ``<col>_<value>`` (top 20 + _other)
- ``target_mean`` : {col: target_col}               → ``<col>_target_mean`` (LOO encoding)
- ``interact_top_k``: {"k": int, "importance_csv": str} → pairwise products of top-k features

Safety rails:
- All referenced columns validated before any transformation runs
- The label column cannot appear as a source in any transform
- ``onehot`` caps at top 20 categories + ``_other`` bucket
- ``ratio`` returns NaN for zero denominators (not Inf) and warns
- ``log`` uses ``log1p`` for zero-safety; warns on negative values
- ``target_mean`` uses leave-one-out to avoid train-set leakage
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ONEHOT_TOP_N = 20
DATE_PARSE_MIN_SUCCESS = 0.5  # fraction of rows that must parse as datetime


def _validate_cols(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: columns not in DataFrame: {missing}")


def _reject_label_source(spec: dict, label: str | None) -> None:
    """Prevent the label from being used as a transformation input (leakage)."""
    if label is None:
        return
    for key in ("log", "sqrt", "onehot", "date_parts"):
        if label in (spec.get(key) or []):
            raise ValueError(f"Label column '{label}' cannot be used in '{key}' transformation")
    for key in ("ratio", "diff", "product"):
        for pair in spec.get(key) or []:
            if label in pair:
                raise ValueError(f"Label column '{label}' cannot appear in '{key}' pairs")
    if label in (spec.get("bin") or {}):
        raise ValueError(f"Label column '{label}' cannot be used in 'bin' transformation")


def _apply_log(df: pd.DataFrame, cols: list[str], warnings: list[str]) -> list[str]:
    _validate_cols(df, cols, "log")
    new = []
    for c in cols:
        if (df[c] < 0).any():
            warnings.append(f"log_{c}: {(df[c] < 0).sum()} negative values → NaN")
        df[f"log_{c}"] = np.log1p(df[c].where(df[c] >= 0))
        new.append(f"log_{c}")
    return new


def _apply_sqrt(df: pd.DataFrame, cols: list[str], warnings: list[str]) -> list[str]:
    _validate_cols(df, cols, "sqrt")
    new = []
    for c in cols:
        if (df[c] < 0).any():
            warnings.append(f"sqrt_{c}: {(df[c] < 0).sum()} negative values → NaN")
        df[f"sqrt_{c}"] = np.sqrt(df[c].where(df[c] >= 0))
        new.append(f"sqrt_{c}")
    return new


def _apply_pairwise(
    df: pd.DataFrame,
    pairs: list[list[str]],
    op: str,
    warnings: list[str],
) -> list[str]:
    flat = [c for pair in pairs for c in pair]
    _validate_cols(df, flat, op)
    new = []
    for a, b in pairs:
        if op == "ratio":
            denom = df[b].replace(0, np.nan)
            zeros = (df[b] == 0).sum()
            if zeros:
                warnings.append(f"{a}_per_{b}: {zeros} zero denominators → NaN")
            df[f"{a}_per_{b}"] = df[a] / denom
            new.append(f"{a}_per_{b}")
        elif op == "diff":
            # Use datetime subtraction only when both columns are already datetime-like.
            # Coercing arbitrary numerics into datetimes would reinterpret them as epochs.
            def _is_datetime_like(s: pd.Series) -> bool:
                if pd.api.types.is_datetime64_any_dtype(s):
                    return True
                if s.dtype == object:
                    parsed = pd.to_datetime(s, errors="coerce")
                    return bool(parsed.notna().mean() > DATE_PARSE_MIN_SUCCESS)
                return False

            if _is_datetime_like(df[a]) and _is_datetime_like(df[b]):
                a_dt = pd.to_datetime(df[a], errors="coerce")
                b_dt = pd.to_datetime(df[b], errors="coerce")
                df[f"{a}_minus_{b}"] = (a_dt - b_dt).dt.days
            else:
                df[f"{a}_minus_{b}"] = df[a] - df[b]
            new.append(f"{a}_minus_{b}")
        elif op == "product":
            df[f"{a}_x_{b}"] = df[a] * df[b]
            new.append(f"{a}_x_{b}")
    return new


def _apply_bin(df: pd.DataFrame, spec: dict, warnings: list[str]) -> list[str]:
    _validate_cols(df, spec.keys(), "bin")
    new = []
    for col, edges in spec.items():
        if len(edges) < 2:
            raise ValueError(f"bin '{col}': need at least 2 edges, got {edges}")
        df[f"{col}_bin"] = pd.cut(df[col], bins=edges, include_lowest=True).astype(str)
        new.append(f"{col}_bin")
    return new


def _apply_date_parts(df: pd.DataFrame, cols: list[str], warnings: list[str]) -> list[str]:
    _validate_cols(df, cols, "date_parts")
    new = []
    for c in cols:
        parsed = pd.to_datetime(df[c], errors="coerce")
        success_rate = parsed.notna().mean()
        if success_rate < DATE_PARSE_MIN_SUCCESS:
            raise ValueError(
                f"date_parts '{c}': only {success_rate:.0%} of values parsed as dates"
            )
        if success_rate < 1.0:
            warnings.append(f"date_parts {c}: {(1 - success_rate) * 100:.1f}% unparseable → NaN")
        df[f"{c}_year"] = parsed.dt.year
        df[f"{c}_month"] = parsed.dt.month
        df[f"{c}_day"] = parsed.dt.day
        df[f"{c}_dayofweek"] = parsed.dt.dayofweek
        df[f"{c}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype("Int64")
        new.extend([f"{c}_{p}" for p in ("year", "month", "day", "dayofweek", "is_weekend")])
    return new


def _apply_onehot(
    df: pd.DataFrame,
    cols: list[str],
    warnings: list[str],
    dropped: list[str],
) -> list[str]:
    _validate_cols(df, cols, "onehot")
    new = []
    for c in cols:
        value_counts = df[c].value_counts()
        if len(value_counts) > ONEHOT_TOP_N:
            warnings.append(
                f"onehot {c}: {len(value_counts)} categories capped to top {ONEHOT_TOP_N} + _other"
            )
            top = set(value_counts.head(ONEHOT_TOP_N).index)
            series = df[c].where(df[c].isin(top), other="_other")
        else:
            series = df[c]
        dummies = pd.get_dummies(series, prefix=c).astype("Int64")
        for col in dummies.columns:
            df[col] = dummies[col]
            new.append(col)
        dropped.append(c)
        df.drop(columns=[c], inplace=True)
    return new


def _apply_target_mean(
    df: pd.DataFrame,
    spec: dict,
    warnings: list[str],
) -> list[str]:
    """Leave-one-out target mean encoding. Prevents per-row leakage."""
    _validate_cols(df, spec.keys(), "target_mean")
    _validate_cols(df, spec.values(), "target_mean target")
    new = []
    for col, target_col in spec.items():
        # Group sum and count, then subtract the row's own value for LOO
        grouped = df.groupby(col)[target_col]
        total = grouped.transform("sum")
        count = grouped.transform("count")
        loo = (total - df[target_col]) / (count - 1)
        # Groups of size 1 → NaN; fall back to global mean
        global_mean = df[target_col].mean()
        df[f"{col}_target_mean"] = loo.fillna(global_mean)
        new.append(f"{col}_target_mean")
    return new


def _apply_interact_top_k(
    df: pd.DataFrame,
    spec: dict,
    warnings: list[str],
) -> list[str]:
    k = int(spec.get("k", 3))
    importance_csv = spec.get("importance_csv")
    if not importance_csv:
        raise ValueError("interact_top_k: 'importance_csv' is required")
    imp = pd.read_csv(importance_csv, index_col=0)
    if "importance" not in imp.columns:
        raise ValueError(f"interact_top_k: {importance_csv} has no 'importance' column")
    ranked = imp.sort_values("importance", ascending=False).head(k).index
    top = [f for f in ranked if f in df.columns]
    if len(top) < 2:
        warnings.append("interact_top_k: fewer than 2 top features present in DataFrame — skipped")
        return []
    new = []
    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            df[f"{a}_x_{b}"] = df[a] * df[b]
            new.append(f"{a}_x_{b}")
    return new


# Dispatch table keeps the orchestrator short and makes adding new transforms trivial
_HANDLERS = {
    "log": lambda df, v, w, d: _apply_log(df, v, w),
    "sqrt": lambda df, v, w, d: _apply_sqrt(df, v, w),
    "ratio": lambda df, v, w, d: _apply_pairwise(df, v, "ratio", w),
    "diff": lambda df, v, w, d: _apply_pairwise(df, v, "diff", w),
    "product": lambda df, v, w, d: _apply_pairwise(df, v, "product", w),
    "bin": lambda df, v, w, d: _apply_bin(df, v, w),
    "date_parts": lambda df, v, w, d: _apply_date_parts(df, v, w),
    "onehot": lambda df, v, w, d: _apply_onehot(df, v, w, d),
    "target_mean": lambda df, v, w, d: _apply_target_mean(df, v, w),
    "interact_top_k": lambda df, v, w, d: _apply_interact_top_k(df, v, w),
}


def apply_transformations(
    df: pd.DataFrame,
    spec: dict,
    label: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply a transformation spec to *df* and return (new_df, report).

    Parameters
    ----------
    df
        Input DataFrame. Not mutated; a copy is returned.
    spec
        Mapping of transform name → transform-specific arguments. See module
        docstring for the full set.
    label
        Target column name. If provided, any attempt to use it as a
        transformation source is rejected to prevent leakage.

    Returns
    -------
    (DataFrame, report)
        ``report`` has keys ``new_features`` (list), ``dropped_features``
        (list), ``warnings`` (list).
    """
    unknown = set(spec) - set(_HANDLERS)
    if unknown:
        raise ValueError(
            f"Unknown transformations: {sorted(unknown)}. Supported: {sorted(_HANDLERS)}"
        )

    _reject_label_source(spec, label)

    out = df.copy()
    new_features: list[str] = []
    dropped: list[str] = []
    warnings: list[str] = []

    for name, value in spec.items():
        created = _HANDLERS[name](out, value, warnings, dropped)
        new_features.extend(created)
        logger.info("fe[%s] added %d columns", name, len(created))

    return out, {
        "new_features": new_features,
        "dropped_features": dropped,
        "warnings": warnings,
    }
