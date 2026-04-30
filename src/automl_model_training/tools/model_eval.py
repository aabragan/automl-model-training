from __future__ import annotations

from pathlib import Path

import pandas as pd


def tool_model_subset_evaluate(
    run_dir: str,
    score_tolerance: float = 0.01,
) -> dict:
    """Report per-model test-set scores from an AutoGluon ensemble.

    Answers the common question: "is the full ensemble actually worth it,
    or would deploying a single model be almost as good?" by surfacing
    the score and inference cost of every model in the ensemble and
    flagging the cheapest model whose score is within ``score_tolerance``
    of the best.

    Reads ``leaderboard_test.csv`` (already written by ``tool_train``) so
    no re-inference is needed. This matches the repo convention of
    treating leaderboard artifacts as the canonical per-model summary
    and keeps the tool cheap to invoke.

    Parameters
    ----------
    run_dir : str
        Training run directory.
    score_tolerance : float
        A single model qualifies as a "near-best substitute" when its
        test score is within ``score_tolerance`` of the ensemble best
        (on the same absolute scale). Default 0.01.

    Returns
    -------
    dict with keys:
        models : list of {
            model,
            score_test,
            score_val,
            pred_time_test,       — inference time (s)
            fit_time,
            stack_level,          — 1=base model, >1=stacker
            is_ensemble,          — heuristic: WeightedEnsemble* or stack_level>1
        }
        best_model              — ensemble top by score_test
        best_single_model       — top non-ensemble model
        score_gap               — best ensemble score − best single-model score
        recommended_deploy      — {"model", "reason", "speedup"} — cheapest
                                    model within score_tolerance of best, or
                                    None if only the ensemble qualifies
        hints : [str]

    Raises
    ------
    FileNotFoundError : missing leaderboard_test.csv
    """
    lb_path = Path(run_dir) / "leaderboard_test.csv"
    if not lb_path.exists():
        raise FileNotFoundError(f"tool_model_subset_evaluate: missing {lb_path}")

    lb = pd.read_csv(lb_path)
    required = {"model", "score_test", "stack_level"}
    missing = required - set(lb.columns)
    if missing:
        raise ValueError(
            f"leaderboard_test.csv is missing expected columns {sorted(missing)}; "
            f"got {list(lb.columns)}"
        )

    # AutoGluon always stores score_test as higher-is-better (it negates the
    # raw metric internally for lower-is-better metrics like RMSE and log_loss).
    # We can therefore sort by raw score_test descending regardless of metric,
    # and use score_test directly for the "within tolerance" check.
    lb = lb.copy()
    lb = lb.sort_values("score_test", ascending=False).reset_index(drop=True)

    def _is_ensemble(row: pd.Series) -> bool:
        model = str(row["model"])
        return model.startswith("WeightedEnsemble") or int(row.get("stack_level", 1)) > 1

    models_out: list[dict] = []
    for _, row in lb.iterrows():
        models_out.append(
            {
                "model": str(row["model"]),
                "score_test": round(float(row["score_test"]), 6),
                "score_val": (
                    round(float(row["score_val"]), 6)
                    if "score_val" in row and pd.notna(row["score_val"])
                    else None
                ),
                "pred_time_test": (
                    round(float(row["pred_time_test"]), 6)
                    if "pred_time_test" in row and pd.notna(row["pred_time_test"])
                    else None
                ),
                "fit_time": (
                    round(float(row["fit_time"]), 3)
                    if "fit_time" in row and pd.notna(row["fit_time"])
                    else None
                ),
                "stack_level": int(row["stack_level"]) if pd.notna(row["stack_level"]) else 1,
                "is_ensemble": bool(_is_ensemble(row)),
            }
        )

    if not models_out:
        raise RuntimeError(
            f"tool_model_subset_evaluate: leaderboard_test.csv at {lb_path} is empty."
        )

    best = models_out[0]
    # Single-model subset: exclude ensembles
    singles = [m for m in models_out if not m["is_ensemble"]]
    best_single = singles[0] if singles else None

    score_gap = None
    if best_single is not None:
        score_gap = round(best["score_test"] - best_single["score_test"], 6)

    # Deployment recommendation: the cheapest model (by pred_time_test) whose
    # score is within score_tolerance of the best. Tie-break by higher score.
    # Because score_test is always higher-is-better after AutoGluon's internal
    # negation, we can compare the raw scores directly instead of absolutes.
    recommended_deploy: dict | None = None
    best_score = best["score_test"]
    candidates = [m for m in models_out if (best_score - m["score_test"]) <= score_tolerance]
    if candidates:
        # Among models within tolerance, prefer the fastest inference time;
        # fall back to higher score when pred_time is missing
        def _sort_key(m: dict) -> tuple:
            pt = m["pred_time_test"]
            pt_val = pt if pt is not None else float("inf")
            return (pt_val, -m["score_test"])

        cheapest = min(candidates, key=_sort_key)
        best_pt = best["pred_time_test"]
        cheapest_pt = cheapest["pred_time_test"]
        if (
            cheapest["model"] != best["model"]
            and best_pt is not None
            and cheapest_pt is not None
            and cheapest_pt > 0
        ):
            speedup = round(best_pt / cheapest_pt, 2)
            recommended_deploy = {
                "model": cheapest["model"],
                "reason": (
                    f"Within {score_tolerance} of the best score "
                    f"({cheapest['score_test']} vs {best['score_test']}) but "
                    f"{speedup}x faster at inference."
                ),
                "speedup": speedup,
            }

    hints: list[str] = []
    if recommended_deploy is not None:
        hints.append(
            f"Consider deploying '{recommended_deploy['model']}' instead of "
            f"'{best['model']}': {recommended_deploy['reason']}"
        )
    if best_single is not None and score_gap is not None and score_gap < score_tolerance:
        hints.append(
            f"Single model '{best_single['model']}' scores {best_single['score_test']}, "
            f"only {score_gap} below the ensemble. The ensemble may not be worth "
            "its inference cost and complexity."
        )
    if len(models_out) == 1:
        hints.append(
            "Only one model in the leaderboard — likely a tuning run rather than "
            "an ensemble. Nothing to compare."
        )

    return {
        "run_dir": run_dir,
        "models": models_out,
        "best_model": best,
        "best_single_model": best_single,
        "score_gap": score_gap,
        "recommended_deploy": recommended_deploy,
        "hints": hints,
    }
