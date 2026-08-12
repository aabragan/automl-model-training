"""Tests for automl_model_training.data."""

from pathlib import Path

import numpy as np
import pandas as pd

from automl_model_training.data import load_and_prepare


def _write_csv(path: Path, n: int = 100, label: str = "target") -> str:
    """Write a small CSV and return its path as a string."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "drop_me": rng.randn(n),
            label: rng.choice([0, 1], n),
        }
    )
    csv_path = path / "data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_and_prepare_returns_correct_shapes(tmp_path: Path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"

    train, test, train_n, test_n, num_cols = load_and_prepare(
        csv_path=csv,
        label="target",
        features_to_drop=[],
        test_size=0.2,
        random_state=42,
        output_dir=str(out),
    )

    assert len(train) + len(test) == 100
    assert len(train) == len(train_n)
    assert len(test) == len(test_n)
    assert set(train.columns) == set(test.columns)


def test_load_and_prepare_drops_features(tmp_path: Path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"

    train, *_ = load_and_prepare(
        csv_path=csv,
        label="target",
        features_to_drop=["drop_me"],
        test_size=0.2,
        random_state=42,
        output_dir=str(out),
    )

    assert "drop_me" not in train.columns


def test_load_and_prepare_ignores_missing_drop_cols(tmp_path: Path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"

    train, *_ = load_and_prepare(
        csv_path=csv,
        label="target",
        features_to_drop=["nonexistent_col"],
        test_size=0.2,
        random_state=42,
        output_dir=str(out),
    )

    # Should not raise, original columns intact
    assert "feat_a" in train.columns


def test_load_and_prepare_warns_on_missing_drop_cols(tmp_path: Path, caplog):
    """Typos in --drop must produce a warning, not vanish silently."""
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"

    with caplog.at_level("WARNING", logger="automl_model_training.data"):
        load_and_prepare(
            csv_path=csv,
            label="target",
            features_to_drop=["drop_me", "drp_me_typo"],
            test_size=0.2,
            random_state=42,
            output_dir=str(out),
        )

    assert any("drp_me_typo" in r.message for r in caplog.records)


def test_regression_lock_disables_stratified_split(tmp_path: Path):
    """A locked regression problem_type must never stratify, even when the
    target has <= 20 unique values. A singleton value would make a
    stratified train_test_split raise — succeeding proves no stratification."""
    rng = np.random.RandomState(0)
    values = [1.5, 2.5, 3.5, 4.5, 5.5]
    df = pd.DataFrame(
        {
            "feat_a": rng.randn(60),
            "target": list(rng.choice(values, 59)) + [99.9],  # singleton value
        }
    )
    csv_path = tmp_path / "reg.csv"
    df.to_csv(csv_path, index=False)

    train, test, *_ = load_and_prepare(
        csv_path=str(csv_path),
        label="target",
        features_to_drop=[],
        test_size=0.2,
        random_state=42,
        output_dir=str(tmp_path / "out"),
        problem_type="regression",
    )

    assert len(train) + len(test) == 60


def test_auto_detect_still_stratifies_low_cardinality_labels(tmp_path: Path):
    """Without a problem_type lock, the cardinality heuristic still applies:
    a binary label gets a stratified (class-balanced) split."""
    df = pd.DataFrame(
        {
            "feat_a": np.arange(100, dtype=float),
            "target": [0] * 80 + [1] * 20,
        }
    )
    csv_path = tmp_path / "cls.csv"
    df.to_csv(csv_path, index=False)

    train, test, *_ = load_and_prepare(
        csv_path=str(csv_path),
        label="target",
        features_to_drop=[],
        test_size=0.2,
        random_state=42,
        output_dir=str(tmp_path / "out"),
    )

    # Stratified 80/20 → the 20-row test split holds exactly 4 positives
    assert int(test["target"].sum()) == 4


def test_no_shuffle_split_takes_contiguous_tail(tmp_path: Path):
    """With shuffle=False the test set is exactly the last test_size
    fraction of rows in file order — no time interleaving."""
    n = 100
    df = pd.DataFrame(
        {
            "feat_a": np.arange(n, dtype=float),
            "row_order": np.arange(n),
            "target": np.arange(n, dtype=float),
        }
    )
    csv_path = tmp_path / "ordered.csv"
    df.to_csv(csv_path, index=False)

    train, test, *_ = load_and_prepare(
        csv_path=str(csv_path),
        label="target",
        features_to_drop=[],
        test_size=0.2,
        random_state=42,
        output_dir=str(tmp_path / "out"),
        problem_type="regression",
        shuffle=False,
    )

    assert list(train["row_order"]) == list(range(80))
    assert list(test["row_order"]) == list(range(80, 100))


def test_no_shuffle_split_disables_stratification(tmp_path: Path):
    """An unshuffled split must not stratify, even for a low-cardinality
    label — sklearn raises when stratify is combined with shuffle=False,
    so succeeding proves stratification was turned off."""
    df = pd.DataFrame(
        {
            "feat_a": np.arange(100, dtype=float),
            "target": [0] * 80 + [1] * 20,
        }
    )
    csv_path = tmp_path / "cls.csv"
    df.to_csv(csv_path, index=False)

    train, test, *_ = load_and_prepare(
        csv_path=str(csv_path),
        label="target",
        features_to_drop=[],
        test_size=0.2,
        random_state=42,
        output_dir=str(tmp_path / "out"),
        shuffle=False,
    )

    # Tail slice of the ordered label: all 20 test rows are the positives
    assert int(test["target"].sum()) == 20
