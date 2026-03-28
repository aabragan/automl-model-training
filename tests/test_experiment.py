"""Tests for experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from automl_model_training.experiment import (
    compare_experiments,
    load_experiments,
    record_experiment,
)


class TestRecordExperiment:
    def test_creates_jsonl_file(self, tmp_path: Path):
        log = str(tmp_path / "exp.jsonl")
        record_experiment("run1", {"label": "target"}, {"f1": 0.9}, experiment_log=log)
        assert Path(log).exists()

    def test_appends_entries(self, tmp_path: Path):
        log = str(tmp_path / "exp.jsonl")
        record_experiment("run1", {"a": 1}, {"f1": 0.8}, experiment_log=log)
        record_experiment("run2", {"a": 2}, {"f1": 0.9}, experiment_log=log)

        lines = Path(log).read_text().strip().splitlines()
        assert len(lines) == 2

    def test_entry_has_required_keys(self, tmp_path: Path):
        log = str(tmp_path / "exp.jsonl")
        entry = record_experiment("run1", {"label": "t"}, {"f1": 0.85}, experiment_log=log)

        assert "timestamp" in entry
        assert entry["output_dir"] == "run1"
        assert entry["params"] == {"label": "t"}
        assert entry["metrics"] == {"f1": 0.85}

    def test_reads_model_info_if_present(self, tmp_path: Path):
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "model_info.json").write_text(
            json.dumps({"best_model": "LightGBM", "problem_type": "binary"})
        )

        log = str(tmp_path / "exp.jsonl")
        entry = record_experiment(str(run_dir), {}, {}, experiment_log=log)

        assert entry["model_info"]["best_model"] == "LightGBM"

    def test_no_model_info_when_missing(self, tmp_path: Path):
        log = str(tmp_path / "exp.jsonl")
        entry = record_experiment("nonexistent_dir", {}, {}, experiment_log=log)
        assert "model_info" not in entry


class TestLoadExperiments:
    def test_returns_empty_for_missing_file(self, tmp_path: Path):
        entries = load_experiments(str(tmp_path / "nope.jsonl"))
        assert entries == []

    def test_loads_all_entries(self, tmp_path: Path):
        log = str(tmp_path / "exp.jsonl")
        record_experiment("r1", {"a": 1}, {}, experiment_log=log)
        record_experiment("r2", {"a": 2}, {}, experiment_log=log)

        entries = load_experiments(log)
        assert len(entries) == 2
        assert entries[0]["output_dir"] == "r1"
        assert entries[1]["output_dir"] == "r2"


class TestCompareExperiments:
    def test_returns_empty_df_for_no_log(self, tmp_path: Path):
        df = compare_experiments(str(tmp_path / "nope.jsonl"))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_flattens_params_and_metrics(self, tmp_path: Path):
        log = str(tmp_path / "exp.jsonl")
        record_experiment("r1", {"preset": "best"}, {"f1": 0.9}, experiment_log=log)

        df = compare_experiments(log)
        assert len(df) == 1
        assert "param_preset" in df.columns
        assert "metric_f1" in df.columns
        assert df.iloc[0]["param_preset"] == "best"

    def test_last_n_filter(self, tmp_path: Path):
        log = str(tmp_path / "exp.jsonl")
        for i in range(5):
            record_experiment(f"r{i}", {"i": i}, {}, experiment_log=log)

        df = compare_experiments(log, last_n=2)
        assert len(df) == 2
        assert df.iloc[0]["param_i"] == 3
        assert df.iloc[1]["param_i"] == 4

    def test_includes_model_info_fields(self, tmp_path: Path):
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "model_info.json").write_text(
            json.dumps({"best_model": "CatBoost", "problem_type": "regression"})
        )

        log = str(tmp_path / "exp.jsonl")
        record_experiment(str(run_dir), {}, {}, experiment_log=log)

        df = compare_experiments(log)
        assert df.iloc[0]["best_model"] == "CatBoost"
        assert df.iloc[0]["problem_type"] == "regression"
