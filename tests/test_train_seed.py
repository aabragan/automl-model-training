"""Tests for --seed and --profile train features."""

from __future__ import annotations

from automl_model_training.config import DEFAULT_RANDOM_STATE


class TestSeedDefault:
    def test_default_seed_matches_config(self):
        """The --seed default should match DEFAULT_RANDOM_STATE."""
        from automl_model_training.train import _base_parser

        parser = _base_parser("test")
        args = parser.parse_args(["dummy.csv"])
        assert args.seed == DEFAULT_RANDOM_STATE

    def test_custom_seed_is_passed(self):
        from automl_model_training.train import _base_parser

        parser = _base_parser("test")
        args = parser.parse_args(["dummy.csv", "--seed", "123"])
        assert args.seed == 123


class TestProfileFlag:
    def test_profile_flag_defaults_to_false(self):
        from automl_model_training.train import _base_parser

        parser = _base_parser("test")
        args = parser.parse_args(["dummy.csv"])
        assert args.profile is False

    def test_profile_flag_can_be_enabled(self):
        from automl_model_training.train import _base_parser

        parser = _base_parser("test")
        args = parser.parse_args(["dummy.csv", "--profile"])
        assert args.profile is True
