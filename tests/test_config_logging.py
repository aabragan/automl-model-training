"""Tests for config.setup_logging."""

from __future__ import annotations

import logging

from automl_model_training.config import PACKAGE_LOGGER, setup_logging


class TestSetupLogging:
    def test_default_level_is_info(self):
        setup_logging()
        logger = logging.getLogger(PACKAGE_LOGGER)
        assert logger.level == logging.INFO

    def test_verbose_sets_debug(self):
        setup_logging(verbose=True)
        logger = logging.getLogger(PACKAGE_LOGGER)
        assert logger.level == logging.DEBUG

    def test_quiet_sets_warning(self):
        setup_logging(quiet=True)
        logger = logging.getLogger(PACKAGE_LOGGER)
        assert logger.level == logging.WARNING

    def test_handler_writes_to_stdout(self):
        setup_logging()
        logger = logging.getLogger(PACKAGE_LOGGER)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_clears_previous_handlers(self):
        setup_logging()
        setup_logging()
        logger = logging.getLogger(PACKAGE_LOGGER)
        # Should not accumulate handlers across calls
        assert len(logger.handlers) == 1
