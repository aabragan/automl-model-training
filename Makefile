.PHONY: install test lint format typecheck check clean

install:
	uv sync

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=automl_model_training --cov-report=term-missing

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

check: lint typecheck test

clean:
	rm -rf output/ predictions_output/ experiments.jsonl
	find . -type d -name __pycache__ -exec rm -rf {} +
