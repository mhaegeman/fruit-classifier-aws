.PHONY: install install-dev lint test clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not spark and not aws"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
