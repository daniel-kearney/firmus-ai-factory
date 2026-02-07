# ─────────────────────────────────────────────────────────────────────────────
# Firmus AI Factory — Development Makefile
# ─────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON ?= python3
PIP    ?= pip3
SRC    := src/firmus_ai_factory
TESTS  := tests
EXAMPLES := examples

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: install
install:  ## Install package in development mode with all dependencies
	$(PIP) install -e ".[dev]"
	$(PIP) install numpy scipy pandas matplotlib pydantic numba \
	               pymoo optuna cvxpy statsmodels control \
	               python-dateutil pytz plotly \
	               pytest pytest-cov pytest-timeout pytest-xdist \
	               ruff black isort mypy pre-commit

.PHONY: install-ci
install-ci:  ## Install minimal dependencies for CI
	$(PIP) install -e ".[dev]"
	$(PIP) install ruff black isort mypy pytest pytest-cov pytest-timeout

# ─────────────────────────────────────────────────────────────────────────────
# Quality
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: lint
lint:  ## Run all linters (ruff, black, isort)
	ruff check $(SRC) $(TESTS) $(EXAMPLES)
	black --check --diff $(SRC) $(TESTS) $(EXAMPLES)
	isort --check-only --diff $(SRC) $(TESTS) $(EXAMPLES)

.PHONY: format
format:  ## Auto-format code (black, isort, ruff fix)
	isort $(SRC) $(TESTS) $(EXAMPLES)
	black $(SRC) $(TESTS) $(EXAMPLES)
	ruff check --fix $(SRC) $(TESTS) $(EXAMPLES)

.PHONY: typecheck
typecheck:  ## Run mypy type checking
	mypy $(SRC) --ignore-missing-imports --no-strict-optional

.PHONY: check
check: lint typecheck  ## Run all quality checks (lint + typecheck)

# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: test
test:  ## Run all tests
	$(PYTHON) -m pytest $(TESTS) -v --timeout=120

.PHONY: test-unit
test-unit:  ## Run unit tests only
	$(PYTHON) -m pytest $(TESTS)/unit -v --timeout=60

.PHONY: test-integration
test-integration:  ## Run integration tests only
	$(PYTHON) -m pytest $(TESTS)/integration -v --timeout=120

.PHONY: test-fast
test-fast:  ## Run tests excluding slow markers
	$(PYTHON) -m pytest $(TESTS) -v --timeout=60 -m "not slow"

.PHONY: test-parallel
test-parallel:  ## Run tests in parallel (requires pytest-xdist)
	$(PYTHON) -m pytest $(TESTS) -v --timeout=120 -n auto

.PHONY: coverage
coverage:  ## Run tests with coverage report
	$(PYTHON) -m pytest $(TESTS) -v --timeout=120 \
		--cov=firmus_ai_factory \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/ \
		--cov-report=xml:coverage.xml

.PHONY: coverage-open
coverage-open: coverage  ## Run coverage and open HTML report
	open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html

# ─────────────────────────────────────────────────────────────────────────────
# Examples
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: demo
demo:  ## Run the closed-loop control demo
	$(PYTHON) $(EXAMPLES)/04_closed_loop_emulated_sensors.py

.PHONY: demo-thermal
demo-thermal:  ## Run GPU thermal analysis example
	$(PYTHON) $(EXAMPLES)/01_gpu_thermal_analysis.py

.PHONY: demo-integration
demo-integration:  ## Run complete system integration example
	$(PYTHON) $(EXAMPLES)/03_complete_system_integration.py

# ─────────────────────────────────────────────────────────────────────────────
# Build & Release
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: build
build:  ## Build distribution packages
	$(PYTHON) -m build

.PHONY: release-check
release-check: check test build  ## Full pre-release validation
	twine check dist/*
	@echo "✅ All checks passed. Ready to release."

.PHONY: tag
tag:  ## Create a version tag (usage: make tag VERSION=v0.2.0)
	@if [ -z "$(VERSION)" ]; then echo "Usage: make tag VERSION=v0.2.0"; exit 1; fi
	git tag -a $(VERSION) -m "Release $(VERSION)"
	git push origin $(VERSION)
	@echo "✅ Tag $(VERSION) pushed. GitHub Actions will create the release."

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage coverage.xml
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help message
	@echo "Firmus AI Factory — Development Commands"
	@echo "─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
