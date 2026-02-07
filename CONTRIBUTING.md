# Contributing to Firmus AI Factory

Thank you for your interest in contributing to the Firmus AI Factory digital twin platform. This document provides guidelines and instructions for contributing.

## Development Setup

Clone the repository and install all dependencies in development mode:

```bash
git clone https://github.com/daniel-kearney/firmus-ai-factory.git
cd firmus-ai-factory
python -m venv .venv
source .venv/bin/activate
make install
```

Install pre-commit hooks to automatically check code quality before each commit:

```bash
pre-commit install
```

## Development Workflow

The project follows a trunk-based development model with short-lived feature branches. All changes must pass the CI pipeline before merging into `main`.

**Step 1: Create a feature branch** from `main` with a descriptive name following the convention `feature/<module>-<description>` for new features, `fix/<module>-<description>` for bug fixes, or `docs/<description>` for documentation changes.

```bash
git checkout -b feature/grid-battery-dispatch
```

**Step 2: Make your changes** following the code style guidelines. Each module should maintain its mathematical rigor with clear docstrings explaining the physical models, equations, and assumptions. Use SI units throughout unless a domain convention dictates otherwise (e.g., kW for power, °C for temperature).

**Step 3: Run quality checks** locally before committing. The `make check` command runs both linting and type checking, while `make test` executes the full test suite. For faster iteration during development, use `make test-unit` to run only unit tests.

```bash
make format     # Auto-format code
make check      # Lint + type check
make test       # Run all tests
make coverage   # Generate coverage report
```

**Step 4: Commit with conventional commit messages** using the format `<type>(<scope>): <description>`. Valid types include `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, and `ci`. The scope should reference the module name (e.g., `power`, `grid`, `optimization`).

```bash
git commit -m "feat(grid): add battery dispatch co-optimization"
```

**Step 5: Push and open a pull request** against `main`. The CI pipeline will automatically run linting, type checking, and the full test matrix across Python 3.9, 3.10, and 3.11. All checks must pass before the PR can be merged.

## Code Standards

All source code must comply with the following standards. The CI pipeline enforces these automatically, and pre-commit hooks catch issues before they reach the remote repository.

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Ruff** | Fast Python linter | `ruff.toml` |
| **Black** | Code formatter (100-char line length) | `pyproject.toml` |
| **isort** | Import sorting (Black-compatible profile) | `pyproject.toml` |
| **mypy** | Static type checking | `mypy.ini` |

## Testing Requirements

Every new feature or bug fix must include corresponding tests. Unit tests go in `tests/unit/test_<module>.py` and integration tests in `tests/integration/`. The minimum coverage threshold is 40%, and PRs should not decrease overall coverage.

Tests should validate both the mathematical correctness of physical models (e.g., energy conservation, efficiency bounds) and the software behaviour (e.g., edge cases, error handling). Use the fixtures defined in `tests/conftest.py` for consistent test setup.

## Module Architecture

Each module follows a consistent structure with a public API exposed through `__init__.py`, dataclass-based configuration, and clear separation between the mathematical model and its integration interface. When adding a new module, follow the existing patterns in `src/firmus_ai_factory/power/` or `src/firmus_ai_factory/grid/` as reference implementations.

## Releasing

Releases are automated through GitHub Actions. To create a new release, update the version in `setup.py`, commit the change, and create a version tag:

```bash
make release-check   # Validate everything passes
make tag VERSION=v0.2.0
```

The release workflow will automatically run the full test suite, build distribution packages, create a GitHub Release with auto-generated changelog, and optionally publish to PyPI (when the `PYPI_TOKEN` secret is configured).
