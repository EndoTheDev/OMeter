# tests — Test Suite

## Purpose

Pytest-based test suite covering all core library modules. Uses pytest-asyncio for async tests and pytest-httpx for mocking HTTP calls.

## Ownership

- **Owner:** EndoTheDev
- **Framework:** pytest with coverage

## Local Contracts

- Tests mock at import boundary with `unittest.mock.patch` — no injection paths in production code.
- One test file per source module: `test_{module}.py`.
- Use `pytest-asyncio` for async test functions.
- Use `pytest-httpx` for mocking `httpx` HTTP calls.

## Work Guidance

- Run tests: `uv run pytest --cov`
- Run a single file: `uv run pytest tests/test_api.py -v`
- Coverage target: no hard minimum, but don't drop existing coverage.

## Verification

- `uv run pytest --cov` — all tests pass, no regressions.

## Child DOX Index

None — leaf node.
