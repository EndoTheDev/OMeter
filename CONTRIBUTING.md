# Contributing to OMeter

## Branch Strategy

- Never commit directly to main
- Always work from a feature branch
- Branch naming: feat/description, fix/description, docs/description

## PR Workflow

1. Create a branch from main
2. Make your changes
3. Run tests: uv run pytest --cov (must pass, 100% coverage)
4. Push and open a PR against main
5. Wait for review, then merge

## Code Style

- Self-documenting code — no comments/docstrings unless absolutely necessary
- Descriptive variable/function names
- Avoid over-engineering
- Prefer classes over functions
- DRY principle
- No backward compatibility — always start fresh

## Running Tests

```bash
uv run pytest --cov
```
