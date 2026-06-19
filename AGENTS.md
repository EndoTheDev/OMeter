# OMeter

## Commands

- `uv run ...` for running python commands.
- `uv run pytest --cov` for running tests with coverage report.

## Rules

- write self-documenting code
- no comments or docstrings unless absolutely necessary (like for tools)
- use descriptive variable and function names
- avoid over-engineering or unnecessary complexity
- prefer classes over functions
- follow the DRY (Don't Repeat Yourself) principle
- never stage, commit or push any changes (read-only git commands only!)
- no backward compatibility or legacy code, always start fresh and clean
- never modify .env file. only update .env.example.

## Infos

- ollama cloud does not support embedding models yet.
