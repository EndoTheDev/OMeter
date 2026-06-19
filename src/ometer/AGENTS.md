# src/ometer — Core Library

## Purpose

All Python source code for the OMeter CLI tool. Six modules handling distinct concerns: CLI entry point, configuration, API communication, display/UI, export, and history persistence.

## Ownership

- **Owner:** EndoTheDev
- **Package:** `ometer` (installed via `uv tool install .`)

## Local Contracts

- No injection paths in production constructors — not even optional params for test injection. Tests mock at import boundary with `unittest.mock.patch`.
- No re-exports at intermediate layers. Each package imports symbols directly from the owning module.
- `model_name` everywhere (not `model_id`).
- ProviderConfig has no `provider` field — class identity carries it.

## Work Guidance

- Source layout: `src/ometer/{cli,api,config,display,export,history}.py`
- Entry point: `ometer.cli:main_entrypoint`
- CLI uses `rich` + `InquirerPy` for interactive mode.
- API calls use `httpx` for HTTP communication with Ollama endpoints.
- History uses SQLite via stdlib `sqlite3`.

## Verification

- `uv run pytest --cov` — all tests must pass.
- `uv run ometer --help` — CLI must start without errors.

## Child DOX Index

None — leaf node.
