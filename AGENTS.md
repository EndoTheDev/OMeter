# OMeter

## Purpose

Benchmark and compare Ollama models across local and cloud endpoints. Provides a CLI tool for listing, benchmarking, and exporting model performance data, plus a GitHub Pages dashboard for public trend visualization.

## Ownership

- **Owner:** EndoTheDev
- **Repository:** https://github.com/EndoTheDev/OMeter
- **License:** MIT

## Core Contract

- No backward compatibility or legacy code — always start fresh and clean.
- No comments or docstrings unless absolutely necessary (like for tools).
- Use descriptive variable and function names.
- Prefer classes over functions.
- Follow the DRY principle.
- Never modify `.env` — only update `.env.example`.
- Never stage, commit, or push without explicit permission.

## Read Before Editing

- The project uses `uv` for everything — no pip.
- Python 3.14+ required. The formatter normalizes `except A, B:` syntax (comma-separated, not `as`). It is NOT a bug.
- Source lives under `src/ometer/` (hatchling layout).
- Tests live under `tests/` with pytest.
- The dashboard is a single HTML file at `docs/index.html` with Vue 3 + Chart.js from CDN.
- Benchmark data is at `docs/data/benchmark-history.json`, updated by a GitHub Actions cron.

## Update After Editing

- Run `uv run pytest --cov` after any code change.
- If the dashboard HTML changes, verify it loads locally with `python3 -m http.server 8080` in `docs/`.
- If the workflow changes, verify the YAML is valid.
- Update this AGENTS.md if project structure, conventions, or commands change.

## Hierarchy

```
AGENTS.md (root — this file)
├── src/ometer/AGENTS.md  — core library
├── tests/AGENTS.md       — test suite
├── docs/AGENTS.md        — documentation
└── .github/AGENTS.md     — CI/CD
```

## Child Doc Shape

Every child AGENTS.md must follow this section order:

1. **Purpose** — what this directory owns
2. **Ownership** — who maintains it
3. **Local Contracts** — rules specific to this scope
4. **Work Guidance** — how to work here
5. **Verification** — how to verify changes
6. **Child DOX Index** — list of sub-children (if any)

Children may NOT weaken the root Core Contract. They may only add scope-specific rules.

## Style

- Write in the imperative mood.
- Be concise — one sentence per rule where possible.
- No diary entries, no commit history, no rationales.
- Use fenced code blocks for commands and file paths.

## Closeout

- After any change to this file or its children, run `uv run pytest --cov` to confirm nothing broke.
- If a child doc is added or removed, update the Hierarchy section and the parent's Child DOX Index.

## User Preferences

- Endo prefers direct, no-fluff communication.
- Approves with "lgtm" / "go ahead".
- Expects todo lists updated during multi-step work.
- Prefers ponytail-review output in table format.
- Controls when DOX/AGENTS.md updates happen — may say "skip DOX for now".
- During review apply passes, wants item-by-item control.

## Commands

- `uv run ...` for running python commands.
- `uv run pytest --cov` for running tests with coverage report.
- `uv run python -m http.server 8080` in `docs/` to serve the dashboard locally.

## Infos

- ollama cloud does not support embedding models yet.
- GitHub Pages URL: https://EndoTheDev.github.io/OMeter/
- Benchmark cron: every 6h via `.github/workflows/benchmark.yml`
- Secret: `OLLAMA_CLOUD_API_KEY` in GitHub Actions secrets

## Child DOX Index

| Path                   | Status | What it owns                                                      |
| ---------------------- | ------ | ----------------------------------------------------------------- |
| `src/ometer/AGENTS.md` | Stub   | Core library modules (cli, api, config, display, export, history) |
| `tests/AGENTS.md`      | Stub   | Test suite conventions and coverage targets                       |
| `docs/AGENTS.md`       | Stub   | Documentation files and the web dashboard                         |
| `.github/AGENTS.md`    | Stub   | CI/CD workflows and merge scripts                                 |
