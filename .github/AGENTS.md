# .github — CI/CD

## Purpose

GitHub Actions workflows and helper scripts for automated benchmarking and deployment.

## Ownership

- **Owner:** EndoTheDev
- **Workflow:** `.github/workflows/benchmark.yml`

## Local Contracts

- Benchmark runs every hour via cron.
- Uses `uv run python` (not `python3`) to use the project venv.
- `GITHUB_TOKEN` needs `contents: write` to push benchmark data back.
- Merge script at `.github/merge_benchmark.py` hardens against empty/invalid input.
- Secret `OLLAMA_CLOUD_API_KEY` stored in GitHub Actions secrets.

## Work Guidance

- To test the workflow locally, push to main and check Actions tab.
- The merge script appends latest snapshot to `docs/data/benchmark-history.json` with no cap.
- After a repo rename, update the repository URLs in `pyproject.toml` and this AGENTS.md if needed.

## Verification

- Check <https://github.com/EndoTheDev/OMeter/actions> after push.
- Verify `docs/data/benchmark-history.json` has new entries after a run.

## Child DOX Index

None — leaf node.
