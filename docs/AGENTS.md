# docs — Documentation & Dashboard

## Purpose

Project documentation files and the GitHub Pages web dashboard. The dashboard is a single-page Vue 3 + Chart.js application served from `docs/index.html`.

## Ownership

- **Owner:** EndoTheDev
- **Published at:** https://EndoTheDev.github.io/OMeter/

## Local Contracts

- Dashboard is a single HTML file — no build step, no npm/Vite/SFCs.
- Vue 3, Chart.js, Tailwind CSS loaded from CDN.
- Benchmark data lives at `docs/data/benchmark-history.json`.
- Catppuccin Latte/Mocha for light/dark mode via `prefers-color-scheme`.
- No API keys or credentials in the dashboard.

## Work Guidance

- Serve locally: `uv run python -m http.server 8080` in `docs/`.
- The dashboard auto-fetches `data/benchmark-history.json` relative to itself.
- Charts only render when ≥2 benchmark runs exist.
- All filtering (capabilities, context, params, search) is client-side Vue computed properties.

## Verification

- Open `http://localhost:8080` after serving — no console errors.
- Toggle system color scheme — both dark and light modes render correctly.
- Check that `data/benchmark-history.json` loads (check Network tab).

## Child DOX Index

None — leaf node.
