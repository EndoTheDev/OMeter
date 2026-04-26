# Architecture

OllamaMeter is structured as five modules inside `src/ometer/`, each handling a specific concern in the request lifecycle.

## Module Overview

| Module  | File         | Responsibility                                                                                          |
| ------- | ------------ | ------------------------------------------------------------------------------------------------------- |
| CLI     | `cli.py`     | Entry point, argument parsing (`argparse`), interactive model selection (`InquirerPy`), export dispatch |
| Config  | `config.py`  | Hierarchical `.env` loading, runtime settings validation and clamping                                   |
| API     | `api.py`     | HTTP communication with Ollama, TTFT/TPS calculation, `BenchmarkResult` dataclass                       |
| Display | `display.py` | Rich-based terminal UI, live table updates, color thresholding, async task orchestration                |
| Export  | `export.py`  | JSON/CSV result formatting, file output, `ExportRow` dataclass                                          |

## Request Lifecycle

```txt
User runs `ometer` or `ollamameter`
         │
         ▼
   ┌──────────┐
   │  cli.py  │  main_entrypoint()
   │          │  build_parser()  ──►  argparse flags
   │          │  resolve_mode()  ──►  local / cloud / both
   │          │  match_model()   ──►  exact or family match
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ config.py│  Config.from_env()
   │          │  _load_env()  ──►  hierarchical .env search
   │          │  Clamps num_runs (1–3), num_parallel (1–10)
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │  api.py  │  fetch_tags()  ──►  GET /api/tags
   │          │  fetch_model_show()  ──►  POST /api/show
   │          │  benchmark_model()
   │          │    ├─ benchmark_chat_single_run()  ──►  POST /api/chat (streamed)
   │          │    └─ benchmark_embed_single_run()  ──►  POST /api/embed
   │          │  Returns BenchmarkResult(ttft, tps, error, runs)
   └────┬─────┘
        │
        ▼
   ┌───────────┐
    │ display.py│  stream_table()
    │           │    ├─ asyncio.Semaphore for concurrency gating
    │           │    ├─ Live table updates as tasks complete
    │           │    ├─ console.status progress in export mode
    │           │    ├─ _collect_pending() shared async collection
    │           │    └─ _build_colored_table() with percentile thresholds
    │           │  process_single_model(export_only=True)  ──►  skips display row, returns ExportRow only
   └──────┬────┘
          │
          ▼
   ┌───────────┐
   │ export.py │  format_json()  ──►  JSON array output
   │           │  format_csv()   ──►  CSV with header row
   │           │  export_results()  ──►  stdout or file
   └───────────┘
```

## Key Data Entities

### BenchmarkResult

```txt
BenchmarkResult
├── ttft: float | None         # Average time-to-first-token across runs
├── tps: float | None          # Average tokens-per-second across runs
├── error: str | None          # First error encountered, if any
└── runs: list[dict]           # Per-run details (prompt, ttft, tps, error)
```

Defined in `src/ometer/api.py:14-19`.

### ExportRow

```txt
ExportRow
├── model: str              # Model name (e.g. "llama3:latest")
├── size: str               # Parameter size (e.g. "8B")
├── context: str            # Context length (e.g. "8192")
├── quant: str              # Quantization level (e.g. "Q4_0")
├── capabilities: str        # Capabilities (e.g. "completion, vision")
├── ttft: float | None      # Average time-to-first-token
├── tps: float | None       # Average tokens-per-second
├── error: str | None       # First error, if any
└── runs: list[dict]        # Per-run {"prompt", "ttft", "tps", "error"}
```

Defined in `src/ometer/export.py:10-20`.

### Config

```txt
Config
├── local_base_url: str            # OLLAMA_LOCAL_BASE_URL (default: http://localhost:11434)
├── cloud_base_url: str            # OLLAMA_CLOUD_BASE_URL (default: https://ollama.com)
├── cloud_api_key: str             # OLLAMA_CLOUD_API_KEY
├── num_runs: int                  # Clamped 1–3
├── bench_prompts_active: list     # Subset of _BENCH_PROMPTS[:num_runs]
└── num_parallel: int              # Clamped 1–10
```

Defined in `src/ometer/config.py:25-39`.

### CLI Arguments

Handled by `build_parser()` in `src/ometer/cli.py`:

| Flag         | Type                  | Description                                   |
| ------------ | --------------------- | --------------------------------------------- |
| `--version`  |                       | Show version and exit                         |
| `--local`    | bool                  | Show only local models                        |
| `--cloud`    | bool                  | Show only cloud models                        |
| `--model`    | list[str] (`nargs=+`) | Filter models by name (exact or family match) |
| `--ttft`     | bool                  | Benchmark time-to-first-token                 |
| `--tps`      | bool                  | Benchmark tokens-per-second                   |
| `--verbose`  | bool                  | Show per-run breakdown                        |
| `--runs`     | int (1–3)             | Number of benchmark prompts per model         |
| `--parallel` | int (1–10)            | Concurrent model benchmarks                   |
| `--json`     | optional path         | Export results as JSON (stdout if no path)    |
| `--csv`      | optional path         | Export results as CSV (stdout if no path)     |

`--json` and `--csv` are mutually exclusive.

## Entry Points

Two CLI commands are registered in `pyproject.toml`:

```txt
ometer      ──►  ometer.cli:main_entrypoint
ollamameter ──►  ometer.cli:main_entrypoint
```

The module can also be invoked via `python -m ometer`, handled by `src/ometer/__main__.py`.

## Subsystem Interaction Flow

```txt
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐     ┌───────────┐
│  User   │────►│  cli.py  │────►│config.py │────►│  api.py   │────►│display.py │
│         │     │          │     │          │     │           │     │           │
│  flags  │     │ resolve  │     │ from_env │     │ fetch_tags│     │ live table│
│  or     │     │ _mode    │     │          │     │ benchmark │     │ semaphore │
│  menu   │     │          │     │          │     │ _model    │     │ export    │
└─────────┘     └────┬─────┘     └──────────┘     └─────┬─────┘     └─────┬─────┘
                     │                                  │                 │
                     │              ┌───────────┐       │                 │
                     └─────────────►│export.py  │◄──────┘                 │
                                    │           │─────────────────────────┘
                                    │ JSON/CSV  │
                                    └───────────┘
```
