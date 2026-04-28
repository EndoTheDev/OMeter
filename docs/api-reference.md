# API Reference

This document covers the Ollama API endpoints used by OMeter and the internal functions that interact with them.

## Ollama API Endpoints

OMeter communicates with Ollama instances over HTTP using `httpx.AsyncClient`.

### GET /api/tags

Lists available models on an endpoint.

- **Used by**: `fetch_tags()` (`src/ometer/api.py:35-39`)
- **Response**: `{ "models": [{ "name": "...", "modified_at": "...", "details": { ... } }] }`
- **Purpose**: Discover which models are available on the local or cloud instance.

### POST /api/show

Retrieves detailed metadata for a specific model.

- **Used by**: `fetch_model_show()` (`src/ometer/api.py:42-51`)
- **Request body**: `{ "model": "<model_name>" }`
- **Response**: `{ "details": { ... }, "model_info": { ... }, "capabilities": ["chat", "embedding", ...] }`
- **Purpose**: Extract parameter size, quantization level, context length, and capabilities.

### POST /api/chat (streamed)

Streams chat completions for benchmarking LLMs.

- **Used by**: `benchmark_chat_single_run()` (`src/ometer/api.py:58-132`)
- **Request body**: `{ "model": "...", "messages": [{"role": "user", "content": "..."}], "stream": true }`
- **Auth**: Bearer token in `Authorization` header for cloud endpoints.
- **Timeout**: 300 seconds.
- **Response**: Newline-delimited JSON chunks. The final `done: true` chunk contains `eval_count` and `eval_duration`.

### POST /api/embed

Benchmarks embedding models.

- **Used by**: `benchmark_embed_single_run()` (`src/ometer/api.py:135-169`)
- **Request body**: `{ "model": "...", "input": "..." }`
- **Response**: `{ "prompt_eval_count": ..., "total_duration": ... }`
- **Timeout**: 300 seconds.
- **Note**: Only used for embedding models on **local** endpoints. Cloud embedding is not supported.

## Internal Functions

### api.py

| Function                                                                              | Lines   | Returns           | Description                                      |
| ------------------------------------------------------------------------------------- | ------- | ----------------- | ------------------------------------------------ |
| `fetch_tags(client, base_url)`                                                        | 35-39   | `list[dict]`      | Fetches model list from `/api/tags`              |
| `fetch_model_show(client, base_url, model_name)`                                      | 42-51   | `dict`            | Fetches model metadata from `/api/show`          |
| `is_embedding_model(show_data)`                                                       | 54-55   | `bool`            | Checks if `"embedding"` is in capabilities       |
| `benchmark_chat_single_run(client, base_url, model_name, prompt, headers, show_data)` | 58-132  | `dict`            | Single benchmark run via `/api/chat`             |
| `benchmark_embed_single_run(client, base_url, model_name, prompt, headers)`           | 135-169 | `dict`            | Single benchmark run via `/api/embed`            |
| `benchmark_model(client, config, base_url, model_name, show_data, headers)`           | 172-217 | `BenchmarkResult` | Runs all prompts, averages results               |
| `sort_by_modified(models)`                                                            | 22-32   | `list[dict]`      | Sorts model list by `modified_at` (newest first) |

### config.py

| Function/Class                    | Lines | Description                                               |
| --------------------------------- | ----- | --------------------------------------------------------- |
| `_load_env()`                     | 15-23 | Searches for `.env` files in 3 locations                  |
| `Config`                          | 25-39 | Settings class with clamped `num_runs` and `num_parallel` |
| `Config.from_env(runs, parallel)` | 41-67 | Factory: loads env, applies CLI overrides                 |

### cli.py

| Function                                                                                                  | Lines   | Description                                                |
| --------------------------------------------------------------------------------------------------------- | ------- | ---------------------------------------------------------- |
| `main(mode, show_ttft, show_tps, verbose, target_models, config, export_fmt, export_path, sort, reverse)` | 19-140  | Async main: fetches models, dispatches to display/export   |
| `match_model(model_name, target)`                                                                         | 143-152 | Exact or family-prefix model name matching                 |
| `build_parser(prog)`                                                                                      | 155-211 | Builds `argparse.ArgumentParser` with all flags            |
| `resolve_mode(args, is_tty, prompt_fn)`                                                                   | 214-232 | Determines local/cloud/both from flags or interactive menu |
| `main_entrypoint()`                                                                                       | 235-287 | Synchronous entry point registered in `pyproject.toml`     |

### display.py

| Function                                                                     | Lines   | Description                                                 |
| ---------------------------------------------------------------------------- | ------- | ----------------------------------------------------------- |
| `SortSpec.parse(raw, reverse=False)`                                         | 34-45   | Parses sort string into structured field + direction        |
| `_size_value(size_str)`                                                      | 53-61   | Extracts numeric parameter value from size string           |
| `_context_value(context_str)`                                                | 64-69   | Parses context length string to integer                     |
| `_modified_value(modified_str)`                                              | 71-76   | Parses ISO date string to datetime                          |
| `_sort_key(spec, export)`                                                    | 79-90   | Returns sort key for a single ExportRow                     |
| `sort_results(rows, exports, spec)`                                          | 93-103  | Sorts paired rows and exports in-place                      |
| `extract_context_length(model_info)`                                         | 107-110 | Finds `*.context_length` key in model info                  |
| `format_size(parameter_size, model_name)`                                    | 114-137 | Formats parameter count (e.g. `7000000000` → `7B`)          |
| `format_capabilities(caps)`                                                  | 140-141 | Joins sorted capabilities list                              |
| `format_float_or_na(val)`                                                    | 144-147 | Formats float or returns `"n/a"`                            |
| `build_table(title, show_ttft, show_tps, verbose, num_runs)`                 | 150-169 | Creates `rich.Table` with appropriate columns               |
| `_thresholds(values)`                                                        | 200-207 | Computes 33rd/66th percentile boundaries                    |
| `_color(cell, thresholds, lower_is_better)`                                  | 210-233 | Applies green/orange/red styling                            |
| `_build_colored_table(...)`                                                  | 236-271 | Rebuilds table with threshold-based coloring                |
| `process_single_model(..., export_only=False)`                               | 275-343 | Merges tag + show data + benchmark into row and ExportRow   |
| `_benchmark_model_task(..., export_only=False)`                              | 346-385 | Async task: fetch show data, run benchmark, produce row     |
| `_collect_pending(pending, completed_rows, completed_exports, bench_errors)` | 388-398 | Awaits next completed task, collects results into dicts     |
| `stream_table(..., export_only=False)`                                       | 401-537 | Orchestrates live table rendering or export-only collection |

### export.py

| Function                                                                  | Lines   | Description                                                           |
| ------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------- |
| `format_json(rows, num_runs, show_ttft, show_tps, verbose)`               | 23-52   | Formats results as JSON array string; `error` key omitted when `None` |
| `format_csv(rows, num_runs, show_ttft, show_tps, verbose)`                | 55-98   | Formats results as CSV string with headers                            |
| `export_results(rows, fmt, path, num_runs, show_ttft, show_tps, verbose)` | 101-118 | Dispatches to format_json/format_csv and writes output                |

## BenchmarkResult Dataclass

```python
@dataclass
class BenchmarkResult:
    ttft: float | None     # Average time-to-first-token
    tps: float | None      # Average tokens-per-second
    error: str | None       # First error, if any
    runs: list[dict]       # Per-run {"prompt", "ttft", "tps", "error"}
```

Defined at `src/ometer/api.py:14-19`.

## ExportRow Dataclass

```python
@dataclass
class ExportRow:
    model: str              # Model name
    size: str               # Parameter size (e.g. "8B")
    context: str            # Context length (e.g. "8192")
    quant: str              # Quantization level (e.g. "Q4_0")
    capabilities: str       # Comma-separated capabilities
    ttft: float | None      # Average time-to-first-token
    tps: float | None       # Average tokens-per-second
    error: str | None       # First error, if any
    runs: list[dict]        # Per-run {"prompt", "ttft", "tps", "error"}
    modified_at: str        # ISO timestamp from /api/tags
```

Defined at `src/ometer/export.py:10-21`.

## Table Column Layout

The order and visibility of columns depends on which flags are active:

| Column       | Always | With `--ttft` | With `--tps` | With `--verbose` |
| ------------ | ------ | :-----------: | :----------: | :--------------: |
| Model        | Yes    |               |              |                  |
| Size         | Yes    |               |              |                  |
| Context      | Yes    |               |              |                  |
| Quant        | Yes    |               |              |                  |
| Capabilities | Yes    |               |              |                  |
| TTFT1…TTFTn  |        |      Yes      |              |       Yes        |
| TTFT         |        |      Yes      |              |                  |
| TPS1…TPSn    |        |               |     Yes      |       Yes        |
| TPS          |        |               |     Yes      |                  |
