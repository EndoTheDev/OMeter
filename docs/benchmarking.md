# Benchmarking Pipeline

The benchmarking pipeline is an asynchronous workflow that discovers models, measures performance, and renders results with real-time updates.

## Pipeline Stages

```txt
1. Discovery    ──►  fetch tags from local and/or cloud endpoints
2. Inspection   ──►  fetch model metadata via /api/show (size, quant, capabilities)
3. Dispatch     ──►  schedule benchmark tasks with asyncio.Semaphore
4. Measurement  ──►  stream responses, compute TTFT and TPS per run
5. Aggregation  ──►  average results across runs, compute percentile thresholds
6. Rendering    ──►  build colorized rich.Table, display via Live context
```

## Measurement Methodology

### TTFT — Time-To-First-Token

Measures latency from request start to the first meaningful content token.

**Chat models** (`src/ometer/api.py:58-132`):

1. Records `start = time.perf_counter()` before the request.
2. Streams the response from `POST /api/chat`.
3. On each chunk, checks for `message.content` (or `message.thinking` for thinking models).
4. Sets `first_token_time` on the first non-empty content/thinking field.
5. Final TTFT = `first_token_time - start` (seconds).

**Thinking model detection** (`src/ometer/api.py:80`):

Models with `"thinking"` in their `capabilities` list have their TTFT measured from the first thinking or content token, not just content. This ensures accurate latency measurement for models like DeepSeek-R1.

**Embedding models** (`src/ometer/api.py:135-169`):

Since embedding requests are not streamed, TTFT for embedding models is the total request duration: `time.perf_counter() - start`.

### TPS — Tokens-Per-Second

Measures generation throughput.

**Chat models**: Derived from the final `done` chunk in the Ollama stream:

```txt
tps = eval_count / (eval_duration / 1e9)
```

If `eval_duration` is unavailable, falls back to `total_duration`.

**Embedding models**: Uses `prompt_eval_count` and `total_duration` from the `/api/embed` response:

```txt
tps = prompt_eval_count / (total_duration / 1e9)
```

### Error Handling

- If the stream ends without a `done` chunk, `error` is set to `"Stream ended without completion"`.
- Errors from individual runs are collected into the `BenchmarkResult.error` field (first error wins).
- Runs with errors are excluded from the averaged TTFT/TPS.
- If all runs fail, `ttft` and `tps` are `None`.

## Multi-Prompt Averaging

Each model is tested against up to 3 distinct prompts defined in `_BENCH_PROMPTS` (`src/ometer/config.py:8-12`):

```txt
_BENCH_PROMPTS = [
    "why is the sky blue?",
    "explain quantum computing in one paragraph",
    "write a haiku about rain",
]
```

The `Config.num_runs` value (clamped 1–3) controls how many prompts are used. Results are averaged across successful runs to reduce variance from network jitter or hardware fluctuations.

## Concurrency Model

Parallel benchmarking is controlled by `asyncio.Semaphore` in `display.py`:

```txt
semaphore = asyncio.Semaphore(config.num_parallel)
```

- `OMETER_PARALLEL` / `--parallel` sets concurrency (default 1, max 10).
- The semaphore gates both `/api/show` requests and benchmark runs.
- `asyncio.wait(FIRST_COMPLETED)` is used to update the live table as each model finishes.
- The shared async collection logic is extracted into `_collect_pending()`, which awaits the next completed task and stores results into dicts.

## Sorting

After all models are benchmarked and collected, results are sorted via `sort_results()` in `display.py`.

Supported sort fields (via `--sort`):

| Field      | Default order        | Reverse (`--reverse`) |
| ---------- | -------------------- | --------------------- |
| `name`     | A–Z ascending        | Z–A descending        |
| `modified` | Newest first         | Oldest first          |
| `ttft`     | Lowest (best) first  | Highest (worst) first |
| `tps`      | Highest (best) first | Lowest (worst) first  |
| `size`     | Largest first        | Smallest first        |
| `ctx`      | Largest first        | Smallest first        |

Sorting applies in both live display mode and export-only mode.

## Export Mode

When `--json` or `--csv` is used, `stream_table()` runs in export-only mode (`export_only=True`):

- The rich `Live` table is skipped — no table is rendered to the terminal.
- `process_single_model()` skips building display rows when `export_only=True`, returning an empty list for the display row and only populating the `ExportRow`.
- If benchmarks are active (`--ttft` or `--tps`), a `console.status` spinner shows progress: `"Benchmarking 2/5 model(s)…"`.
- On completion, results are written to stdout or a file via `export_results()`.
- For list-only mode (no benchmarks), only the existing `"Fetching details for N model(s)…"` status is shown.
- When sorting is active (`--sort`), the final export is sorted before outputting.

## Color Thresholds

After all models are benchmarked, `_build_colored_table()` computes percentile-based thresholds:

1. `_thresholds()` calculates the 33rd and 66th percentile boundaries.
2. `_color()` applies styling based on whether lower or higher values are better:

| Metric | Lower is better? |    Green    |   Orange   |     Red     |
| ------ | :--------------: | :---------: | :--------: | :---------: |
| TTFT   |       Yes        | Lowest 33%  | Middle 33% | Highest 33% |
| TPS    |        No        | Highest 33% | Middle 33% | Lowest 33%  |

Error values (`"err"`, `"n/a"`) are always colored red.

## Embedding Model Handling

Embedding models are detected via `is_embedding_model()` (`src/ometer/api.py:54-55`):

```python
def is_embedding_model(show_data):
    return "embedding" in show_data.get("capabilities", [])
```

When an embedding model is detected on a **local** endpoint, the benchmark uses `POST /api/embed` instead of `POST /api/chat`. Cloud embedding models are not currently supported (see AGENTS.md).

## Data Flow

```txt
benchmark_model()
│
├─ for each prompt in config.bench_prompts_active:
│   │
│   ├─ is_embedding_model?  ──►  benchmark_embed_single_run()
│   │                             POST /api/embed
│   │                              └─ returns {ttft, tps, error}
│   │
│   └─ else  ──►  benchmark_chat_single_run()
│                  POST /api/chat (streamed)
│                   └─ returns {ttft, tps, error}
│
└─ averages successful runs ──►  BenchmarkResult(ttft, tps, error, runs)
```
