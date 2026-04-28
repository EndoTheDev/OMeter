# Configuration

OMeter uses environment variables for all runtime configuration, with CLI flags taking precedence when provided.

## Environment Variable Loading

The `_load_env()` function in `src/ometer/config.py:15-23` searches for `.env` files in order and stops at the first one found:

```txt
1. ./.env                  # Current working directory (project-specific)
2. ~/.env                  # Home directory (global fallback)
3. ~/.config/ometer/.env   # XDG config directory (recommended for global installs)
```

## Environment Variable Reference

| Variable                | Default                  | Description                                         |
| ----------------------- | ------------------------ | --------------------------------------------------- |
| `OLLAMA_LOCAL_BASE_URL` | `http://localhost:11434` | URL for the local Ollama instance                   |
| `OLLAMA_CLOUD_BASE_URL` | `https://ollama.com`     | URL for the cloud Ollama instance                   |
| `OLLAMA_CLOUD_API_KEY`  | `""`                     | Bearer token for cloud endpoint authentication      |
| `OMETER_RUNS`           | `3`                      | Number of benchmark prompts per model (clamped 1–3) |
| `OMETER_PARALLEL`       | `1`                      | Max concurrent model benchmarks (clamped 1–10)      |

### Setting Up a Config File

```bash
mkdir -p ~/.config/ometer
cat > ~/.config/ometer/.env << 'EOF'
OLLAMA_CLOUD_BASE_URL=https://ollama.com
OLLAMA_CLOUD_API_KEY=your_api_key_here
OLLAMA_LOCAL_BASE_URL=http://localhost:11434
OMETER_RUNS=3
OMETER_PARALLEL=1
EOF
```

The cloud API key is **only needed for benchmarking cloud models**. Local model listing works without it.

## CLI Flag Reference

CLI flags are parsed by `build_parser()` in `src/ometer/cli.py`. When provided, they override the corresponding environment variable values.

| Flag         | Overrides         | Type                  | Description                                          |
| ------------ | ----------------- | --------------------- | ---------------------------------------------------- |
| `--version`  | N/A               |                       | Show version and exit                                |
| `--local`    | N/A (mode)        | bool                  | Show only local models                               |
| `--cloud`    | N/A (mode)        | bool                  | Show only cloud models                               |
| `--model`    | N/A (filter)      | list[str] (`nargs=+`) | Filter models by name (exact or family match)        |
| `--sort`     | N/A (order)       | str                   | Sort by field (name, modified, ttft, tps, size, ctx) |
| `--reverse`  | N/A (order)       | bool                  | Reverse the sort order (requires `--sort`)           |
| `--ttft`     | N/A (metric)      | bool                  | Enable time-to-first-token benchmarking              |
| `--tps`      | N/A (metric)      | bool                  | Enable tokens-per-second benchmarking                |
| `--verbose`  | N/A (display)     | bool                  | Show per-run breakdown in output table               |
| `--runs`     | `OMETER_RUNS`     | int (1–3)             | Number of benchmark prompts per model                |
| `--parallel` | `OMETER_PARALLEL` | int (1–10)            | Number of models benchmarked concurrently            |
| `--json`     | N/A (export)      | optional path         | Export results as JSON (stdout if no path)           |
| `--csv`      | N/A (export)      | optional path         | Export results as CSV (stdout if no path)            |

`--json` and `--csv` are mutually exclusive.

## Flag Precedence

```txt
CLI flags  >  Environment variables  >  Built-in defaults

--runs 3          ──► overrides OMETER_RUNS
--parallel 4      ──► overrides OMETER_PARALLEL
--local           ──► sets mode, no env var equivalent
--cloud           ──► sets mode, no env var equivalent
--model llama3    ──► filters to matching model name(s)
--json            ──► prints JSON to stdout
--json out.json   ──► writes JSON to file
--csv             ──► prints CSV to stdout
--csv out.csv     ──► writes CSV to file
```

## Mode Resolution

`resolve_mode()` in `src/ometer/cli.py` determines which endpoint to query:

```txt
--local --cloud     ──►  None (query both)
--local            ──►  "local"
--cloud            ──►  "cloud"
(no flags, TTY)    ──►  InquirerPy interactive menu
(no flags, pipe)   ──►  None (query both, no menu)
--model <names>    ──►  None (query both, filter to names)
```

## Config Clamping

The `Config` class (`src/ometer/config.py:25-39`) enforces runtime bounds:

- `num_runs`: Clamped to `[1, 3]` via `max(1, min(3, num_runs))`
- `num_parallel`: Clamped to `[1, 10]` via `max(1, min(10, num_parallel))`
- `bench_prompts_active`: Automatically sliced as `_BENCH_PROMPTS[:num_runs]`
