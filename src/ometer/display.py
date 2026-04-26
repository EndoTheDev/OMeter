from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ometer.api import BenchmarkResult, benchmark_model, fetch_model_show
from ometer.config import Config

console = Console()


def extract_context_length(model_info: dict[str, Any]) -> int:
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            return value
    return 0


def format_size(parameter_size: str | None, model_name: str) -> str:
    if parameter_size:
        try:
            size_int = int(parameter_size)
            if size_int >= 1_000_000_000_000:
                return f"{size_int // 1_000_000_000_000}T"
            if size_int >= 1_000_000_000:
                return f"{size_int // 1_000_000_000}B"
            if size_int >= 1_000_000:
                return f"{size_int // 1_000_000}M"
            return str(size_int)
        except (ValueError, TypeError):
            pass

        m = re.match(r"(\d+(?:\.\d+)?)\s*([BKMGT])", str(parameter_size), re.IGNORECASE)
        if m:
            return f"{m.group(1)}{m.group(2).upper()}"

    m = re.search(r"(\d+(?:\.\d+)?)\s*[BKMGT]", model_name, re.IGNORECASE)
    if m:
        return m.group(0).upper()

    return "0B"


def format_capabilities(caps: list[str]) -> str:
    return ", ".join(sorted(caps))


def format_float_or_na(val: float | None) -> str:
    if val is None:
        return "n/a"
    return f"{val:.2f}"


def build_table(
    title: str, show_ttft: bool, show_tps: bool, verbose: bool, num_runs: int
) -> Table:
    table = Table(title=title, title_style="bold cyan")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right", style="green")
    table.add_column("Context", justify="right", style="yellow")
    table.add_column("Quant", style="magenta")
    table.add_column("Capabilities", style="white")
    if show_ttft:
        if verbose:
            for i in range(1, num_runs + 1):
                table.add_column(f"TTFT{i}", justify="right", style="blue")
        table.add_column("TTFT", justify="right", style="blue")
    if show_tps:
        if verbose:
            for i in range(1, num_runs + 1):
                table.add_column(f"TPS{i}", justify="right", style="bright_red")
        table.add_column("TPS", justify="right", style="bright_red")
    return table


def _column_indices(
    show_ttft: bool, show_tps: bool, verbose: bool, num_runs: int
) -> tuple[list[int], list[int]]:
    ttft_indices: list[int] = []
    tps_indices: list[int] = []
    idx = 5
    if show_ttft:
        if verbose:
            ttft_indices.extend(range(idx, idx + num_runs))
            idx += num_runs
        ttft_indices.append(idx)
        idx += 1
    if show_tps:
        if verbose:
            tps_indices.extend(range(idx, idx + num_runs))
            idx += num_runs
        tps_indices.append(idx)
        idx += 1
    return ttft_indices, tps_indices


def _parse_value(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def _thresholds(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    low_idx = max(0, n // 3 - 1)
    high_idx = min(n - 1, 2 * n // 3)
    return sorted_vals[low_idx], sorted_vals[high_idx]


def _color(
    cell: str, thresholds: tuple[float, float] | None, lower_is_better: bool
) -> Text:
    if cell in ("err", "n/a"):
        return Text(cell, style="red")
    val = _parse_value(cell)
    if val is None or thresholds is None:
        return Text(cell)
    low, high = thresholds
    if lower_is_better:
        if val <= low:
            style = "green"
        elif val >= high:
            style = "red"
        else:
            style = "orange3"
    else:
        if val >= high:
            style = "green"
        elif val <= low:
            style = "red"
        else:
            style = "orange3"
    return Text(cell, style=style)


def _build_colored_table(
    title: str,
    show_ttft: bool,
    show_tps: bool,
    verbose: bool,
    num_runs: int,
    rows: list[list[str]],
) -> Table:
    table = build_table(title, show_ttft, show_tps, verbose, num_runs)
    ttft_indices, tps_indices = _column_indices(show_ttft, show_tps, verbose, num_runs)

    ttft_values: list[float] = []
    tps_values: list[float] = []
    for row in rows:
        for i in ttft_indices:
            v = _parse_value(row[i])
            if v is not None:
                ttft_values.append(v)
        for i in tps_indices:
            v = _parse_value(row[i])
            if v is not None:
                tps_values.append(v)

    ttft_thresholds = _thresholds(ttft_values)
    tps_thresholds = _thresholds(tps_values)

    for row in rows:
        styled: list[str | Text] = list(row)
        for i in ttft_indices:
            styled[i] = _color(row[i], ttft_thresholds, lower_is_better=True)
        for i in tps_indices:
            styled[i] = _color(row[i], tps_thresholds, lower_is_better=False)
        table.add_row(*styled)

    return table


def process_single_model(
    tag_model: dict[str, Any],
    show_data: dict[str, Any],
    benchmark: BenchmarkResult,
    show_ttft: bool,
    show_tps: bool,
    verbose: bool,
    num_runs: int,
) -> list[str]:
    model_name = tag_model["name"]
    tag_details = tag_model.get("details", {})
    show_details = show_data.get("details", {})
    details: dict[str, Any] = {}
    details.update(tag_details)
    details.update({k: v for k, v in show_details.items() if v})

    model_info = show_data.get("model_info", {})
    capabilities = show_data.get("capabilities", [])

    size = format_size(details.get("parameter_size"), model_name)
    context = str(extract_context_length(model_info))
    quant = details.get("quantization_level", "")
    caps = format_capabilities(capabilities) if capabilities else ""

    row = [model_name, size, context, quant, caps]

    runs = benchmark.runs

    if show_ttft:
        if verbose:
            for i in range(num_runs):
                if i < len(runs):
                    err = runs[i].get("error")
                    ttft = runs[i]["ttft"]
                else:
                    err = ""
                    ttft = None
                if err:
                    row.append("err")
                else:
                    row.append(format_float_or_na(ttft))
        row.append(format_float_or_na(benchmark.ttft))
    if show_tps:
        if verbose:
            for i in range(num_runs):
                if i < len(runs):
                    err = runs[i].get("error")
                    tps = runs[i]["tps"]
                else:
                    err = ""
                    tps = None
                if err:
                    row.append("err")
                else:
                    row.append(format_float_or_na(tps))
        row.append(format_float_or_na(benchmark.tps))
    return row


async def _benchmark_model_task(
    idx: int,
    model: dict[str, Any],
    show_result: dict[str, Any] | BaseException,
    client: httpx.AsyncClient,
    config: Config,
    base_url: str,
    show_ttft: bool,
    show_tps: bool,
    verbose: bool,
    chat_headers: dict[str, str] | None,
    semaphore: asyncio.Semaphore,
) -> tuple[int, list[str], list[str]]:
    show_data: dict[str, Any] = {}
    errors: list[str] = []
    if isinstance(show_result, BaseException):
        errors.append(f"{model['name']} /api/show failed: {show_result}")
    else:
        show_data = show_result  # type: ignore

    bench = BenchmarkResult(ttft=None, tps=None, error=None)
    if show_ttft or show_tps:
        async with semaphore:
            bench = await benchmark_model(
                client, config, base_url, model["name"], show_data, chat_headers
            )
        if bench.error:
            errors.append(f"{model['name']}: {bench.error}")

    row = process_single_model(
        model, show_data, bench, show_ttft, show_tps, verbose, config.num_runs
    )
    return idx, row, errors


async def stream_table(
    client: httpx.AsyncClient,
    config: Config,
    base_url: str,
    models: list[dict[str, Any]],
    title: str,
    show_ttft: bool,
    show_tps: bool,
    verbose: bool,
    chat_headers: dict[str, str] | None = None,
) -> None:
    table = build_table(title, show_ttft, show_tps, verbose, config.num_runs)

    semaphore = asyncio.Semaphore(config.num_parallel)

    async def _throttled_show(m: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await fetch_model_show(client, base_url, m["name"])

    show_tasks = [asyncio.create_task(_throttled_show(m)) for m in models]
    with console.status(f"Fetching details for {len(models)} model(s)…"):
        show_results = await asyncio.gather(*show_tasks, return_exceptions=True)

    pending: set[asyncio.Task[tuple[int, list[str], list[str]]]] = set()
    for idx, (model, show_result) in enumerate(zip(models, show_results)):
        task = asyncio.create_task(
            _benchmark_model_task(
                idx,
                model,
                show_result,
                client,
                config,
                base_url,
                show_ttft,
                show_tps,
                verbose,
                chat_headers,
                semaphore,
            )
        )
        pending.add(task)

    completed_rows: dict[int, list[str]] = {}
    ordered_rows: list[list[str]] = []
    next_row = 0
    total = len(models)
    is_benchmarking = show_ttft or show_tps

    bench_errors: list[str] = []
    if is_benchmarking:
        spinner = Spinner("dots", f"Benchmarking 0/{total} model(s)…")
        live_renderable = Group(table, spinner)
    else:
        spinner = None
        live_renderable = table
    with Live(live_renderable, console=console, refresh_per_second=4) as live:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                idx, row, errors = task.result()
                bench_errors.extend(errors)
                completed_rows[idx] = row

            while next_row in completed_rows:
                row = completed_rows.pop(next_row)
                table.add_row(*row)
                ordered_rows.append(row)
                next_row += 1

            if spinner:
                spinner.update(
                    text=(
                        f"Benchmarking {next_row}/{total} model(s)…"
                        if pending
                        else f"Completed {next_row}/{total}"
                    )
                )
                live.update(Group(table, spinner))
            else:
                live.update(table)

        if ordered_rows and (show_ttft or show_tps):
            final_table = _build_colored_table(
                title, show_ttft, show_tps, verbose, config.num_runs, ordered_rows
            )
            live.update(final_table)

    for err in bench_errors:
        console.print(f"[yellow]⚠ {err}[/yellow]")
