from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ometer.api import BenchmarkResult, benchmark_model, fetch_model_show
from ometer.config import Config
from ometer.export import ExportRow

console = Console()


SORT_FIELDS: dict[str, bool] = {
    "name": True,
    "modified": False,
    "ttft": True,
    "tps": False,
    "size": False,
    "ctx": False,
}


@dataclass(frozen=True)
class SortSpec:
    field: str
    ascending: bool

    @classmethod
    def parse(cls, raw: str | None, reverse: bool = False) -> SortSpec | None:
        if raw is None:
            return None
        field = raw.strip()
        if field not in SORT_FIELDS:
            raise ValueError(
                f"Unknown sort field '{field}'. "
                f"Expected: {', '.join(SORT_FIELDS)}"
            )
        natural_asc = SORT_FIELDS[field]
        ascending = not natural_asc if reverse else natural_asc
        return cls(field=field, ascending=ascending)


def _size_value(size_str: str) -> float:
    m = re.match(r"(\d+(?:\.\d+)?)\s*([BKMGT])", size_str, re.IGNORECASE)
    if m:
        multipliers = {"B": 1e9, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
        return float(m.group(1)) * multipliers.get(m.group(2).upper(), 1)
    try:
        return float(size_str)
    except ValueError:
        return 0.0


def _context_value(context_str: str) -> int:
    try:
        return int(context_str)
    except ValueError:
        return 0


def _modified_value(modified_str: str) -> datetime:
    iso = modified_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _sort_key(spec: SortSpec, export: ExportRow) -> Any:
    if spec.field == "modified":
        return _modified_value(export.modified_at)
    if spec.field == "size":
        return _size_value(export.size)
    if spec.field == "ctx":
        return _context_value(export.context)
    if spec.field == "ttft":
        return export.ttft if export.ttft is not None else float("inf")
    if spec.field == "tps":
        return export.tps if export.tps is not None else float("-inf")
    return export.model.lower()


def sort_results(
    rows: list[list[str]],
    exports: list[ExportRow],
    spec: SortSpec | None,
) -> tuple[list[list[str]], list[ExportRow]]:
    if spec is None:
        return rows, exports
    paired = sorted(
        zip(rows, exports), key=lambda pair: _sort_key(spec, pair[1]), reverse=not spec.ascending
    )
    new_rows, new_exports = zip(*paired) if paired else ([], [])
    return list(new_rows), list(new_exports)


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
    table = Table(title=title, title_style="bold cyan", show_lines=True)
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

    for idx, row in enumerate(rows):
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
    export_only: bool = False,
) -> tuple[list[str], ExportRow]:
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

    export_row = ExportRow(
        model=model_name,
        size=size,
        context=context,
        quant=quant,
        capabilities=caps,
        ttft=benchmark.ttft,
        tps=benchmark.tps,
        error=benchmark.error,
        runs=benchmark.runs,
        modified_at=tag_model.get("modified_at", ""),
    )

    if export_only:
        return [], export_row

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

    return row, export_row


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
    export_only: bool = False,
) -> tuple[int, list[str], ExportRow, list[str]]:
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

    row, export_row = process_single_model(
        model,
        show_data,
        bench,
        show_ttft,
        show_tps,
        verbose,
        config.num_runs,
        export_only=export_only,
    )
    return idx, row, export_row, errors


async def _collect_pending(
    pending: set[asyncio.Task[tuple[int, list[str], ExportRow, list[str]]]],
    completed_rows: dict[int, list[str]],
    completed_exports: dict[int, ExportRow],
    bench_errors: list[str],
) -> None:
    done, still_pending = await asyncio.wait(
        pending, return_when=asyncio.FIRST_COMPLETED
    )
    pending.clear()
    pending.update(still_pending)
    for task in done:
        idx, row, export_row, errors = task.result()
        bench_errors.extend(errors)
        completed_rows[idx] = row
        completed_exports[idx] = export_row


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
    export_only: bool = False,
    sort_spec: SortSpec | None = None,
) -> list[ExportRow]:
    table = build_table(title, show_ttft, show_tps, verbose, config.num_runs)

    semaphore = asyncio.Semaphore(config.num_parallel)

    async def _throttled_show(m: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await fetch_model_show(client, base_url, m["name"])

    show_tasks = [asyncio.create_task(_throttled_show(m)) for m in models]
    with console.status(f"Fetching details for {len(models)} model(s)…"):
        show_results = await asyncio.gather(*show_tasks, return_exceptions=True)

    pending: set[asyncio.Task[tuple[int, list[str], ExportRow, list[str]]]] = set()
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
                export_only=export_only,
            )
        )
        pending.add(task)

    completed_rows: dict[int, list[str]] = {}
    completed_exports: dict[int, ExportRow] = {}
    ordered_rows: list[list[str]] = []
    ordered_exports: list[ExportRow] = []
    next_row = 0
    total = len(models)
    is_benchmarking = show_ttft or show_tps

    bench_errors: list[str] = []
    spinner: Spinner | None = None
    live_renderable: Table | Group = table

    if not export_only:
        if is_benchmarking:
            spinner = Spinner("dots", f"Benchmarking 0/{total} model(s)…")
            live_renderable = Group(table, spinner)

    def _process_completed() -> None:
        nonlocal next_row
        while next_row in completed_exports:
            row = completed_rows.pop(next_row)
            export = completed_exports.pop(next_row)
            if not export_only:
                table.add_row(*row)
            ordered_rows.append(row)
            ordered_exports.append(export)
            next_row += 1

    def _progress_text() -> str:
        return (
            f"Benchmarking {next_row}/{total} model(s)…"
            if pending
            else f"Completed {next_row}/{total}"
        )

    if export_only:
        if is_benchmarking:
            with console.status(f"Benchmarking 0/{total} model(s)…") as status:
                while pending:
                    await _collect_pending(
                        pending, completed_rows, completed_exports, bench_errors
                    )
                    _process_completed()
                    status.update(_progress_text())
        else:
            while pending:
                await _collect_pending(
                    pending, completed_rows, completed_exports, bench_errors
                )
                _process_completed()
        ordered_rows, ordered_exports = sort_results(
            ordered_rows, ordered_exports, sort_spec
        )
    else:
        with Live(live_renderable, console=console, refresh_per_second=4) as live:
            while pending:
                await _collect_pending(
                    pending, completed_rows, completed_exports, bench_errors
                )
                _process_completed()
                if spinner:
                    spinner.update(text=_progress_text())
                    live.update(Group(table, spinner))
                else:
                    live.update(table)
            ordered_rows, ordered_exports = sort_results(
                ordered_rows, ordered_exports, sort_spec
            )
            if ordered_rows:
                if show_ttft or show_tps:
                    final_table = _build_colored_table(
                        title,
                        show_ttft,
                        show_tps,
                        verbose,
                        config.num_runs,
                        ordered_rows,
                    )
                else:
                    final_table = build_table(
                        title, show_ttft, show_tps, verbose, config.num_runs
                    )
                    for row in ordered_rows:
                        final_table.add_row(*row)
                live.update(final_table)

    for err in bench_errors:
        console.print(f"[yellow]⚠ {err}[/yellow]")

    return ordered_exports
