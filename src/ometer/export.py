from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportRow:
    model: str
    size: str
    context: str
    quant: str
    capabilities: str
    ttft: float | None
    tps: float | None
    error: str | None
    runs: list[dict]
    modified_at: str = ""
    mode: str = ""


def format_json(
    rows: list[ExportRow], num_runs: int, show_ttft: bool, show_tps: bool, verbose: bool
) -> str:
    output = []
    for row in rows:
        entry: dict = {
            "model": row.model,
            "size": row.size,
            "context": row.context,
            "quant": row.quant,
            "capabilities": row.capabilities,
            "modified_at": row.modified_at,
        }
        if show_ttft:
            if verbose:
                for i, run in enumerate(row.runs[:num_runs], 1):
                    entry[f"ttft_run_{i}"] = (
                        run.get("ttft") if run.get("error") is None else None
                    )
            entry["ttft"] = row.ttft
        if show_tps:
            if verbose:
                for i, run in enumerate(row.runs[:num_runs], 1):
                    entry[f"tps_run_{i}"] = (
                        run.get("tps") if run.get("error") is None else None
                    )
            entry["tps"] = row.tps
        if row.error:
            entry["error"] = row.error
        output.append(entry)
    return json.dumps(output, indent=2)


def format_csv(
    rows: list[ExportRow], num_runs: int, show_ttft: bool, show_tps: bool, verbose: bool
) -> str:
    headers = ["model", "size", "context", "quant", "capabilities", "modified_at"]
    if show_ttft:
        if verbose:
            for i in range(1, num_runs + 1):
                headers.append(f"ttft_run_{i}")
        headers.append("ttft")
    if show_tps:
        if verbose:
            for i in range(1, num_runs + 1):
                headers.append(f"tps_run_{i}")
        headers.append("tps")
    headers.append("error")

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()

    for row in rows:
        entry: dict = {
            "model": row.model,
            "size": row.size,
            "context": row.context,
            "quant": row.quant,
            "capabilities": row.capabilities,
            "modified_at": row.modified_at,
        }
        if show_ttft:
            if verbose:
                for i, run in enumerate(row.runs[:num_runs], 1):
                    val = run.get("ttft") if run.get("error") is None else None
                    entry[f"ttft_run_{i}"] = f"{val:.2f}" if val is not None else ""
            entry["ttft"] = f"{row.ttft:.2f}" if row.ttft is not None else ""
        if show_tps:
            if verbose:
                for i, run in enumerate(row.runs[:num_runs], 1):
                    val = run.get("tps") if run.get("error") is None else None
                    entry[f"tps_run_{i}"] = f"{val:.2f}" if val is not None else ""
            entry["tps"] = f"{row.tps:.2f}" if row.tps is not None else ""
        entry["error"] = row.error or ""
        writer.writerow(entry)

    return buf.getvalue()


def export_results(
    rows: list[ExportRow],
    fmt: str,
    path: str | None,
    num_runs: int,
    show_ttft: bool,
    show_tps: bool,
    verbose: bool,
) -> None:
    if fmt == "json":
        content = format_json(rows, num_runs, show_ttft, show_tps, verbose)
    else:
        content = format_csv(rows, num_runs, show_ttft, show_tps, verbose)

    if path:
        Path(path).write_text(content, encoding="utf-8")
    else:
        print(content)


def format_history_json(history_rows: list[dict], verbose: bool) -> str:
    output = []
    for r in history_rows:
        entry: dict = {
            "model": r["model_name"],
            "timestamp": r["timestamp"],
            "size": r["model_size"],
            "context": r["context_length"],
            "quantization": r["quantization"],
            "capabilities": r["capabilities"],
            "mode": r["mode"],
            "ttft": r["ttft"],
            "tps": r["tps"],
            "error": r["error"],
        }
        if verbose:
            entry["prompts"] = r.get("prompts", [])
        output.append(entry)
    return json.dumps(output, indent=2)


def format_history_csv(history_rows: list[dict], verbose: bool) -> str:
    headers = [
        "model",
        "timestamp",
        "size",
        "context",
        "quantization",
        "capabilities",
        "mode",
        "ttft",
        "tps",
        "error",
    ]
    if verbose:
        headers.append("prompts")

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()

    for r in history_rows:
        entry: dict = {
            "model": r["model_name"],
            "timestamp": r["timestamp"],
            "size": r["model_size"] or "",
            "context": r["context_length"] or "",
            "quantization": r["quantization"] or "",
            "capabilities": r["capabilities"] or "",
            "mode": r["mode"] or "",
            "ttft": f'{r["ttft"]:.2f}' if r["ttft"] is not None else "",
            "tps": f'{r["tps"]:.2f}' if r["tps"] is not None else "",
            "error": r["error"] or "",
        }
        if verbose:
            entry["prompts"] = json.dumps(r.get("prompts", []))
        writer.writerow(entry)

    return buf.getvalue()


def export_history(
    history_rows: list[dict],
    fmt: str,
    path: str | None,
    verbose: bool,
) -> None:
    if fmt == "json":
        content = format_history_json(history_rows, verbose)
    else:
        content = format_history_csv(history_rows, verbose)

    if path:
        Path(path).write_text(content, encoding="utf-8")
    else:
        print(content)
