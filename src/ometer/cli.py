from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
from InquirerPy.prompts.list import ListPrompt

from ometer import __version__
from ometer.api import sort_by_modified, fetch_tags
from ometer.config import Config
from ometer.display import (
    console,
    SORT_FIELDS,
    stream_table,
    SortSpec,
    build_history_table,
)
from ometer.export import ExportRow, export_results, export_history
from ometer.history import (
    get_connection,
    get_latest_per_model,
    save_run,
    build_run_data,
)


@dataclass
class MainOptions:
    mode: str | None = None
    show_ttft: bool = False
    show_tps: bool = False
    verbose: bool = False
    target_models: list[str] | None = None
    num_predict: int | None = None
    export_fmt: str | None = None
    export_path: str | None = None
    sort: str | None = None
    reverse: bool = False
    show_history: bool = False


async def main(config: Config, options: MainOptions) -> None:
    if options.show_history:
        conn = get_connection()
        rows = get_latest_per_model(conn)
        conn.close()
        if options.target_models:
            rows = [
                r
                for r in rows
                if any(match_model(r["model_name"], m) for m in options.target_models)
            ]
        if options.export_fmt:
            export_history(
                rows, options.export_fmt, options.export_path, verbose=options.verbose
            )
        elif rows:
            build_history_table(rows, verbose=options.verbose)
        else:
            console.print("[dim]No history found.[/dim]")
        return

    export_only = options.export_fmt is not None
    chat_headers: dict[str, str] | None = None
    if config.cloud_api_key:
        chat_headers = {"Authorization": f"Bearer {config.cloud_api_key}"}

    async with httpx.AsyncClient() as client:
        local_models: list[dict] = []
        sort_spec = SortSpec.parse(options.sort, reverse=options.reverse)
        if options.mode in (None, "local"):
            try:
                with console.status("Fetching local models…"):
                    local_models = sort_by_modified(
                        await fetch_tags(client, config.local_base_url)
                    )
            except Exception as e:
                console.print(f"[yellow]⚠ Skipping local Ollama ({e}).[/yellow]")

        if (
            (options.show_ttft or options.show_tps)
            and options.mode in (None, "cloud")
            and not config.cloud_api_key
        ):
            console.print(
                "[yellow]⚠ OLLAMA_CLOUD_API_KEY not found. Cloud benchmarking will fail without it.\n"
                "  Create a .env file or export the variable.[/yellow]"
            )

        cloud_models: list[dict] = []
        if options.mode in (None, "cloud"):
            try:
                with console.status("Fetching cloud models…"):
                    cloud_models = sort_by_modified(
                        await fetch_tags(client, config.cloud_base_url)
                    )
            except Exception as e:
                console.print(f"[red]✗ Failed to fetch cloud models: {e}[/red]")

        if options.target_models:
            local_models = [
                m
                for m in local_models
                if any(match_model(m["name"], t) for t in options.target_models)
            ]
            cloud_models = [
                m
                for m in cloud_models
                if any(match_model(m["name"], t) for t in options.target_models)
            ]

            searched: list[str] = []
            if options.mode in (None, "local"):
                searched.append("local")
            if options.mode in (None, "cloud"):
                searched.append("cloud")
            if not local_models and not cloud_models:
                console.print(
                    f"[red]✗ No model matching '{', '.join(options.target_models)}' found in {', '.join(searched)}.[/red]"
                )
                sys.exit(1)

        all_exports: list[ExportRow] = []

        if local_models:
            local_title = "Ollama Local Models"
            if options.target_models:
                local_title += f" — {', '.join(options.target_models)}"
            local_exports = await stream_table(
                client,
                config,
                config.local_base_url,
                local_models,
                local_title,
                options.show_ttft,
                options.show_tps,
                options.verbose,
                num_predict=options.num_predict,
                export_only=export_only,
                sort_spec=sort_spec,
                mode="local",
            )
            all_exports.extend(local_exports)
            if options.mode is None and not export_only:
                console.print()
        elif options.mode in (None, "local"):
            console.print("[dim]No local models found.[/dim]")

        if cloud_models:
            cloud_title = "Ollama Cloud Available Models"
            if options.target_models:
                cloud_title += f" — {', '.join(options.target_models)}"
            cloud_exports = await stream_table(
                client,
                config,
                config.cloud_base_url,
                cloud_models,
                cloud_title,
                options.show_ttft,
                options.show_tps,
                options.verbose,
                chat_headers,
                num_predict=options.num_predict,
                export_only=export_only,
                sort_spec=sort_spec,
                mode="cloud",
            )
            all_exports.extend(cloud_exports)
        elif options.mode in (None, "cloud"):
            console.print("[dim]No cloud models found.[/dim]")

        if options.export_fmt and all_exports:
            export_results(
                all_exports,
                options.export_fmt,
                options.export_path,
                config.num_runs,
                options.show_ttft,
                options.show_tps,
                options.verbose,
            )

        if all_exports and (options.show_ttft or options.show_tps):
            conn = get_connection()
            for export in all_exports:
                run_data = build_run_data(
                    model_name=export.model,
                    model_size=export.size,
                    context_length=(
                        int(export.context) if export.context.isdigit() else 0
                    ),
                    quantization=export.quant,
                    capabilities=export.capabilities,
                    ttft=export.ttft,
                    tps=export.tps,
                    error=export.error,
                    mode=export.mode,
                    prompts=[r["prompt"] for r in export.runs],
                    num_predict=options.num_predict,
                    parallel=config.num_parallel,
                )
                save_run(conn, run_data)
            conn.close()


def match_model(model_name: str, target: str) -> bool:
    if not target:
        return False
    if model_name == target:
        return True
    model_family = model_name.split(":")[0]
    target_family = target.split(":")[0]
    if model_family == target_family:
        return True
    return False


def build_parser(prog: str = "ometer") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Ollama model benchmarking",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--local", action="store_true", help="Show only local models")
    parser.add_argument("--cloud", action="store_true", help="Show only cloud models")
    parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        default=None,
        help="Filter to specific model names (exact or family match, e.g. --model llama3:latest llama3.2)",
    )
    parser.add_argument(
        "--ttft", action="store_true", help="Benchmark time-to-first-token"
    )
    parser.add_argument(
        "--tps", action="store_true", help="Benchmark tokens-per-second"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-run breakdown in table"
    )
    parser.add_argument(
        "--runs",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Number of benchmark prompts per model (1-3)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs=1,
        default=None,
        help="Custom benchmark prompt(s). Pass a filename to read one prompt per line, or a single inline prompt. Overrides --runs.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of models benchmarked in parallel (default 1)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        choices=list(SORT_FIELDS),
        help=(
            "Sort results by field. "
            "Default order is best-first (e.g. ttft lowest first, tps highest first). "
            "Use --reverse to invert."
        ),
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the sort order (--sort required)",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=None,
        help="Maximum number of generated tokens passed through as Ollama num_predict",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show benchmark history from previous runs",
    )
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument(
        "--json",
        nargs="?",
        const="",
        default=None,
        dest="json_export",
        help="Export results as JSON (optionally specify file path)",
    )
    export_group.add_argument(
        "--csv",
        nargs="?",
        const="",
        default=None,
        dest="csv_export",
        help="Export results as CSV (optionally specify file path)",
    )
    return parser


def resolve_mode(
    args: argparse.Namespace,
    is_tty: bool,
    prompt_fn: Callable[[], str],
) -> str | None:
    if args.local and args.cloud:
        return None
    if args.local and not args.cloud:
        return "local"
    if args.cloud and not args.local:
        return "cloud"
    if not is_tty or args.model is not None:
        return None
    choice = prompt_fn()
    if choice == "cancel":
        raise SystemExit(0)
    if choice in ("local", "cloud"):
        return choice
    return None


def main_entrypoint() -> None:
    prog = Path(sys.argv[0]).name
    parser = build_parser(prog)
    args = parser.parse_args()

    if args.reverse and not args.sort:
        parser.error("--reverse requires --sort")

    raw_prompt = args.prompts[0] if args.prompts else None
    if raw_prompt is not None:
        path = Path(raw_prompt)
        if path.is_file():
            resolved_prompts = [
                line.strip() for line in path.read_text().splitlines() if line.strip()
            ]
        else:
            resolved_prompts = [raw_prompt]
    else:
        resolved_prompts = None

    config = Config.from_env(
        runs=args.runs, parallel=args.parallel, prompts=resolved_prompts
    )

    export_fmt = None
    export_path = None
    if args.json_export is not None:
        export_fmt = "json"
        export_path = args.json_export or None
    elif args.csv_export is not None:
        export_fmt = "csv"
        export_path = args.csv_export or None

    if args.history:
        options = MainOptions(
            verbose=args.verbose,
            target_models=args.model,
            export_fmt=export_fmt,
            export_path=export_path,
            show_history=True,
        )
        asyncio.run(main(config, options))
        return

    def _interactive_prompt() -> str:
        return ListPrompt(
            message="Which Ollama models would you like to list?",
            choices=[
                {"name": "Both", "value": "both"},
                {"name": "Local only", "value": "local"},
                {"name": "Cloud only", "value": "cloud"},
                {"name": "Cancel", "value": "cancel"},
            ],
            default="both",
            qmark="",
            amark="",
            pointer="→",
        ).execute()

    try:
        mode = resolve_mode(args, sys.stdin.isatty(), _interactive_prompt)
    except SystemExit:
        console.print("[dim]Canceled.[/dim]")
        raise

    options = MainOptions(
        mode=mode,
        show_ttft=args.ttft,
        show_tps=args.tps,
        verbose=args.verbose,
        target_models=args.model,
        num_predict=args.num_predict,
        export_fmt=export_fmt,
        export_path=export_path,
        sort=args.sort,
        reverse=args.reverse,
    )

    try:
        asyncio.run(main(config, options))
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("[dim]Interrupted.[/dim]")
        sys.exit(130)
