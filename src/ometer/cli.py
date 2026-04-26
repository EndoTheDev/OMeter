from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Callable

import httpx
from InquirerPy.prompts.list import ListPrompt

from ometer.api import sort_by_modified, fetch_tags
from ometer.config import Config
from ometer.display import console, stream_table


async def main(
    mode: str | None,
    show_ttft: bool,
    show_tps: bool,
    verbose: bool,
    target_model: str | None,
    config: Config,
) -> None:
    chat_headers: dict[str, str] | None = None
    if config.cloud_api_key:
        chat_headers = {"Authorization": f"Bearer {config.cloud_api_key}"}

    async with httpx.AsyncClient() as client:
        local_models: list[dict] = []
        if mode in (None, "local"):
            try:
                with console.status("Fetching local models…"):
                    local_models = sort_by_modified(
                        await fetch_tags(client, config.local_base_url)
                    )
            except Exception as e:
                console.print(f"[yellow]⚠ Skipping local Ollama ({e}).[/yellow]")

        if (
            (show_ttft or show_tps)
            and mode in (None, "cloud")
            and not config.cloud_api_key
        ):
            console.print(
                "[yellow]⚠ OLLAMA_CLOUD_API_KEY not found. Cloud benchmarking will fail without it.\n"
                "  Create a .env file or export the variable.[/yellow]"
            )

        cloud_models: list[dict] = []
        if mode in (None, "cloud"):
            try:
                with console.status("Fetching cloud models…"):
                    cloud_models = sort_by_modified(
                        await fetch_tags(client, config.cloud_base_url)
                    )
            except Exception as e:
                console.print(f"[red]✗ Failed to fetch cloud models: {e}[/red]")

        if target_model:
            local_models = [m for m in local_models if m["name"] == target_model]
            cloud_models = [m for m in cloud_models if m["name"] == target_model]

            searched: list[str] = []
            if mode in (None, "local"):
                searched.append("local")
            if mode in (None, "cloud"):
                searched.append("cloud")
            if not local_models and not cloud_models:
                console.print(
                    f"[red]✗ Model '{target_model}' not found in {', '.join(searched)}.[/red]"
                )
                sys.exit(1)

        if local_models:
            local_title = "Ollama Local Models"
            if target_model:
                local_title += f" — {target_model}"
            await stream_table(
                client,
                config,
                config.local_base_url,
                local_models,
                local_title,
                show_ttft,
                show_tps,
                verbose,
            )
            if mode is None:
                console.print()
        elif mode in (None, "local"):
            console.print("[dim]No local models found.[/dim]")

        if cloud_models:
            cloud_title = "Ollama Cloud Available Models"
            if target_model:
                cloud_title += f" — {target_model}"
            await stream_table(
                client,
                config,
                config.cloud_base_url,
                cloud_models,
                cloud_title,
                show_ttft,
                show_tps,
                verbose,
                chat_headers,
            )
        elif mode in (None, "cloud"):
            console.print("[dim]No cloud models found.[/dim]")


def build_parser(prog: str = "ometer") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Ollama model benchmarking",
    )
    parser.add_argument("--local", action="store_true", help="Show only local models")
    parser.add_argument("--cloud", action="store_true", help="Show only cloud models")
    parser.add_argument(
        "--model", type=str, default=None, help="Filter to one model (exact match)"
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
        "--parallel",
        type=int,
        default=None,
        help="Number of models benchmarked in parallel (default 1)",
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

    config = Config.from_env(runs=args.runs, parallel=args.parallel)

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

    try:
        asyncio.run(main(mode, args.ttft, args.tps, args.verbose, args.model, config))
    except KeyboardInterrupt:
        console.print("[dim]Interrupted.[/dim]")
        sys.exit(130)
