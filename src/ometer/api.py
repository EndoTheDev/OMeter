from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from ometer.config import Config


@dataclass
class BenchmarkResult:
    ttft: float | None
    tps: float | None
    error: str | None
    runs: list[dict[str, Any]] = field(default_factory=list)


def sort_by_modified(models: list[dict[str, Any]]) -> list[dict[str, Any]]:

    def _key(model: dict[str, Any]) -> datetime:
        raw = model.get("modified_at", "1970-01-01T00:00:00Z")
        iso = raw.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(iso)
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

    return sorted(models, key=_key, reverse=True)


async def fetch_tags(client: httpx.AsyncClient, base_url: str) -> list[dict[str, Any]]:
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        should_retry = False
        try:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                should_retry = True
            else:
                raise
        except httpx.RequestError as e:
            if attempt < max_attempts:
                should_retry = True
            else:
                raise
        if should_retry:
            await asyncio.sleep(2 ** (attempt - 1))


async def fetch_model_show(
    client: httpx.AsyncClient, base_url: str, model_name: str
) -> dict[str, Any]:
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        should_retry = False
        try:
            resp = await client.post(
                f"{base_url}/api/show",
                json={"model": model_name},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                should_retry = True
            else:
                raise
        except httpx.RequestError as e:
            if attempt < max_attempts:
                should_retry = True
            else:
                raise
        if should_retry:
            await asyncio.sleep(2 ** (attempt - 1))


def is_embedding_model(show_data: dict[str, Any]) -> bool:
    return "embedding" in show_data.get("capabilities", [])


async def benchmark_chat_single_run(
    client: httpx.AsyncClient,
    base_url: str,
    model_name: str,
    prompt: str,
    headers: dict[str, str] | None = None,
    show_data: dict[str, Any] | None = None,
    num_predict: int | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if num_predict is not None:
        payload["options"] = {"num_predict": num_predict}

    is_thinking = bool(show_data) and "thinking" in show_data.get("capabilities", [])

    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        first_token_time: float = -1.0
        eval_count = 0
        eval_duration = 0
        total_duration = 0
        error: str | None = None
        seen_done = False
        should_retry = False

        try:
            async with client.stream(
                "POST",
                f"{base_url}/api/chat",
                json=payload,
                headers=headers,
                timeout=300.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("error"):
                        error = chunk["error"]
                        break

                    msg = chunk.get("message") or {}
                    if is_thinking:
                        if first_token_time < 0 and (
                            msg.get("thinking") or msg.get("content")
                        ):
                            first_token_time = time.perf_counter() - start
                    else:
                        if first_token_time < 0 and msg.get("content"):
                            first_token_time = time.perf_counter() - start

                    if chunk.get("done"):
                        eval_count = chunk.get("eval_count", 0)
                        eval_duration = chunk.get("eval_duration", 0)
                        total_duration = chunk.get("total_duration", 0)
                        seen_done = True
                        break
        except httpx.HTTPStatusError as e:
            error = str(e)
            if e.response.status_code in (429, 500, 502, 503, 504):
                should_retry = True
        except httpx.RequestError as e:
            error = str(e)
            should_retry = True
        except Exception as e:
            error = str(e)

        if not seen_done and not error:
            error = "Stream ended without completion"
            should_retry = True

        if error and should_retry and attempt < max_attempts:
            await asyncio.sleep(2 ** (attempt - 1))
            continue

        if first_token_time >= 0:
            ttft = first_token_time
        else:
            ttft = None
        duration = eval_duration or total_duration
        tps = eval_count / (duration / 1e9) if duration else None

        return {"ttft": ttft, "tps": tps, "error": error}

    return {"ttft": None, "tps": None, "error": error or "Unknown failure"}


async def benchmark_embed_single_run(
    client: httpx.AsyncClient,
    base_url: str,
    model_name: str,
    prompt: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "input": prompt,
    }

    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        prompt_eval_count = 0
        total_duration = 0
        error: str | None = None
        should_retry = False

        try:
            resp = await client.post(
                f"{base_url}/api/embed",
                json=payload,
                headers=headers,
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()
            prompt_eval_count = data.get("prompt_eval_count", 0)
            total_duration = data.get("total_duration", 0)
        except httpx.HTTPStatusError as e:
            error = str(e)
            if e.response.status_code in (429, 500, 502, 503, 504):
                should_retry = True
        except httpx.RequestError as e:
            error = str(e)
            should_retry = True
        except Exception as e:
            error = str(e)

        if error and should_retry and attempt < max_attempts:
            await asyncio.sleep(2 ** (attempt - 1))
            continue

        ttft = time.perf_counter() - start
        tps = prompt_eval_count / (total_duration / 1e9) if total_duration else None

        return {"ttft": ttft, "tps": tps, "error": error}

    return {"ttft": None, "tps": None, "error": error or "Unknown failure"}


async def benchmark_model(
    client: httpx.AsyncClient,
    config: Config,
    base_url: str,
    model_name: str,
    show_data: dict[str, Any],
    headers: dict[str, str] | None = None,
    num_predict: int | None = None,
) -> BenchmarkResult:
    runs: list[dict[str, Any]] = []
    errors: list[str] = []

    use_embed = is_embedding_model(show_data) and base_url.rstrip(
        "/"
    ) == config.local_base_url.rstrip("/")

    for prompt in config.bench_prompts_active:
        if use_embed:
            result = await benchmark_embed_single_run(
                client, base_url, model_name, prompt, headers
            )
        else:
            result = await benchmark_chat_single_run(
                client, base_url, model_name, prompt, headers, show_data, num_predict
            )
        runs.append({"prompt": prompt, **result})
        if result["error"]:
            errors.append(result["error"])

    good_runs = [r for r in runs if not r["error"]]
    if good_runs:
        ttft_runs = [r for r in good_runs if r["ttft"] is not None]
        avg_ttft: float | None = (
            sum(r["ttft"] for r in ttft_runs) / len(ttft_runs) if ttft_runs else None
        )
        avg_tps = None
        tps_runs = [r for r in good_runs if r["tps"] is not None]
        if tps_runs:
            avg_tps = sum(r["tps"] for r in tps_runs) / len(tps_runs)
        first_error = errors[0] if errors else None
        return BenchmarkResult(ttft=avg_ttft, tps=avg_tps, error=first_error, runs=runs)

    return BenchmarkResult(
        ttft=None,
        tps=None,
        error=errors[0] if errors else "All benchmark runs failed",
        runs=runs,
    )
