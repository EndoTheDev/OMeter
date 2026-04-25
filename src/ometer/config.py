from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_BENCH_PROMPTS = [
    "why is the sky blue?",
    "explain quantum computing in one paragraph",
    "write a haiku about rain",
]


def _load_env() -> None:
    for path in (
        Path.cwd() / ".env",
        Path.home() / ".env",
        Path.home() / ".config" / "ometer" / ".env",
    ):
        if path.exists():
            load_dotenv(dotenv_path=path)


class Config:
    def __init__(
        self,
        local_base_url: str,
        cloud_base_url: str,
        cloud_api_key: str,
        num_runs: int,
        num_parallel: int,
    ) -> None:
        self.local_base_url = local_base_url
        self.cloud_base_url = cloud_base_url
        self.cloud_api_key = cloud_api_key
        self.num_runs = max(1, min(3, num_runs))
        self.bench_prompts_active = _BENCH_PROMPTS[: self.num_runs]
        self.num_parallel = max(1, min(10, num_parallel))

    @classmethod
    def from_env(cls, runs: int | None = None, parallel: int | None = None) -> Config:
        _load_env()
        local_base_url = os.getenv("OLLAMA_LOCAL_BASE_URL", "http://localhost:11434")
        cloud_base_url = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
        cloud_api_key = os.getenv("OLLAMA_CLOUD_API_KEY", "")

        num_runs_raw = os.getenv("OLLAMAMETER_RUNS", "3").strip()
        try:
            num_runs = int(num_runs_raw)
        except ValueError:
            num_runs = 3

        num_parallel_raw = os.getenv("OLLAMAMETER_PARALLEL", "1").strip()
        try:
            num_parallel = int(num_parallel_raw)
        except ValueError:
            num_parallel = 1

        if runs is not None:
            num_runs = runs
        if parallel is not None:
            num_parallel = parallel

        return cls(
            local_base_url, cloud_base_url, cloud_api_key, num_runs, num_parallel
        )
