from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
import pytest_httpx

from ometer.api import (
    BenchmarkResult,
    benchmark_chat_single_run,
    benchmark_embed_single_run,
    benchmark_model,
    fetch_model_show,
    fetch_tags,
    is_embedding_model,
    sort_by_modified,
)
from ometer.config import Config


class TestSortByModified:
    def test_sorts_newest_first(self):
        models = [
            {"name": "old", "modified_at": "2023-01-01T00:00:00Z"},
            {"name": "new", "modified_at": "2024-06-15T12:00:00Z"},
            {"name": "mid", "modified_at": "2023-07-01T00:00:00Z"},
        ]
        result = sort_by_modified(models)
        assert [m["name"] for m in result] == ["new", "mid", "old"]

    def test_missing_modified_at_goes_last(self):
        models = [
            {"name": "no_date"},
            {"name": "has_date", "modified_at": "2024-01-01T00:00:00Z"},
        ]
        result = sort_by_modified(models)
        assert result[0]["name"] == "has_date"

    def test_empty_list(self):
        assert sort_by_modified([]) == []

    def test_invalid_date_falls_to_end(self):
        models = [
            {"name": "valid", "modified_at": "2024-01-01T00:00:00Z"},
            {"name": "bad", "modified_at": "not-a-date"},
        ]
        result = sort_by_modified(models)
        assert result[0]["name"] == "valid"
        assert result[1]["name"] == "bad"


class TestIsEmbeddingModel:
    def test_embedding_capability(self):
        assert is_embedding_model({"capabilities": ["embedding"]}) is True

    def test_no_capabilities(self):
        assert is_embedding_model({}) is False

    def test_other_capabilities(self):
        assert is_embedding_model({"capabilities": ["completion"]}) is False

    def test_mixed_capabilities(self):
        assert is_embedding_model({"capabilities": ["completion", "embedding"]}) is True


class TestFetchTags:
    @pytest.mark.asyncio
    async def test_fetch_tags(self, httpx_mock: pytest_httpx.HTTPXMock):
        models = [{"name": "llama3"}, {"name": "mistral"}]
        httpx_mock.add_response(
            url="http://localhost:11434/api/tags",
            json={"models": models},
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_tags(client, "http://localhost:11434")
        assert result == models

    @pytest.mark.asyncio
    async def test_fetch_tags_empty(self, httpx_mock: pytest_httpx.HTTPXMock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/tags",
            json={},
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_tags(client, "http://localhost:11434")
        assert result == []


class TestFetchModelShow:
    @pytest.mark.asyncio
    async def test_fetch_model_show(self, httpx_mock: pytest_httpx.HTTPXMock):
        show_data = {
            "capabilities": ["completion"],
            "details": {"parameter_size": "7B"},
        }
        httpx_mock.add_response(
            url="http://localhost:11434/api/show",
            json=show_data,
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_model_show(client, "http://localhost:11434", "llama3")
        assert result == show_data


class TestBenchmarkChatSingleRun:
    @pytest.mark.asyncio
    async def test_successful_run(self, httpx_mock: pytest_httpx.HTTPXMock):
        chunks = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps(
                {
                    "message": {"content": " world"},
                    "done": True,
                    "eval_count": 10,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 2_000_000_000,
                }
            ),
        ]
        body = "\n".join(chunks)
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            content=body.encode(),
        )
        async with httpx.AsyncClient() as client:
            result = await benchmark_chat_single_run(
                client, "http://localhost:11434", "llama3", "hi"
            )
        assert result["ttft"] is not None
        assert result["ttft"] >= 0
        assert result["tps"] > 0
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_error_in_stream(self, httpx_mock: pytest_httpx.HTTPXMock):
        chunks = [
            json.dumps({"error": "model not found"}),
        ]
        body = "\n".join(chunks)
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            content=body.encode(),
        )
        async with httpx.AsyncClient() as client:
            result = await benchmark_chat_single_run(
                client, "http://localhost:11434", "missing", "hi"
            )
        assert result["error"] == "model not found"

    @pytest.mark.asyncio
    async def test_stream_without_done(self, httpx_mock: pytest_httpx.HTTPXMock):
        chunks = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
        ]
        body = "\n".join(chunks)
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            content=body.encode(),
        )
        async with httpx.AsyncClient() as client:
            result = await benchmark_chat_single_run(
                client, "http://localhost:11434", "llama3", "hi"
            )
        assert result["error"] == "Stream ended without completion"

    @pytest.mark.asyncio
    async def test_thinking_model_first_token(self, httpx_mock: pytest_httpx.HTTPXMock):
        chunks = [
            json.dumps({"message": {"thinking": "hmm"}, "done": False}),
            json.dumps(
                {
                    "message": {"content": "answer"},
                    "done": True,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_000_000_000,
                }
            ),
        ]
        body = "\n".join(chunks)
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            content=body.encode(),
        )
        show_data = {"capabilities": ["thinking"]}
        async with httpx.AsyncClient() as client:
            result = await benchmark_chat_single_run(
                client, "http://localhost:11434", "thinker", "hi", show_data=show_data
            )
        assert result["ttft"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_blank_line_in_stream(self, httpx_mock: pytest_httpx.HTTPXMock):
        chunks = [
            "",
            json.dumps(
                {
                    "message": {"content": "Hi"},
                    "done": True,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_000_000_000,
                }
            ),
        ]
        body = "\n".join(chunks)
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            content=body.encode(),
        )
        async with httpx.AsyncClient() as client:
            result = await benchmark_chat_single_run(
                client, "http://localhost:11434", "llama3", "hi"
            )
        assert result["ttft"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_invalid_json_in_stream(self, httpx_mock: pytest_httpx.HTTPXMock):
        chunks = [
            "this is not json",
            json.dumps(
                {
                    "message": {"content": "Hi"},
                    "done": True,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_000_000_000,
                }
            ),
        ]
        body = "\n".join(chunks)
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            content=body.encode(),
        )
        async with httpx.AsyncClient() as client:
            result = await benchmark_chat_single_run(
                client, "http://localhost:11434", "llama3", "hi"
            )
        assert result["ttft"] is not None
        assert result["error"] is None


class TestBenchmarkEmbedSingleRun:
    @pytest.mark.asyncio
    async def test_embedding_successful_run(self, httpx_mock: pytest_httpx.HTTPXMock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/embed",
            json={"prompt_eval_count": 8, "total_duration": 500_000_000},
        )
        async with httpx.AsyncClient() as client:
            result = await benchmark_embed_single_run(
                client, "http://localhost:11434", "nomic", "hello"
            )
        assert result["ttft"] is not None
        assert result["ttft"] >= 0
        assert result["tps"] > 0
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_http_error(self, httpx_mock: pytest_httpx.HTTPXMock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/embed",
            status_code=500,
        )
        async with httpx.AsyncClient() as client:
            result = await benchmark_embed_single_run(
                client, "http://localhost:11434", "nomic", "hello"
            )
        assert result["error"] is not None


class TestBenchmarkModel:
    @pytest.mark.asyncio
    async def test_chat_model(self, httpx_mock: pytest_httpx.HTTPXMock):
        for i in range(3):
            chunks = [
                json.dumps(
                    {
                        "message": {"content": "hi"},
                        "done": True,
                        "eval_count": 10,
                        "eval_duration": 1_000_000_000,
                        "total_duration": 2_000_000_000,
                    }
                ),
            ]
            httpx_mock.add_response(
                url="http://localhost:11434/api/chat",
                content="\n".join(chunks).encode(),
            )
        cfg = Config("http://localhost:11434", "https://ollama.com", "", 3, 1)
        show_data = {"capabilities": ["completion"]}
        async with httpx.AsyncClient() as client:
            result = await benchmark_model(
                client, cfg, "http://localhost:11434", "llama3", show_data
            )
        assert isinstance(result, BenchmarkResult)
        assert result.ttft is not None
        assert result.tps is not None
        assert result.tps > 0
        assert result.error is None
        assert len(result.runs) == 3

    @pytest.mark.asyncio
    async def test_embedding_model_local(self, httpx_mock: pytest_httpx.HTTPXMock):
        for _ in range(3):
            httpx_mock.add_response(
                url="http://localhost:11434/api/embed",
                json={"prompt_eval_count": 8, "total_duration": 500_000_000},
            )
        cfg = Config("http://localhost:11434", "https://ollama.com", "", 3, 1)
        show_data = {"capabilities": ["embedding"]}
        async with httpx.AsyncClient() as client:
            result = await benchmark_model(
                client, cfg, "http://localhost:11434", "nomic", show_data
            )
        assert result.error is None
        assert len(result.runs) == 3

    @pytest.mark.asyncio
    async def test_all_runs_fail(self, httpx_mock: pytest_httpx.HTTPXMock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            status_code=500,
        )
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            status_code=500,
        )
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            status_code=500,
        )
        cfg = Config("http://localhost:11434", "https://ollama.com", "", 3, 1)
        show_data = {"capabilities": ["completion"]}
        async with httpx.AsyncClient() as client:
            result = await benchmark_model(
                client, cfg, "http://localhost:11434", "llama3", show_data
            )
        assert result.error is not None
