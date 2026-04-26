from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ometer.api import BenchmarkResult
from ometer.config import Config
from ometer.display import (
    _benchmark_model_task,
    _build_colored_table,
    _color,
    _column_indices,
    _parse_value,
    _thresholds,
    build_table,
    extract_context_length,
    format_capabilities,
    format_float_or_na,
    format_size,
    process_single_model,
    stream_table,
)


class TestExtractContextLength:
    def test_finds_context_length(self):
        info = {"model.context_length": 4096, "model.vocab_size": 32000}
        assert extract_context_length(info) == 4096

    def test_missing_key(self):
        assert extract_context_length({}) == 0

    def test_nested_key(self):
        info = {"general.context_length": 8192}
        assert extract_context_length(info) == 8192


class TestFormatSize:
    def test_trillion(self):
        assert format_size("1000000000000", "model") == "1T"

    def test_billion(self):
        assert format_size("7000000000", "model") == "7B"

    def test_million(self):
        assert format_size("300000000", "model") == "300M"

    def test_small_number(self):
        assert format_size("500", "model") == "500"

    def test_string_with_suffix(self):
        assert format_size("7B", "model") == "7B"

    def test_string_with_suffix_case_insensitive(self):
        assert format_size("7b", "model") == "7B"

    def test_none_fallback_to_name(self):
        assert format_size(None, "llama3-8b") == "8B"

    def test_none_no_match(self):
        assert format_size(None, "tiny") == "0B"

    def test_float_string(self):
        assert format_size("8.5B", "model") == "8.5B"


class TestFormatCapabilities:
    def test_sorted(self):
        assert format_capabilities(["vision", "completion"]) == "completion, vision"

    def test_empty(self):
        assert format_capabilities([]) == ""

    def test_single(self):
        assert format_capabilities(["embedding"]) == "embedding"


class TestFormatFloatOrNa:
    def test_value(self):
        assert format_float_or_na(3.14159) == "3.14"

    def test_none(self):
        assert format_float_or_na(None) == "n/a"

    def test_zero(self):
        assert format_float_or_na(0.0) == "0.00"


class TestParseValue:
    def test_float_string(self):
        assert _parse_value("3.14") == 3.14

    def test_int_string(self):
        assert _parse_value("42") == 42.0

    def test_invalid(self):
        assert _parse_value("abc") is None

    def test_empty(self):
        assert _parse_value("") is None


class TestThresholds:
    def test_basic(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = _thresholds(vals)
        assert result is not None
        low, high = result
        assert low <= high

    def test_empty(self):
        assert _thresholds([]) is None

    def test_single_value(self):
        result = _thresholds([5.0])
        assert result is not None
        low, high = result
        assert low == 5.0
        assert high == 5.0


class TestColor:
    def test_err(self):
        result = _color("err", (1.0, 3.0), lower_is_better=True)
        assert str(result) == "err"
        assert "red" in str(result.style)

    def test_na(self):
        result = _color("n/a", (1.0, 3.0), lower_is_better=True)
        assert "red" in str(result.style)

    def test_unparseable(self):
        result = _color("abc", None, lower_is_better=True)
        assert result.plain == "abc"

    def test_no_thresholds(self):
        result = _color("2.0", None, lower_is_better=True)
        assert result.plain == "2.0"

    def test_lower_is_better_green(self):
        result = _color("0.5", (1.0, 3.0), lower_is_better=True)
        assert "green" in str(result.style)

    def test_lower_is_better_red(self):
        result = _color("5.0", (1.0, 3.0), lower_is_better=True)
        assert "red" in str(result.style)

    def test_lower_is_better_orange(self):
        result = _color("2.0", (1.0, 3.0), lower_is_better=True)
        assert "orange3" in str(result.style)

    def test_higher_is_better_green(self):
        result = _color("5.0", (1.0, 3.0), lower_is_better=False)
        assert "green" in str(result.style)

    def test_higher_is_better_red(self):
        result = _color("0.5", (1.0, 3.0), lower_is_better=False)
        assert "red" in str(result.style)

    def test_higher_is_better_orange(self):
        result = _color("2.0", (1.0, 3.0), lower_is_better=False)
        assert "orange3" in str(result.style)


class TestColumnIndices:
    def test_ttft_only(self):
        ttft_idx, tps_idx = _column_indices(
            show_ttft=True, show_tps=False, verbose=False, num_runs=3
        )
        assert ttft_idx == [5]
        assert tps_idx == []

    def test_tps_only(self):
        ttft_idx, tps_idx = _column_indices(
            show_ttft=False, show_tps=True, verbose=False, num_runs=3
        )
        assert ttft_idx == []
        assert tps_idx == [5]

    def test_both(self):
        ttft_idx, tps_idx = _column_indices(
            show_ttft=True, show_tps=True, verbose=False, num_runs=3
        )
        assert ttft_idx == [5]
        assert tps_idx == [6]

    def test_verbose_ttft(self):
        ttft_idx, tps_idx = _column_indices(
            show_ttft=True, show_tps=False, verbose=True, num_runs=2
        )
        assert ttft_idx == [5, 6, 7]
        assert tps_idx == []

    def test_verbose_both(self):
        ttft_idx, tps_idx = _column_indices(
            show_ttft=True, show_tps=True, verbose=True, num_runs=2
        )
        assert ttft_idx == [5, 6, 7]
        assert tps_idx == [8, 9, 10]

    def test_neither(self):
        ttft_idx, tps_idx = _column_indices(
            show_ttft=False, show_tps=False, verbose=False, num_runs=3
        )
        assert ttft_idx == []
        assert tps_idx == []


class TestBuildTable:
    def test_basic_columns(self):
        table = build_table(
            "Test", show_ttft=False, show_tps=False, verbose=False, num_runs=3
        )
        assert len(table.columns) == 5

    def test_with_ttft(self):
        table = build_table(
            "Test", show_ttft=True, show_tps=False, verbose=False, num_runs=3
        )
        assert len(table.columns) == 6

    def test_with_tps(self):
        table = build_table(
            "Test", show_ttft=False, show_tps=True, verbose=False, num_runs=3
        )
        assert len(table.columns) == 6

    def test_with_both(self):
        table = build_table(
            "Test", show_ttft=True, show_tps=True, verbose=False, num_runs=3
        )
        assert len(table.columns) == 7

    def test_verbose_both(self):
        table = build_table(
            "Test", show_ttft=True, show_tps=True, verbose=True, num_runs=2
        )
        assert len(table.columns) == 5 + 2 + 1 + 2 + 1


class TestProcessSingleModel:
    def test_basic_row(self):
        tag_model = {
            "name": "llama3",
            "details": {"parameter_size": "8B", "quantization_level": "Q4_0"},
        }
        show_data = {
            "details": {},
            "capabilities": ["completion"],
            "model_info": {"model.context_length": 8192},
        }
        benchmark = BenchmarkResult(
            ttft=1.23,
            tps=45.6,
            error=None,
            runs=[
                {"prompt": "hi", "ttft": 1.23, "tps": 45.6, "error": None},
            ],
        )
        row = process_single_model(
            tag_model,
            show_data,
            benchmark,
            show_ttft=True,
            show_tps=True,
            verbose=False,
            num_runs=1,
        )
        assert row[0] == "llama3"
        assert row[1] == "8B"
        assert row[2] == "8192"
        assert row[3] == "Q4_0"
        assert "completion" in row[4]

    def test_no_benchmark(self):
        tag_model = {"name": "llama3", "details": {}}
        show_data = {"details": {}, "capabilities": [], "model_info": {}}
        benchmark = BenchmarkResult(ttft=None, tps=None, error=None, runs=[])
        row = process_single_model(
            tag_model,
            show_data,
            benchmark,
            show_ttft=False,
            show_tps=False,
            verbose=False,
            num_runs=3,
        )
        assert row == ["llama3", "0B", "0", "", ""]


class TestProcessSingleModelVerbose:
    def test_verbose_ttft_with_error(self):
        tag_model = {"name": "llama3", "details": {"parameter_size": "8B"}}
        show_data = {"details": {}, "capabilities": [], "model_info": {}}
        benchmark = BenchmarkResult(
            ttft=1.0,
            tps=None,
            error=None,
            runs=[
                {"prompt": "p1", "ttft": 1.0, "tps": None, "error": "timeout"},
                {"prompt": "p2", "ttft": None, "tps": None, "error": None},
            ],
        )
        row = process_single_model(
            tag_model,
            show_data,
            benchmark,
            show_ttft=True,
            show_tps=False,
            verbose=True,
            num_runs=2,
        )
        assert "err" in row
        assert "n/a" in row

    def test_verbose_tps_with_error(self):
        tag_model = {"name": "llama3", "details": {"parameter_size": "8B"}}
        show_data = {"details": {}, "capabilities": [], "model_info": {}}
        benchmark = BenchmarkResult(
            ttft=None,
            tps=50.0,
            error=None,
            runs=[
                {"prompt": "p1", "ttft": None, "tps": 50.0, "error": None},
                {"prompt": "p2", "ttft": None, "tps": None, "error": "fail"},
            ],
        )
        row = process_single_model(
            tag_model,
            show_data,
            benchmark,
            show_ttft=False,
            show_tps=True,
            verbose=True,
            num_runs=2,
        )
        assert "50.00" in row
        assert "err" in row

    def test_verbose_both(self):
        tag_model = {"name": "llama3", "details": {"parameter_size": "8B"}}
        show_data = {"details": {}, "capabilities": [], "model_info": {}}
        benchmark = BenchmarkResult(
            ttft=1.5,
            tps=40.0,
            error=None,
            runs=[
                {"prompt": "p1", "ttft": 1.5, "tps": 40.0, "error": None},
                {"prompt": "p2", "ttft": 2.0, "tps": 35.0, "error": None},
            ],
        )
        row = process_single_model(
            tag_model,
            show_data,
            benchmark,
            show_ttft=True,
            show_tps=True,
            verbose=True,
            num_runs=2,
        )
        assert len(row) == 5 + 2 + 1 + 2 + 1

    def test_verbose_fewer_runs_than_num_runs(self):
        tag_model = {"name": "llama3", "details": {}}
        show_data = {"details": {}, "capabilities": [], "model_info": {}}
        benchmark = BenchmarkResult(
            ttft=1.0,
            tps=30.0,
            error=None,
            runs=[
                {"prompt": "p1", "ttft": 1.0, "tps": 30.0, "error": None},
            ],
        )
        row = process_single_model(
            tag_model,
            show_data,
            benchmark,
            show_ttft=True,
            show_tps=True,
            verbose=True,
            num_runs=3,
        )
        assert "n/a" in row
        assert len(row) == 5 + 3 + 1 + 3 + 1


class TestBuildColoredTable:
    def _make_rows(self, n=2):
        rows = []
        for i in range(n):
            rows.append(
                [
                    "model",
                    "7B",
                    "4096",
                    "Q4_0",
                    "completion",
                    f"{1.0 + i:.2f}",
                    f"{30.0 + i:.2f}",
                ]
            )
        return rows

    def test_ttft_only(self):
        rows = self._make_rows()
        table = _build_colored_table(
            "Test", show_ttft=True, show_tps=False, verbose=False, num_runs=1, rows=rows
        )
        assert table is not None

    def test_tps_only(self):
        rows = self._make_rows()
        table = _build_colored_table(
            "Test", show_ttft=False, show_tps=True, verbose=False, num_runs=1, rows=rows
        )
        assert table is not None

    def test_both(self):
        rows = self._make_rows()
        table = _build_colored_table(
            "Test", show_ttft=True, show_tps=True, verbose=False, num_runs=1, rows=rows
        )
        assert table is not None

    def test_with_err_values(self):
        rows = [["model", "7B", "4096", "Q4_0", "completion", "err", "err"]]
        table = _build_colored_table(
            "Test", show_ttft=True, show_tps=True, verbose=False, num_runs=1, rows=rows
        )
        assert table is not None

    def test_with_na_values(self):
        rows = [["model", "7B", "4096", "Q4_0", "completion", "n/a", "n/a"]]
        table = _build_colored_table(
            "Test", show_ttft=True, show_tps=True, verbose=False, num_runs=1, rows=rows
        )
        assert table is not None

    def test_verbose(self):
        rows = [
            [
                "model",
                "7B",
                "4096",
                "Q4_0",
                "completion",
                "1.00",
                "2.00",
                "1.50",
                "30.00",
                "35.00",
                "32.50",
            ]
        ]
        table = _build_colored_table(
            "Test", show_ttft=True, show_tps=True, verbose=True, num_runs=2, rows=rows
        )
        assert table is not None


class TestBenchmarkModelTask:
    @pytest.mark.asyncio
    async def test_successful_show_and_benchmark(self):
        model = {"name": "llama3"}
        show_data = {
            "capabilities": ["completion"],
            "details": {},
            "model_info": {"model.context_length": 4096},
        }
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        bench_result = BenchmarkResult(
            ttft=1.0,
            tps=50.0,
            error=None,
            runs=[{"prompt": "hi", "ttft": 1.0, "tps": 50.0, "error": None}],
        )
        with patch(
            "ometer.display.benchmark_model",
            new_callable=AsyncMock,
            return_value=bench_result,
        ):
            idx, row, errors = await _benchmark_model_task(
                0,
                model,
                show_data,
                AsyncMock(),
                config,
                "http://localhost:11434",
                show_ttft=True,
                show_tps=True,
                verbose=False,
                chat_headers=None,
                semaphore=asyncio.Semaphore(1),
            )
        assert idx == 0
        assert len(errors) == 0
        assert "llama3" in row

    @pytest.mark.asyncio
    async def test_show_failure(self):
        model = {"name": "llama3"}
        err = RuntimeError("connection refused")
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        bench_result = BenchmarkResult(ttft=None, tps=None, error=None, runs=[])
        with patch(
            "ometer.display.benchmark_model",
            new_callable=AsyncMock,
            return_value=bench_result,
        ):
            idx, row, errors = await _benchmark_model_task(
                0,
                model,
                err,
                AsyncMock(),
                config,
                "http://localhost:11434",
                show_ttft=True,
                show_tps=True,
                verbose=False,
                chat_headers=None,
                semaphore=asyncio.Semaphore(1),
            )
        assert len(errors) == 1
        assert "/api/show failed" in errors[0]

    @pytest.mark.asyncio
    async def test_no_benchmark(self):
        model = {"name": "llama3"}
        show_data = {"capabilities": [], "details": {}, "model_info": {}}
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        idx, row, errors = await _benchmark_model_task(
            0,
            model,
            show_data,
            AsyncMock(),
            config,
            "http://localhost:11434",
            show_ttft=False,
            show_tps=False,
            verbose=False,
            chat_headers=None,
            semaphore=asyncio.Semaphore(1),
        )
        assert idx == 0
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_benchmark_error(self):
        model = {"name": "llama3"}
        show_data = {"capabilities": [], "details": {}, "model_info": {}}
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        bench_result = BenchmarkResult(
            ttft=None, tps=None, error="model not found", runs=[]
        )
        with patch(
            "ometer.display.benchmark_model",
            new_callable=AsyncMock,
            return_value=bench_result,
        ):
            idx, row, errors = await _benchmark_model_task(
                0,
                model,
                show_data,
                AsyncMock(),
                config,
                "http://localhost:11434",
                show_ttft=True,
                show_tps=True,
                verbose=False,
                chat_headers=None,
                semaphore=asyncio.Semaphore(1),
            )
        assert len(errors) == 1
        assert "model not found" in errors[0]


class TestStreamTable:
    @pytest.mark.asyncio
    async def test_list_only_no_benchmark(self):
        model = {
            "name": "llama3",
            "modified_at": "2024-01-01T00:00:00Z",
            "details": {"parameter_size": "8B"},
        }
        show_data = {
            "capabilities": ["completion"],
            "details": {},
            "model_info": {"model.context_length": 4096},
        }
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        client = AsyncMock()
        with patch(
            "ometer.display.fetch_model_show",
            new_callable=AsyncMock,
            return_value=show_data,
        ):
            await stream_table(
                client,
                config,
                "http://localhost:11434",
                [model],
                "Test Table",
                show_ttft=False,
                show_tps=False,
                verbose=False,
            )

    @pytest.mark.asyncio
    async def test_with_benchmark_single_model(self):
        model = {
            "name": "llama3",
            "modified_at": "2024-01-01T00:00:00Z",
            "details": {"parameter_size": "8B"},
        }
        show_data = {
            "capabilities": ["completion"],
            "details": {},
            "model_info": {"model.context_length": 4096},
        }
        bench_result = BenchmarkResult(
            ttft=1.0,
            tps=50.0,
            error=None,
            runs=[{"prompt": "hi", "ttft": 1.0, "tps": 50.0, "error": None}],
        )
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        client = AsyncMock()
        with (
            patch(
                "ometer.display.fetch_model_show",
                new_callable=AsyncMock,
                return_value=show_data,
            ),
            patch(
                "ometer.display.benchmark_model",
                new_callable=AsyncMock,
                return_value=bench_result,
            ),
        ):
            await stream_table(
                client,
                config,
                "http://localhost:11434",
                [model],
                "Test Table",
                show_ttft=True,
                show_tps=True,
                verbose=False,
            )

    @pytest.mark.asyncio
    async def test_with_benchmark_multiple_models(self):
        models = [
            {
                "name": "llama3",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"parameter_size": "8B"},
            },
            {
                "name": "mistral",
                "modified_at": "2024-02-01T00:00:00Z",
                "details": {"parameter_size": "7B"},
            },
        ]
        show_data = {
            "capabilities": ["completion"],
            "details": {},
            "model_info": {"model.context_length": 4096},
        }
        bench_result = BenchmarkResult(
            ttft=1.0,
            tps=50.0,
            error=None,
            runs=[{"prompt": "hi", "ttft": 1.0, "tps": 50.0, "error": None}],
        )
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 2)
        client = AsyncMock()
        with (
            patch(
                "ometer.display.fetch_model_show",
                new_callable=AsyncMock,
                return_value=show_data,
            ),
            patch(
                "ometer.display.benchmark_model",
                new_callable=AsyncMock,
                return_value=bench_result,
            ),
        ):
            await stream_table(
                client,
                config,
                "http://localhost:11434",
                models,
                "Test Table",
                show_ttft=True,
                show_tps=True,
                verbose=False,
            )

    @pytest.mark.asyncio
    async def test_show_failure_still_benchmarks(self):
        model = {"name": "llama3", "details": {}, "modified_at": "2024-01-01T00:00:00Z"}
        err = RuntimeError("connection refused")
        bench_result = BenchmarkResult(
            ttft=1.0,
            tps=50.0,
            error=None,
            runs=[{"prompt": "hi", "ttft": 1.0, "tps": 50.0, "error": None}],
        )
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        client = AsyncMock()
        with (
            patch(
                "ometer.display.fetch_model_show",
                new_callable=AsyncMock,
                return_value=err,
            ),
            patch(
                "ometer.display.benchmark_model",
                new_callable=AsyncMock,
                return_value=bench_result,
            ),
        ):
            await stream_table(
                client,
                config,
                "http://localhost:11434",
                [model],
                "Test Table",
                show_ttft=True,
                show_tps=True,
                verbose=False,
            )

    @pytest.mark.asyncio
    async def test_benchmark_errors_printed(self):
        model = {"name": "llama3", "details": {}, "modified_at": "2024-01-01T00:00:00Z"}
        show_data = {"capabilities": ["completion"], "details": {}, "model_info": {}}
        bench_result = BenchmarkResult(
            ttft=None,
            tps=None,
            error="timeout",
            runs=[{"prompt": "hi", "ttft": None, "tps": None, "error": "timeout"}],
        )
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        client = AsyncMock()
        with (
            patch(
                "ometer.display.fetch_model_show",
                new_callable=AsyncMock,
                return_value=show_data,
            ),
            patch(
                "ometer.display.benchmark_model",
                new_callable=AsyncMock,
                return_value=bench_result,
            ),
        ):
            await stream_table(
                client,
                config,
                "http://localhost:11434",
                [model],
                "Test Table",
                show_ttft=True,
                show_tps=True,
                verbose=False,
            )

    @pytest.mark.asyncio
    async def test_no_models(self):
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        client = AsyncMock()
        await stream_table(
            client,
            config,
            "http://localhost:11434",
            [],
            "Test Table",
            show_ttft=False,
            show_tps=False,
            verbose=False,
        )

    @pytest.mark.asyncio
    async def test_with_chat_headers(self):
        model = {
            "name": "llama3",
            "details": {"parameter_size": "8B"},
            "modified_at": "2024-01-01T00:00:00Z",
        }
        show_data = {
            "capabilities": ["completion"],
            "details": {},
            "model_info": {"model.context_length": 4096},
        }
        bench_result = BenchmarkResult(
            ttft=1.0,
            tps=50.0,
            error=None,
            runs=[{"prompt": "hi", "ttft": 1.0, "tps": 50.0, "error": None}],
        )
        config = Config("http://localhost:11434", "https://ollama.com", "key123", 1, 1)
        client = AsyncMock()
        headers = {"Authorization": "Bearer key123"}
        with (
            patch(
                "ometer.display.fetch_model_show",
                new_callable=AsyncMock,
                return_value=show_data,
            ),
            patch(
                "ometer.display.benchmark_model",
                new_callable=AsyncMock,
                return_value=bench_result,
            ),
        ):
            await stream_table(
                client,
                config,
                "https://ollama.com",
                [model],
                "Cloud",
                show_ttft=True,
                show_tps=True,
                verbose=False,
                chat_headers=headers,
            )
