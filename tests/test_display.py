from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from ometer.api import BenchmarkResult
from ometer.config import Config
from ometer.display import (
    SortSpec,
    _benchmark_model_task,
    _build_colored_table,
    _collect_pending,
    _color,
    _column_indices,
    _context_value,
    _modified_value,
    _parse_value,
    _size_value,
    _thresholds,
    build_table,
    extract_context_length,
    format_capabilities,
    format_float_or_na,
    format_size,
    process_single_model,
    sort_results,
    stream_table,
)
from ometer.export import ExportRow


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

    def test_zero_as_string(self):
        assert format_size("0", "model") == "0"

    def test_negative_integer(self):
        assert format_size("-5", "model") == "-5"

    def test_very_large_suffix(self):
        assert format_size("100T", "model") == "100T"

    def test_whitespace_padded_suffix(self):
        assert format_size(" 8B", "model") == "0B"

    def test_empty_string(self):
        assert format_size("", "model") == "0B"

    def test_none_with_name_suffix(self):
        assert format_size(None, "mistral-7b-instruct") == "7B"

    def test_decimal_in_name(self):
        assert format_size(None, "llama3.2-3b") == "3B"


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
        row, export_row = process_single_model(
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
        assert export_row.model == "llama3"
        assert export_row.ttft == 1.23
        assert export_row.tps == 45.6


class TestSizeValue:
    def test_basic(self):
        assert _size_value("8B") == 8e9
        assert _size_value("7B") == 7e9

    def test_with_decimal(self):
        assert _size_value("8.5B") == 8.5e9

    def test_suffixes(self):
        assert _size_value("1K") == 1e3
        assert _size_value("2M") == 2e6
        assert _size_value("3G") == 3e9
        assert _size_value("4T") == 4e12

    def test_case_insensitive(self):
        assert _size_value("8b") == 8e9
        assert _size_value("1k") == 1e3

    def test_no_match_returns_zero(self):
        assert _size_value("") == 0.0
        assert _size_value("xyz") == 0.0

    def test_raw_integer(self):
        assert _size_value("500000") == 500000.0


class TestContextValue:
    def test_basic(self):
        assert _context_value("4096") == 4096

    def test_invalid_returns_zero(self):
        assert _context_value("") == 0
        assert _context_value("abc") == 0


class TestModifiedValue:
    def test_valid_iso(self):
        assert _modified_value("2024-06-15T12:00:00Z") == datetime(
            2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc
        )

    def test_invalid_returns_min(self):
        assert _modified_value("not-a-date") == datetime.min.replace(
            tzinfo=timezone.utc
        )


class TestSortSpec:
    def test_parse_none(self):
        assert SortSpec.parse(None) is None

    def test_parse_name_default_ascending(self):
        spec = SortSpec.parse("name")
        assert spec == SortSpec("name", True)

    def test_parse_modified_default_descending(self):
        spec = SortSpec.parse("modified")
        assert spec == SortSpec("modified", False)

    def test_parse_ttft_default_ascending(self):
        spec = SortSpec.parse("ttft")
        assert spec == SortSpec("ttft", True)

    def test_parse_tps_default_descending(self):
        spec = SortSpec.parse("tps")
        assert spec == SortSpec("tps", False)

    def test_parse_size_default_descending(self):
        spec = SortSpec.parse("size")
        assert spec == SortSpec("size", False)

    def test_parse_context_default_descending(self):
        spec = SortSpec.parse("ctx")
        assert spec == SortSpec("ctx", False)

    def test_parse_reverse_name(self):
        spec = SortSpec.parse("name", reverse=True)
        assert spec == SortSpec("name", False)

    def test_parse_reverse_ttft(self):
        spec = SortSpec.parse("ttft", reverse=True)
        assert spec == SortSpec("ttft", False)

    def test_parse_reverse_tps(self):
        spec = SortSpec.parse("tps", reverse=True)
        assert spec == SortSpec("tps", True)

    def test_parse_reverse_modified(self):
        spec = SortSpec.parse("modified", reverse=True)
        assert spec == SortSpec("modified", True)

    def test_eq_and_hash(self):
        a = SortSpec("name", True)
        b = SortSpec("name", True)
        assert a == b
        assert hash(a) == hash(b)

    def test_parse_invalid_field_raises(self):
        with pytest.raises(ValueError):
            SortSpec.parse("tpx")


class TestSortResults:
    def _make_pair(
        self,
        model: str = "llama3",
        size: str = "8B",
        context: str = "4096",
        ttft: float | None = 1.0,
        tps: float | None = 50.0,
        modified_at: str = "",
    ) -> tuple[list[str], ExportRow]:
        row = [model, size, context, "Q4_0", "completion"]
        export = ExportRow(
            model=model,
            size=size,
            context=context,
            quant="Q4_0",
            capabilities="completion",
            ttft=ttft,
            tps=tps,
            error=None,
            runs=[],
            modified_at=modified_at,
        )
        return row, export

    def test_none_spec_returns_unchanged(self):
        rows, exports = zip(*[self._make_pair(model="a"), self._make_pair(model="b")])
        result_rows, result_exports = sort_results(list(rows), list(exports), None)
        assert [e.model for e in result_exports] == ["a", "b"]

    def test_sort_by_name_ascending(self):
        pairs = [
            self._make_pair(model="charlie"),
            self._make_pair(model="alpha"),
            self._make_pair(model="bravo"),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("name", True)
        )
        assert [e.model for e in result_exports] == ["alpha", "bravo", "charlie"]

    def test_sort_by_name_descending(self):
        pairs = [
            self._make_pair(model="alpha"),
            self._make_pair(model="charlie"),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("name", False)
        )
        assert [e.model for e in result_exports] == ["charlie", "alpha"]

    def test_sort_by_ttft_ascending_best_first(self):
        pairs = [
            self._make_pair(model="slow", ttft=5.0),
            self._make_pair(model="fast", ttft=1.0),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("ttft", True)
        )
        assert [e.model for e in result_exports] == ["fast", "slow"]

    def test_sort_by_ttft_descending_worst_first(self):
        pairs = [
            self._make_pair(model="fast", ttft=1.0),
            self._make_pair(model="slow", ttft=5.0),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("ttft", False)
        )
        assert [e.model for e in result_exports] == ["slow", "fast"]

    def test_sort_by_tps_descending_best_first(self):
        pairs = [
            self._make_pair(model="slow", tps=10.0),
            self._make_pair(model="fast", tps=100.0),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("tps", False)
        )
        assert [e.model for e in result_exports] == ["fast", "slow"]

    def test_sort_by_tps_ascending_worst_first(self):
        pairs = [
            self._make_pair(model="fast", tps=100.0),
            self._make_pair(model="slow", tps=10.0),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("tps", True)
        )
        assert [e.model for e in result_exports] == ["slow", "fast"]

    def test_sort_by_size_descending(self):
        pairs = [
            self._make_pair(model="small", size="3B"),
            self._make_pair(model="large", size="70B"),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("size", False)
        )
        assert [e.model for e in result_exports] == ["large", "small"]

    def test_sort_by_ctx(self):
        pairs = [
            self._make_pair(model="small", context="4096"),
            self._make_pair(model="large", context="128000"),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("ctx", False)
        )
        assert [e.model for e in result_exports] == ["large", "small"]

    def test_sort_by_modified(self):
        pairs = [
            self._make_pair(model="old", modified_at="2024-01-01T00:00:00Z"),
            self._make_pair(model="new", modified_at="2024-06-01T00:00:00Z"),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("modified", False)
        )
        assert [e.model for e in result_exports] == ["new", "old"]

    def test_sort_by_modified_descending(self):
        pairs = [
            self._make_pair(model="new", modified_at="2024-06-01T00:00:00Z"),
            self._make_pair(model="old", modified_at="2024-01-01T00:00:00Z"),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("modified", True)
        )
        assert [e.model for e in result_exports] == ["old", "new"]

    def test_none_ttft_treated_as_worst(self):
        pairs = [
            self._make_pair(model="good", ttft=1.0),
            self._make_pair(model="bad", ttft=None),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("ttft", True)
        )
        assert [e.model for e in result_exports] == ["good", "bad"]

    def test_none_tps_treated_as_worst(self):
        pairs = [
            self._make_pair(model="good", tps=100.0),
            self._make_pair(model="bad", tps=None),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("tps", False)
        )
        assert [e.model for e in result_exports] == ["good", "bad"]

    def test_unknown_field_falls_back_to_name(self):
        pairs = [
            self._make_pair(model="zebra"),
            self._make_pair(model="alpha"),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("unknown", True)
        )
        assert [e.model for e in result_exports] == ["alpha", "zebra"]

    def test_rows_reordered_with_exports(self):
        pairs = [
            self._make_pair(model="b"),
            self._make_pair(model="a"),
        ]
        rows, exports = zip(*pairs)
        result_rows, result_exports = sort_results(
            list(rows), list(exports), SortSpec("name", True)
        )
        assert [r[0] for r in result_rows] == ["a", "b"]
        assert [e.model for e in result_exports] == ["a", "b"]


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
        row, export_row = process_single_model(
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
        row, export_row = process_single_model(
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
        row, export_row = process_single_model(
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
        row, export_row = process_single_model(
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
            idx, row, export_row, errors = await _benchmark_model_task(
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
                num_predict=None,
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
            idx, row, export_row, errors = await _benchmark_model_task(
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
                num_predict=None,
                semaphore=asyncio.Semaphore(1),
            )
        assert len(errors) == 1
        assert "/api/show failed" in errors[0]

    @pytest.mark.asyncio
    async def test_no_benchmark(self):
        model = {"name": "llama3"}
        show_data = {"capabilities": [], "details": {}, "model_info": {}}
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        idx, row, export_row, errors = await _benchmark_model_task(
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
            num_predict=None,
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
            idx, row, export_row, errors = await _benchmark_model_task(
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
                num_predict=None,
                semaphore=asyncio.Semaphore(1),
            )
        assert len(errors) == 1
        assert "model not found" in errors[0]


class TestCollectPending:
    @pytest.mark.asyncio
    async def test_collects_results(self):
        export = ExportRow(
            model="llama3",
            size="8B",
            context="4096",
            quant="Q4_0",
            capabilities="completion",
            ttft=1.0,
            tps=50.0,
            error=None,
            runs=[],
        )

        async def _result(
            idx: int, row: list[str], ex: ExportRow, errs: list[str]
        ) -> tuple[int, list[str], ExportRow, list[str]]:
            return idx, row, ex, errs

        task = asyncio.create_task(_result(0, ["llama3", "8B"], export, []))
        pending: set[asyncio.Task[tuple[int, list[str], ExportRow, list[str]]]] = {task}
        completed_rows: dict[int, list[str]] = {}
        completed_exports: dict[int, ExportRow] = {}
        bench_errors: list[str] = []

        await _collect_pending(pending, completed_rows, completed_exports, bench_errors)

        assert len(pending) == 0
        assert 0 in completed_rows
        assert completed_exports[0].model == "llama3"


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

    @pytest.mark.asyncio
    async def test_export_only_returns_export_rows(self):
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
            result = await stream_table(
                client,
                config,
                "http://localhost:11434",
                [model],
                "Export Test",
                show_ttft=True,
                show_tps=True,
                verbose=False,
                export_only=True,
            )
        assert len(result) == 1
        assert result[0].model == "llama3"
        assert result[0].ttft == 1.0
        assert result[0].tps == 50.0

    @pytest.mark.asyncio
    async def test_export_only_list_mode(self):
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
            result = await stream_table(
                client,
                config,
                "http://localhost:11434",
                [model],
                "List Export",
                show_ttft=False,
                show_tps=False,
                verbose=False,
                export_only=True,
            )
        assert len(result) == 1
        assert result[0].model == "llama3"
        assert result[0].ttft is None
        assert result[0].tps is None

    @pytest.mark.asyncio
    async def test_export_only_multi_model_benchmark(self):
        models = [
            {
                "name": "llama3",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"parameter_size": "8B"},
            },
            {
                "name": "mistral",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"parameter_size": "7B"},
            },
        ]
        show_data = {
            "capabilities": ["completion"],
            "details": {},
            "model_info": {"model.context_length": 4096},
        }
        call_count = 0

        async def _slow_bench(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.05)
            return BenchmarkResult(
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
                side_effect=_slow_bench,
            ),
        ):
            result = await stream_table(
                client,
                config,
                "http://localhost:11434",
                models,
                "Multi Export",
                show_ttft=True,
                show_tps=True,
                verbose=False,
                export_only=True,
            )
        assert len(result) == 2
        assert result[0].model == "llama3"
        assert result[1].model == "mistral"

    @pytest.mark.asyncio
    async def test_live_multi_model_benchmark(self):
        models = [
            {
                "name": "llama3",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"parameter_size": "8B"},
            },
            {
                "name": "mistral",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"parameter_size": "7B"},
            },
        ]
        show_data = {
            "capabilities": ["completion"],
            "details": {},
            "model_info": {"model.context_length": 4096},
        }
        call_count = 0

        async def _slow_bench(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.05)
            return BenchmarkResult(
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
                side_effect=_slow_bench,
            ),
        ):
            await stream_table(
                client,
                config,
                "http://localhost:11434",
                models,
                "Live Multi",
                show_ttft=True,
                show_tps=True,
                verbose=False,
            )

    @pytest.mark.asyncio
    async def test_list_only_with_sort(self):
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
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        client = AsyncMock()
        with patch(
            "ometer.display.fetch_model_show",
            new_callable=AsyncMock,
            return_value=show_data,
        ):
            result = await stream_table(
                client,
                config,
                "http://localhost:11434",
                models,
                "Sorted List",
                show_ttft=False,
                show_tps=False,
                verbose=False,
                sort_spec=SortSpec.parse("name"),
            )
        assert len(result) == 2
        assert result[0].model == "llama3"
        assert result[1].model == "mistral"

    @pytest.mark.asyncio
    async def test_list_only_live_rebuilds_sorted_table(self):
        models = [
            {
                "name": "mistral",
                "modified_at": "2024-02-01T00:00:00Z",
                "details": {"parameter_size": "7B"},
            },
            {
                "name": "llama3",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"parameter_size": "8B"},
            },
        ]
        show_data = {
            "capabilities": ["completion"],
            "details": {},
            "model_info": {"model.context_length": 4096},
        }
        config = Config("http://localhost:11434", "https://ollama.com", "", 1, 1)
        client = AsyncMock()

        updated = []

        class MockLive:
            def __init__(self, renderable, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def update(self, renderable):
                updated.append(renderable)

        with (
            patch(
                "ometer.display.fetch_model_show",
                new_callable=AsyncMock,
                return_value=show_data,
            ),
            patch("ometer.display.Live", MockLive),
        ):
            result = await stream_table(
                client,
                config,
                "http://localhost:11434",
                models,
                "Sorted",
                show_ttft=False,
                show_tps=False,
                verbose=False,
                sort_spec=SortSpec.parse("size"),
            )

        assert len(result) == 2
        assert [e.model for e in result] == ["llama3", "mistral"]

        assert len(updated) >= 1
        final_table = updated[-1]
        model_column_cells = final_table.columns[0]._cells
        assert len(model_column_cells) == 2
        assert str(model_column_cells[0]) == "llama3"
        assert str(model_column_cells[1]) == "mistral"
