from __future__ import annotations

import json
from pathlib import Path

from ometer.export import ExportRow, format_csv, format_json, export_results


def _make_row(
    model: str = "llama3",
    size: str = "8B",
    context: str = "4096",
    quant: str = "Q4_0",
    capabilities: str = "completion",
    ttft: float | None = 1.23,
    tps: float | None = 45.6,
    error: str | None = None,
    runs: list[dict] | None = None,
    modified_at: str = "2024-01-01T00:00:00Z",
) -> ExportRow:
    if runs is None:
        runs = [{"prompt": "hi", "ttft": 1.23, "tps": 45.6, "error": None}]
    return ExportRow(
        model=model,
        size=size,
        context=context,
        quant=quant,
        capabilities=capabilities,
        ttft=ttft,
        tps=tps,
        error=error,
        runs=runs,
        modified_at=modified_at,
    )


class TestFormatJson:
    def test_basic_ttft_tps(self):
        row = _make_row()
        result = json.loads(
            format_json([row], num_runs=1, show_ttft=True, show_tps=True, verbose=False)
        )
        assert len(result) == 1
        assert result[0]["model"] == "llama3"
        assert result[0]["ttft"] == 1.23
        assert result[0]["tps"] == 45.6
        assert "error" not in result[0]

    def test_with_error(self):
        row = _make_row(error="timeout", ttft=None, tps=None)
        result = json.loads(
            format_json([row], num_runs=1, show_ttft=True, show_tps=True, verbose=False)
        )
        assert result[0]["error"] == "timeout"

    def test_no_ttft(self):
        row = _make_row()
        result = json.loads(
            format_json(
                [row], num_runs=1, show_ttft=False, show_tps=True, verbose=False
            )
        )
        assert "ttft" not in result[0]
        assert "tps" in result[0]

    def test_no_tps(self):
        row = _make_row()
        result = json.loads(
            format_json(
                [row], num_runs=1, show_ttft=True, show_tps=False, verbose=False
            )
        )
        assert "ttft" in result[0]
        assert "tps" not in result[0]

    def test_verbose_runs(self):
        row = _make_row(
            runs=[
                {"prompt": "p1", "ttft": 1.0, "tps": 40.0, "error": None},
                {"prompt": "p2", "ttft": 2.0, "tps": 35.0, "error": None},
            ]
        )
        result = json.loads(
            format_json([row], num_runs=2, show_ttft=True, show_tps=True, verbose=True)
        )
        assert "ttft_run_1" in result[0]
        assert "ttft_run_2" in result[0]
        assert "tps_run_1" in result[0]
        assert "tps_run_2" in result[0]

    def test_verbose_error_run(self):
        row = _make_row(
            runs=[
                {"prompt": "p1", "ttft": 1.0, "tps": 40.0, "error": "timeout"},
            ]
        )
        result = json.loads(
            format_json([row], num_runs=1, show_ttft=True, show_tps=True, verbose=True)
        )
        assert result[0]["ttft_run_1"] is None
        assert result[0]["tps_run_1"] is None

    def test_multiple_rows(self):
        rows = [_make_row(model="llama3"), _make_row(model="mistral", size="7B")]
        result = json.loads(
            format_json(rows, num_runs=1, show_ttft=True, show_tps=True, verbose=False)
        )
        assert len(result) == 2
        assert result[0]["model"] == "llama3"
        assert result[1]["model"] == "mistral"


class TestFormatCsv:
    def test_basic_ttft_tps(self):
        row = _make_row()
        result = format_csv(
            [row], num_runs=1, show_ttft=True, show_tps=True, verbose=False
        )
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "model" in lines[0]
        assert "ttft" in lines[0]
        assert "tps" in lines[0]
        assert "error" in lines[0]

    def test_no_ttft(self):
        row = _make_row()
        result = format_csv(
            [row], num_runs=1, show_ttft=False, show_tps=True, verbose=False
        )
        lines = result.strip().split("\n")
        assert "ttft" not in lines[0]
        assert "tps" in lines[0]

    def test_verbose_runs(self):
        row = _make_row(
            runs=[
                {"prompt": "p1", "ttft": 1.0, "tps": 40.0, "error": None},
                {"prompt": "p2", "ttft": 2.0, "tps": 35.0, "error": None},
            ]
        )
        result = format_csv(
            [row], num_runs=2, show_ttft=True, show_tps=True, verbose=True
        )
        lines = result.strip().split("\n")
        assert "ttft_run_1" in lines[0]
        assert "ttft_run_2" in lines[0]

    def test_none_values_empty_string(self):
        row = _make_row(ttft=None, tps=None, error=None)
        result = format_csv(
            [row], num_runs=1, show_ttft=True, show_tps=True, verbose=False
        )
        lines = result.strip().split("\n")
        data_line = lines[1].split(",")
        ttft_idx = lines[0].split(",").index("ttft")
        tps_idx = lines[0].split(",").index("tps")
        assert data_line[ttft_idx] == ""
        assert data_line[tps_idx] == ""

    def test_error_in_csv(self):
        row = _make_row(error="timeout")
        result = format_csv(
            [row], num_runs=1, show_ttft=True, show_tps=True, verbose=False
        )
        lines = result.strip().split("\n")
        assert "timeout" in lines[1]


class TestExportResults:
    def test_json_to_stdout(self, capsys):
        row = _make_row()
        export_results(
            [row],
            "json",
            None,
            num_runs=1,
            show_ttft=True,
            show_tps=True,
            verbose=False,
        )
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data[0]["model"] == "llama3"

    def test_csv_to_stdout(self, capsys):
        row = _make_row()
        export_results(
            [row], "csv", None, num_runs=1, show_ttft=True, show_tps=True, verbose=False
        )
        captured = capsys.readouterr()
        assert "model" in captured.out
        assert "llama3" in captured.out

    def test_json_to_file(self, tmp_path):
        row = _make_row()
        path = str(tmp_path / "output.json")
        export_results(
            [row],
            "json",
            path,
            num_runs=1,
            show_ttft=True,
            show_tps=True,
            verbose=False,
        )
        data = json.loads(Path(path).read_text())
        assert data[0]["model"] == "llama3"

    def test_csv_to_file(self, tmp_path):
        row = _make_row()
        path = str(tmp_path / "output.csv")
        export_results(
            [row], "csv", path, num_runs=1, show_ttft=True, show_tps=True, verbose=False
        )
        content = Path(path).read_text()
        assert "model" in content
        assert "llama3" in content
