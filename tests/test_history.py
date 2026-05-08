from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from ometer.history import (
    IMPROVEMENT_THRESHOLD,
    _init_db,
    build_run_data,
    get_db_path,
    get_latest_per_model,
    get_model_history,
    get_previous_run,
    save_run,
    trend_arrow,
)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(str(tmp_path / "test.db"))
    db.row_factory = sqlite3.Row
    _init_db(db)
    yield db
    db.close()


def _make_run(
    model_name: str = "llama3",
    ttft: float | None = 1.0,
    tps: float | None = 20.0,
    error: str | None = None,
    timestamp: str = "2025-01-01T00:00:00+00:00",
    mode: str = "local",
) -> dict:
    data = build_run_data(
        model_name=model_name,
        model_size="7B",
        context_length=8192,
        quantization="Q4_0",
        capabilities="completion",
        ttft=ttft,
        tps=tps,
        error=error,
        mode=mode,
        prompts=["why is the sky blue?"],
        num_predict=None,
        parallel=1,
    )
    data["timestamp"] = timestamp
    return data


class TestInitDb:
    def test_creates_table(self, conn: sqlite3.Connection):
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='benchmark_runs'"
        ).fetchone()
        assert result is not None

    def test_creates_index(self, conn: sqlite3.Connection):
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_benchmark_runs_model'"
        ).fetchone()
        assert result is not None

    def test_idempotent(self, conn: sqlite3.Connection):
        _init_db(conn)
        _init_db(conn)
        result = conn.execute(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table'"
        ).fetchone()
        assert result["cnt"] >= 1


class TestSaveAndRetrieve:
    def test_save_and_load_latest(self, conn: sqlite3.Connection):
        run = _make_run("llama3")
        save_run(conn, run)
        latest = get_latest_per_model(conn)
        assert len(latest) == 1
        assert latest[0]["model_name"] == "llama3"
        assert latest[0]["ttft"] == 1.0
        assert latest[0]["tps"] == 20.0

    def test_latest_per_model(self, conn: sqlite3.Connection):
        old = _make_run("llama3", timestamp="2025-01-01T00:00:00+00:00")
        new = _make_run(
            "llama3", ttft=0.5, tps=30.0, timestamp="2025-06-01T00:00:00+00:00"
        )
        other = _make_run("mistral", timestamp="2025-03-01T00:00:00+00:00")
        for r in (old, new, other):
            save_run(conn, r)
        latest = get_latest_per_model(conn)
        by_name = {r["model_name"]: r for r in latest}
        assert by_name["llama3"]["ttft"] == 0.5
        assert by_name["mistral"]["ttft"] == 1.0

    def test_model_history(self, conn: sqlite3.Connection):
        run1 = _make_run("llama3", timestamp="2025-01-01T00:00:00+00:00")
        run2 = _make_run("llama3", ttft=0.5, timestamp="2025-06-01T00:00:00+00:00")
        save_run(conn, run1)
        save_run(conn, run2)
        history = get_model_history(conn, "llama3")
        assert len(history) == 2
        assert history[0]["ttft"] == 0.5

    def test_model_history_limit(self, conn: sqlite3.Connection):
        for i in range(5):
            r = _make_run(
                "llama3", ttft=float(i), timestamp=f"2025-0{i+1}-01T00:00:00+00:00"
            )
            save_run(conn, r)
        history = get_model_history(conn, "llama3", limit=3)
        assert len(history) == 3

    def test_previous_run(self, conn: sqlite3.Connection):
        run1 = _make_run("llama3", timestamp="2025-01-01T00:00:00+00:00")
        run2 = _make_run("llama3", ttft=0.5, timestamp="2025-06-01T00:00:00+00:00")
        save_run(conn, run1)
        save_run(conn, run2)
        prev = get_previous_run(conn, "llama3")
        assert prev is not None
        assert prev["ttft"] == 1.0

    def test_previous_run_none(self, conn: sqlite3.Connection):
        run = _make_run("llama3")
        save_run(conn, run)
        prev = get_previous_run(conn, "llama3")
        assert prev is None

    def test_previous_run_no_history(self, conn: sqlite3.Connection):
        prev = get_previous_run(conn, "nonexistent")
        assert prev is None

    def test_empty_latest(self, conn: sqlite3.Connection):
        assert get_latest_per_model(conn) == []


class TestBuildRunData:
    def test_returns_all_expected_fields(self):
        data = build_run_data(
            model_name="llama3:8b",
            model_size="8B",
            context_length=4096,
            quantization="Q4_0",
            capabilities="completion,vision",
            ttft=1.5,
            tps=25.0,
            error=None,
            mode="local",
            prompts=["hello", "world"],
            num_predict=200,
            parallel=3,
        )
        assert data["model_name"] == "llama3:8b"
        assert data["model_size"] == "8B"
        assert data["context_length"] == 4096
        assert data["quantization"] == "Q4_0"
        assert data["capabilities"] == "completion,vision"
        assert data["ttft"] == 1.5
        assert data["tps"] == 25.0
        assert data["error"] is None
        assert data["mode"] == "local"
        assert data["num_predict"] == 200
        assert data["parallel"] == 3
        assert isinstance(data["id"], str)
        assert "timestamp" in data

    def test_error_field(self):
        data = build_run_data(
            model_name="m",
            model_size="",
            context_length=0,
            quantization="",
            capabilities="",
            ttft=None,
            tps=None,
            error="timeout",
            mode="cloud",
            prompts=[],
            num_predict=None,
            parallel=1,
        )
        assert data["error"] == "timeout"

    def test_id_is_uuid(self):
        data = build_run_data(
            model_name="m",
            model_size="",
            context_length=0,
            quantization="",
            capabilities="",
            ttft=None,
            tps=None,
            error=None,
            mode="local",
            prompts=[],
            num_predict=None,
            parallel=1,
        )
        assert len(data["id"]) == 36

    def test_prompts_stored_as_list(self, conn: sqlite3.Connection):
        prompts = ["prompt1", "prompt2", "prompt3"]
        data = build_run_data(
            model_name="m",
            model_size="",
            context_length=0,
            quantization="",
            capabilities="",
            ttft=1.0,
            tps=10.0,
            error=None,
            mode="local",
            prompts=prompts,
            num_predict=None,
            parallel=1,
        )
        save_run(conn, data)
        row = conn.execute("SELECT prompts FROM benchmark_runs").fetchone()
        loaded = json.loads(row["prompts"])
        assert loaded == prompts


class TestTrendArrow:
    def test_improved_ttft(self):
        assert trend_arrow(0.5, 1.0, lower_is_better=True) == "\u2191"

    def test_degraded_ttft(self):
        assert trend_arrow(1.5, 1.0, lower_is_better=True) == "\u2193"

    def test_improved_tps(self):
        assert trend_arrow(30.0, 20.0, lower_is_better=False) == "\u2191"

    def test_degraded_tps(self):
        assert trend_arrow(10.0, 20.0, lower_is_better=False) == "\u2193"

    def test_stable_within_threshold(self):
        small_diff = IMPROVEMENT_THRESHOLD * 0.5
        assert trend_arrow(1.0, 1.0 + small_diff, lower_is_better=True) == "\u2192"

    def test_no_previous(self):
        assert trend_arrow(1.0, None, lower_is_better=True) == "\u2192"

    def test_zero_previous(self):
        assert trend_arrow(1.0, 0.0, lower_is_better=True) == "\u2192"

    def test_at_exact_threshold_not_arrow(self):
        assert (
            trend_arrow(1.0, 1.0 / (1 + IMPROVEMENT_THRESHOLD), lower_is_better=True)
            != "\u2192"
        )


class TestGetDbPath:
    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("OMETER_HISTORY_DB", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        path = get_db_path()
        assert path.name == "ometer_history.db"
        assert "ometer" in str(path)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OMETER_HISTORY_DB", "/tmp/custom_history.db")
        path = get_db_path()
        assert str(path) == "/tmp/custom_history.db"

    def test_xdg_data_home_override(self, monkeypatch):
        monkeypatch.delenv("OMETER_HISTORY_DB", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", "/custom/xdg")
        path = get_db_path()
        assert "/custom/xdg/ometer" in str(path)
