from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_NAME = "ometer_history.db"

IMPROVEMENT_THRESHOLD = 0.05


def get_db_path() -> Path:
    env_path = os.getenv("OMETER_HISTORY_DB")
    if env_path:
        return Path(env_path)
    data_home = os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")
    return Path(data_home) / "ometer" / DB_NAME


def get_connection() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id              TEXT PRIMARY KEY,
            timestamp       TEXT NOT NULL,
            model_name      TEXT NOT NULL,
            model_size      TEXT,
            context_length  INTEGER,
            quantization    TEXT,
            capabilities    TEXT,
            ttft            REAL,
            tps             REAL,
            error           TEXT,
            mode            TEXT,
            prompts         TEXT,
            num_predict     INTEGER,
            parallel        INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_benchmark_runs_model
        ON benchmark_runs(model_name)
    """)
    conn.commit()


def save_run(conn: sqlite3.Connection, run_data: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO benchmark_runs
            (id, timestamp, model_name, model_size, context_length,
             quantization, capabilities, ttft, tps, error, mode,
             prompts, num_predict, parallel)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            run_data["id"],
            run_data["timestamp"],
            run_data["model_name"],
            run_data.get("model_size"),
            run_data.get("context_length"),
            run_data.get("quantization"),
            run_data.get("capabilities"),
            run_data.get("ttft"),
            run_data.get("tps"),
            run_data.get("error"),
            run_data.get("mode"),
            json.dumps(run_data.get("prompts", [])),
            run_data.get("num_predict"),
            run_data.get("parallel"),
        ),
    )
    conn.commit()


def get_latest_per_model(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    rows = conn.execute("""
        SELECT model_name, timestamp, model_size, context_length,
               quantization, capabilities, ttft, tps, error, mode,
               prompts, num_predict, parallel, id
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY model_name ORDER BY timestamp DESC
            ) as rn
            FROM benchmark_runs
        )
        WHERE rn = 1
        ORDER BY timestamp DESC
    """).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_model_history(
    conn: sqlite3.Connection, model_name: str, limit: int = 20
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT * FROM benchmark_runs
        WHERE model_name = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """,
        (model_name, limit),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_previous_run(
    conn: sqlite3.Connection, model_name: str
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT * FROM benchmark_runs
        WHERE model_name = ?
        ORDER BY timestamp DESC
        LIMIT 1 OFFSET 1
    """,
        (model_name,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d: dict[str, Any] = dict(row)
    d["prompts"] = json.loads(d["prompts"]) if d.get("prompts") else []
    return d


def trend_arrow(current: float, previous: float | None, lower_is_better: bool) -> str:
    if previous is None:
        return "\u2192"
    if previous == 0:
        return "\u2192"
    diff = (current - previous) / abs(previous)
    if abs(diff) < IMPROVEMENT_THRESHOLD:
        return "\u2192"
    improved = (diff < 0 and lower_is_better) or (diff > 0 and not lower_is_better)
    return "\u2191" if improved else "\u2193"


def build_run_data(
    model_name: str,
    model_size: str,
    context_length: int,
    quantization: str | None,
    capabilities: str,
    ttft: float | None,
    tps: float | None,
    error: str | None,
    mode: str,
    prompts: list[str],
    num_predict: int | None,
    parallel: int,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_size": model_size,
        "context_length": context_length,
        "quantization": quantization or "",
        "capabilities": capabilities,
        "ttft": ttft,
        "tps": tps,
        "error": error,
        "mode": mode,
        "prompts": prompts,
        "num_predict": num_predict,
        "parallel": parallel,
    }
