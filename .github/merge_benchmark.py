#!/usr/bin/env python3
"""Append a benchmark snapshot to a time-series history file.

Usage: merge_benchmark.py <latest.json> <history.json>
"""

import json
import sys
from datetime import datetime, timezone

latest_path, history_path = sys.argv[1], sys.argv[2]

with open(latest_path) as f:
    content = f.read().strip()
    if not content:
        sys.exit(0)
    try:
        latest_results = json.loads(content)
    except json.JSONDecodeError:
        sys.exit(0)

try:
    with open(history_path) as f:
        history = json.load(f)
except FileNotFoundError, json.JSONDecodeError:
    history = {"runs": []}

history["runs"].append(
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": latest_results,
    }
)

with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
