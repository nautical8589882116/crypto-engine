"""JSONL tape persistence and time/gap utilities (mirrors parent engine)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TapeWriter:
    """Append-only JSONL writer. buffer_size=0 flushes every record."""

    def __init__(self, path: str | Path, buffer_size: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
        self._buffer_size = buffer_size
        self._buf: list[str] = []
        self._count = 0

    def write(self, model: Any) -> None:
        line = model.model_dump_json() if hasattr(model, "model_dump_json") else json.dumps(model)
        self._buf.append(line)
        if self._buffer_size <= 0 or len(self._buf) >= self._buffer_size:
            self.flush()
        self._count += 1

    def flush(self) -> None:
        if self._buf:
            self._fh.write("\n".join(self._buf) + "\n")
            self._buf.clear()
            self._fh.flush()

    def close(self) -> None:
        self.flush()
        self._fh.close()

    @property
    def count(self) -> int:
        return self._count


def parse_ts(raw: Any) -> float:
    """Accept epoch int/float or ISO-8601; fall back to now."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip()
        try:
            return float(s)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def find_gaps(timestamps: list[float], max_gap_s: float = 5.0) -> list[tuple[float, float]]:
    """Return (start, end) pairs where the feed was silent longer than max_gap_s."""
    gaps = []
    for a, b in zip(timestamps, timestamps[1:]):
        if b - a > max_gap_s:
            gaps.append((a, b))
    return gaps
