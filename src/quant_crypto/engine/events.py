"""Process-wide dashboard event feed (Stage D-feed).

Silent rejections are the enemy of algorithmic trading: when the guard drops
77 signals, the operator must SEE it. The engine publishes feed events here;
the control API exposes them via GET /api/v1/feed?since=<seq>; the dashboard
polls and renders them in the AI suggestion feed with distinct colors.
"""

from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_EVENTS: list[dict] = []
_SEQ = 0
_CAP = 200


def publish_feed_event(kind: str, sev: str, color: str, txt: str) -> int:
    """Append an event to the ring; returns its seq. Thread-safe."""
    global _SEQ
    with _LOCK:
        _SEQ += 1
        _EVENTS.append(
            {
                "seq": _SEQ,
                "ts": round(time.time(), 3),
                "kind": kind,
                "sev": sev,
                "color": color,  # drop -> muted warn, ok -> info, error -> red
                "txt": txt,
            }
        )
        if len(_EVENTS) > _CAP:
            del _EVENTS[: len(_EVENTS) - _CAP]
        return _SEQ


def feed_events_since(seq: int) -> tuple[list[dict], int]:
    """Events with seq > `seq`, plus the current latest seq."""
    with _LOCK:
        events = [e for e in _EVENTS if e["seq"] > seq]
        return events, _SEQ


def reset_feed_events() -> None:
    global _SEQ
    with _LOCK:
        _EVENTS.clear()
        _SEQ = 0
