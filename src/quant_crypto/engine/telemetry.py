"""Live engine telemetry (dashboard concordance).

The dashboard must never guess. When a PaperEngine session runs, it publishes
real counters/state here (module-global, same process as the control plane);
GET /api/v1/status exposes them and the UI binds to them. No session running
=> snapshot is None and the UI shows idle, never fabricated numbers.
"""

from __future__ import annotations

from typing import Any

_ENGINE: dict[str, Any] = {}


def publish_engine(snap: dict[str, Any]) -> None:
    """Merge a live-engine snapshot (called by PaperEngine at key moments)."""
    _ENGINE.update(snap)
    _ENGINE["updated_at"] = snap.get("updated_at")


def clear_engine() -> None:
    _ENGINE.clear()


def engine_snapshot() -> dict[str, Any] | None:
    return _ENGINE or None


def build_pipeline_steps(s: dict[str, Any]) -> list[dict[str, str]]:
    """Real procedure-pipeline rows derived from live counters (not cosmetic)."""
    halted = bool(s.get("halted"))
    phase = s.get("phase", "idle")
    if not s:
        return [{"name": "Engine session", "sub": "idle — nothing running", "status": "idle"}]

    def st(active: bool, done: bool) -> str:
        if halted:
            return "halted"
        return "active" if active else ("completed" if done else "pending")

    ticks = int(s.get("ticks_processed", 0))
    inferences = int(s.get("inferences", 0))
    dropped = int(s.get("viability_dropped", 0))
    entered = int(s.get("entered_trades", 0))
    return [
        {"name": "L2 Ingest", "sub": f"{ticks} ticks", "status": st(phase == "streaming" and ticks == 0, ticks > 0)},
        {"name": "Mamba Inference", "sub": f"{inferences} passes", "status": st(False, inferences > 0)},
        {"name": "Viability Guard", "sub": f"{dropped} dropped", "status": st(inferences > 0 and entered == 0 and dropped == 0, inferences > 0)},
        {"name": "FOK Execution", "sub": f"{entered} entries", "status": st(entered > 0 and phase == "streaming", entered > 0)},
        {"name": "Risk Rails", "sub": "halted" if halted else "armed", "status": "halted" if halted else "active"},
        {"name": "Shadow PnL", "sub": f"{s.get('total_pnl', 0.0):.2f} pts", "status": st(False, "total_pnl" in s)},
    ]
