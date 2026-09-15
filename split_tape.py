"""Split a multi-symbol tape into per-symbol tapes, then build corpus windows.

Mixing symbols in one window is wrong (BTC and ETH tick dynamics differ), so
each symbol gets its own tape and its own .npz corpus.
"""

import json
import sys
from pathlib import Path

# split
src = Path("/opt/data/corpus_tapes/latest.jsonl")
out_dir = Path("/opt/data/corpus_src")
out_dir.mkdir(parents=True, exist_ok=True)
handles = {}
counts = {}
for line in src.open():
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    sym = d.get("symbol", "?")
    if sym not in handles:
        handles[sym] = (out_dir / f"{sym}.jsonl").open("w")
        counts[sym] = 0
    handles[sym].write(line + "\n")
    counts[sym] += 1
for h in handles.values():
    h.close()
print("split counts:", counts, file=sys.stderr)
