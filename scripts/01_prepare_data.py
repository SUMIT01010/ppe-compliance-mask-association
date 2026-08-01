"""Downscale SH17 + remap to the 4 project classes. One-time, ~10 min on 8 cores.

The __main__ guard is load-bearing: macOS uses the spawn start method, so every worker
re-imports this module. Without it each worker re-runs prepare() and the pool dies with
BrokenProcessPool.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prepare import prepare  # noqa: E402
import config  # noqa: E402

if __name__ == "__main__":
    t = time.time()
    m = prepare(config.DATA / "archive", config.DATASETS / "sh17_p7", workers=8)
    print(f"\ndone: {len(m)} images in {time.time()-t:.0f}s -> {config.DATASETS/'sh17_p7'}")
