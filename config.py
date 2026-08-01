"""Central config + device helper for project-7-ppe-compliance.

Import discipline (see skills/uv-python311-ml-env-on-mac.md):
  - torch is NOT imported at module top level. It is lazy-imported inside get_device().
    Config modules that import torch eagerly are what turn the torch/tree-lib segfault on this
    Mac from avoidable into unavoidable.
  - No xgboost / lightgbm / catboost anywhere in this project's processes.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATASETS = ROOT / "datasets"
OUTPUTS = ROOT / "outputs"
RUNS = ROOT / "runs"
CHECKPOINTS = ROOT / "checkpoints"

for _d in (DATA, DATASETS, OUTPUTS, RUNS, CHECKPOINTS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Experiment constants -------------------------------------------------
# Fixed before any training run. Changing these after seeing results invalidates
# the hypothesis test (PROJECT_METHODOLOGY.md section 4).

SEED = 42
CROWDING_STRATA = [(1, 1), (2, 3), (4, 6), (7, 10**6)]  # persons per frame, inclusive
STRATA_LABELS = ["1", "2-3", "4-6", "7+"]

# Association thresholds.
#
# ARM A and ARM B deliberately SHARE one threshold (CONTAINMENT_THRESHOLD). They must, for the
# comparison to isolate the mechanism: the arms differ only in whether the person region is a
# box or a mask. Giving each arm its own tuned threshold would confound the mechanism with
# threshold tuning and the result would not be attributable to segmentation.
CONTAINMENT_THRESHOLD = 0.50   # ARM A & ARM B: fraction of the PPE box inside the person region
BOX_IOU_THRESHOLD = 0.30       # ARM A0 only: the naive IoU rule, kept to show that it collapses

# Splits come from SH17's own train_files.txt / val_files.txt - see src/sh17.py for the
# protocol. There is no fractional re-split; the authors' boundary is never crossed.

# --- Device ---------------------------------------------------------------

def get_device(flag: str | None = None) -> str:
    """Resolve the compute device.

    Honours the DEVICE env var so the same code runs `mps` locally and `cuda` on a cloud box
    (this is the flag CLOUD_RUN.md flips). Never hardcode "mps" at a call site.
    """
    flag = flag or os.environ.get("DEVICE", "auto")
    if flag != "auto":
        return flag

    import torch  # lazy on purpose - see module docstring

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # MPS has float64 gaps and occasional missing ops. Keep training in float32;
        # if an op errors, set PYTORCH_ENABLE_MPS_FALLBACK=1 rather than rewriting the model.
        return "mps"
    return "cpu"
