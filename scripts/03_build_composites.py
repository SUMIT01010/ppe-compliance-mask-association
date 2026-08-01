"""Build the synthetic crowded evaluation set with exact person<->PPE ground truth.

Sources are HELD-OUT single-person images only (SH17's official val split), so the detector has
never seen any person that appears in the evaluation set.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.composites import build_dataset  # noqa: E402
import config  # noqa: E402

if __name__ == "__main__":
    root = config.DATASETS / "sh17_p7"
    manifest = json.loads((root / "manifest.json").read_text())
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    build_dataset(out=config.DATASETS / "composites",
                  img_dir=root / "heldout" / "images",
                  lbl_dir=root / "heldout" / "labels",
                  manifest=manifest, n=n, seed=config.SEED)
