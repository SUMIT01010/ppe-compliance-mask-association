"""Run ARM A0 / ARM A / ARM B over the composites and write the result JSON.

This is the script that decides whether the project's hypothesis holds.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import run  # noqa: E402
import config  # noqa: E402

if __name__ == "__main__":
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        config.RUNS / "detector_demo" / "weights" / "best.pt")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if not weights.exists():
        raise SystemExit(f"no detector weights at {weights} - run 02_train_detector.py first")

    res = run(comp_dir=config.DATASETS / "composites", weights=weights,
              device=config.get_device(),
              iou_thr=config.BOX_IOU_THRESHOLD,
              cont_thr=config.CONTAINMENT_THRESHOLD,
              strata=config.CROWDING_STRATA, labels=config.STRATA_LABELS,
              limit=limit)
    out = config.OUTPUTS / "experiment.json"
    out.write_text(json.dumps(res, indent=2))

    h = res["headline"]["B_vs_A_strict"]
    print("\n=== HEADLINE: non-compliance recall (strict) ===")
    print(f"  ARM A (box containment)  {h['arm_a']:.4f}")
    print(f"  ARM B (mask containment) {h['arm_b']:.4f}")
    print(f"  delta {h['delta']:+.4f}  95% CI [{h['ci95_lo']:+.4f}, {h['ci95_hi']:+.4f}]  "
          f"P(delta>0)={h['p_delta_gt_0']:.3f}")
    print("\n=== delta by crowding stratum ===")
    for k, v in res["delta_by_crowding"].items():
        if v.get("delta") is not None:
            print(f"  {k:>4s} persons: {v['delta']:+.4f}  CI [{v['ci95_lo']:+.4f}, {v['ci95_hi']:+.4f}]")
    print(f"\nwrote {out}")
