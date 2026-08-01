"""Oracle-detector ablation: the association mechanism with detection removed from the loop.

Upper bound on the association gain. See src/oracle.py for why this is an ablation and not a
substitute for the trained-detector run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle import run_oracle  # noqa: E402
import config  # noqa: E402

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    res = run_oracle(comp_dir=config.DATASETS / "composites",
                     device=config.get_device(),
                     iou_thr=config.BOX_IOU_THRESHOLD,
                     cont_thr=config.CONTAINMENT_THRESHOLD,
                     strata=config.CROWDING_STRATA, labels=config.STRATA_LABELS,
                     limit=limit)
    out = config.OUTPUTS / "experiment_oracle.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"\n=== ORACLE ABLATION ({res['n_composites']} composites) ===")
    for arm, v in res["arms"].items():
        o = v["overall"]
        print(f"{arm:22s} nc-recall={o['noncompliance_recall_strict']:.4f}  "
              f"assoc={v['helmet_association_accuracy'] if v['helmet_association_accuracy'] is not None else float('nan'):.4f}  "
              f"undet={o['undetermined_rate']:.4f}  f1c={o['compliant_f1']:.4f}")
    h = res["headline"]
    print(f"\nB - A delta {h['delta']:+.4f}  CI [{h['ci95_lo']:+.4f},{h['ci95_hi']:+.4f}]  "
          f"P(>0)={h['p_delta_gt_0']:.3f}")
    print("\nby crowding:")
    for k, v in res["delta_by_crowding"].items():
        if v.get("delta") is not None:
            print(f"  {k:>4s}: {v['delta']:+.4f}  CI [{v['ci95_lo']:+.4f},{v['ci95_hi']:+.4f}]")
    print(f"\nwrote {out}")
