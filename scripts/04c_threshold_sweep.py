"""Threshold sweep for both arms + containment diagnostics.

Answers the question the single-threshold experiment cannot: is ARM B losing because masks are
the worse mechanism, or because a shared threshold is unfair to a quantity whose range is
compressed? See src/sweep.py for the argument.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sweep import run_sweep  # noqa: E402
import config  # noqa: E402

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    res = run_sweep(comp_dir=config.DATASETS / "composites",
                    device=config.get_device(), thresholds=THRESHOLDS,
                    strata=config.CROWDING_STRATA, labels=config.STRATA_LABELS,
                    limit=limit)
    out = config.OUTPUTS / "threshold_sweep.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"\n=== CONTAINMENT OF THE TRUE OWNER ({res['n_composites']} composites) ===")
    c = res["containment_of_true_owner"]
    print(f"{'':14s} {'mean':>7s} {'median':>7s} {'p05':>7s} {'<0.50':>7s}  n")
    for k in ("helmet_box", "helmet_mask", "head_box", "head_mask"):
        d = c[k]
        print(f"{k:14s} {d['mean']:7.3f} {d['median']:7.3f} {d['p05']:7.3f} "
              f"{d['frac_below_0.50']:7.3f}  {d['n']}")

    print("\n=== SWEEP: non-compliance recall (strict) by threshold ===")
    print(f"{'thr':>5s}  {'ARM A (box)':>12s}  {'ARM B (mask)':>12s}   {'A f1c':>6s} {'B f1c':>6s}")
    for pa, pb in zip(res["curve"]["A_box"], res["curve"]["B_mask"]):
        ra = pa["noncompliance_recall_strict"]
        rb = pb["noncompliance_recall_strict"]
        print(f"{pa['threshold']:5.2f}  {ra:12.4f}  {rb:12.4f}   "
              f"{(pa['compliant_f1'] or 0):6.3f} {(pb['compliant_f1'] or 0):6.3f}")

    b = res["best"]
    print(f"\nbest ARM A: t={b['A_box']['threshold']:.2f} "
          f"recall={b['A_box']['noncompliance_recall_strict']:.4f} "
          f"f1c={b['A_box']['compliant_f1']:.4f}")
    print(f"best ARM B: t={b['B_mask']['threshold']:.2f} "
          f"recall={b['B_mask']['noncompliance_recall_strict']:.4f} "
          f"f1c={b['B_mask']['compliant_f1']:.4f}")
    h = res["best_vs_best"]
    print(f"\nBEST-vs-BEST  B - A = {h['delta']:+.4f}  95% CI [{h['ci95_lo']:+.4f}, "
          f"{h['ci95_hi']:+.4f}]  P(delta>0)={h['p_delta_gt_0']:.3f}")
    print("\nby crowding (each arm at its own best threshold):")
    for k, v in res["best_vs_best_by_crowding"].items():
        if v.get("delta") is not None:
            print(f"  {k:>4s}: {v['delta']:+.4f}  CI [{v['ci95_lo']:+.4f}, {v['ci95_hi']:+.4f}]")
    print(f"\nwrote {out}")
