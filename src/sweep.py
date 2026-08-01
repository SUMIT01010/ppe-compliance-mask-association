"""Threshold sweep + containment diagnostics: is the shared threshold fair to ARM B?

# === CORE CONTRIBUTION ===  (this ablation and the reason it is necessary)

WHY THIS EXISTS
---------------
The main experiment gives ARM A and ARM B one shared threshold (`config.CONTAINMENT_THRESHOLD`)
so the arms differ in exactly one thing - box region vs mask region - and the result is
attributable to the mechanism rather than to threshold search. That is the right instinct and it
has a flaw that only shows up once you look at the geometry:

    a person's segmentation mask is (very nearly) a SUBSET of their bounding box,

so for any PPE box, `mask_containment <= box_containment`, always. The two arms are therefore NOT
being scored on the same scale. A shared threshold is systematically harsher on ARM B, and "ARM B
loses at t=0.50" could mean either "masks are the worse mechanism" or "0.50 is the wrong operating
point for a quantity whose range is compressed".

Those two explanations have opposite conclusions, so the experiment cannot end at a single
threshold. This module sweeps t for BOTH arms and reports each arm's best achievable operating
point. If ARM B's best is still below ARM A's best, the hypothesis is refuted robustly rather
than by an artifact of a number chosen in advance.

WHAT ELSE IT MEASURES
---------------------
The containment score of each ground-truth (helmet, its true owner) pair under both rules. This
is the diagnostic that explains *why* whichever arm wins, wins - a distribution, not a single
number. A helmet sits on top of a head; whether SAM's person mask includes it is the entire
question, and this measures it directly instead of inferring it from the headline.

Runs on ORACLE detections (ground-truth boxes) so detection quality cannot muddy the picture.
SAM still segments the real composite image and has no knowledge of how it was built.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .association import (Detection, box_containment, mask_containment,
                          verdict_from_assignment, _assign)
from .metrics import PersonOutcome, paired_bootstrap, stratified, summarise
from .oracle import _oracle_detections


def _score_arm(dets: list[Detection], use_mask: bool, threshold: float):
    """One arm at one threshold. Same argmax rule for both; only the region changes."""
    if use_mask:
        def score(ppe: Detection, per: Detection) -> float:
            if per.mask is None:
                return box_containment(ppe.box, per.box)
            return mask_containment(ppe.box, per.mask)
    else:
        def score(ppe: Detection, per: Detection) -> float:
            return box_containment(ppe.box, per.box)
    return _assign(dets, score, threshold)


def run_sweep(comp_dir: Path, device: str, thresholds, strata, labels,
              limit: int | None = None, sam_weights: str = "mobile_sam.pt") -> dict:
    records = json.loads((comp_dir / "composites.json").read_text())
    if limit:
        records = records[:limit]

    from ultralytics import SAM
    sam = SAM(sam_weights)

    # outcomes[(arm, threshold)] -> list[PersonOutcome]
    outcomes: dict[tuple[str, float], list[PersonOutcome]] = {
        (arm, t): [] for arm in ("A_box", "B_mask") for t in thresholds}
    # (helmet, true owner) containment under each rule - the mechanism diagnostic
    pair_scores: list[dict] = []

    for n, rec in enumerate(records):
        img = np.array(Image.open(comp_dir / "images" / f"{rec['name']}.jpg").convert("RGB"))
        gt = rec["persons"]
        dets, pidx = _oracle_detections(gt)

        r = sam(img, bboxes=[list(dets[i].box) for i in pidx], verbose=False, device=device)[0]
        if r.masks is not None:
            m = r.masks.data.cpu().numpy().astype(bool)
            for i, mask in zip(pidx, m):
                dets[i].mask = mask

        # --- diagnostic: score every PPE box against its KNOWN owner, both ways ---
        j = len(gt)
        for gi, g in enumerate(gt):
            owner = dets[pidx[gi]]
            for cls, boxes in (("head", g["head_boxes"]), ("helmet", g["helmet_boxes"])):
                for hb in boxes:
                    bc = box_containment(tuple(hb), owner.box)
                    mc = (mask_containment(tuple(hb), owner.mask)
                          if owner.mask is not None else None)
                    pair_scores.append({"cls": cls, "box_containment": bc,
                                        "mask_containment": mc, "n_visible": len(gt)})
                    j += 1

        for t in thresholds:
            for arm, use_mask in (("A_box", False), ("B_mask", True)):
                verdicts = _score_arm(dets, use_mask, t)
                vmap = {v.person_idx: verdict_from_assignment(dets, v) for v in verdicts}
                for gi, g in enumerate(gt):
                    outcomes[(arm, t)].append(PersonOutcome(
                        composite=rec["name"], gid=g["gid"], n_visible=len(gt),
                        overlap=rec["overlap"], gt_compliant=g["compliant"],
                        matched=True, pred_compliant=vmap[pidx[gi]].compliant))

        if (n + 1) % 50 == 0:
            print(f"  sweep {n+1}/{len(records)}", flush=True)

    # --- assemble ---------------------------------------------------------
    curve = {"A_box": [], "B_mask": []}
    for arm in ("A_box", "B_mask"):
        for t in thresholds:
            s = summarise(outcomes[(arm, t)])
            curve[arm].append({
                "threshold": t,
                "noncompliance_recall_strict": s["noncompliance_recall_strict"],
                "compliant_f1": s["compliant_f1"],
                "undetermined_rate": s["undetermined_rate"],
                "verdict_accuracy": s["verdict_accuracy"],
            })

    def best(arm: str) -> dict:
        # Best operating point = highest non-compliance recall, ties broken by compliant F1 so a
        # degenerate "call everyone non-compliant" point cannot win.
        pts = [p for p in curve[arm] if p["noncompliance_recall_strict"] is not None]
        return max(pts, key=lambda p: (p["noncompliance_recall_strict"], p["compliant_f1"] or 0.0))

    ba, bb = best("A_box"), best("B_mask")
    a_out = outcomes[("A_box", ba["threshold"])]
    b_out = outcomes[("B_mask", bb["threshold"])]

    hb = [p for p in pair_scores if p["cls"] == "helmet" and p["mask_containment"] is not None]
    hd = [p for p in pair_scores if p["cls"] == "head" and p["mask_containment"] is not None]

    def dist(xs, key):
        v = np.array([p[key] for p in xs], dtype=float)
        return {"n": int(v.size), "mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
                "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                "frac_below_0.50": float((v < 0.50).mean())}

    return {
        "n_composites": len(records),
        "thresholds": list(thresholds),
        "curve": curve,
        "best": {"A_box": ba, "B_mask": bb},
        # The decisive comparison: each arm at ITS OWN best threshold. If B still loses here, the
        # shared-threshold result was not an artifact.
        "best_vs_best": paired_bootstrap(a_out, b_out),
        "best_vs_best_by_crowding": {
            lab: paired_bootstrap([o for o in a_out if _in_stratum(o.n_visible, strata, labels, lab)],
                                  [o for o in b_out if _in_stratum(o.n_visible, strata, labels, lab)],
                                  n_boot=1000)
            for lab in labels
            if any(_in_stratum(o.n_visible, strata, labels, lab) for o in a_out)
        },
        "containment_of_true_owner": {
            "helmet_box": dist(hb, "box_containment"),
            "helmet_mask": dist(hb, "mask_containment"),
            "head_box": dist(hd, "box_containment"),
            "head_mask": dist(hd, "mask_containment"),
        },
        "stratified_best_B": stratified(b_out, strata, labels, key="n_visible"),
        "stratified_best_A": stratified(a_out, strata, labels, key="n_visible"),
    }


def _in_stratum(n: int, strata, labels, lab: str) -> bool:
    for (lo, hi), l in zip(strata, labels):
        if l == lab:
            return lo <= n <= hi
    return False
