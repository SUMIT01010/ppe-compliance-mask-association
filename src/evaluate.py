"""The experiment: run every arm over the composites and score the mechanism.

Arms (see src/association.py for why ARM A0 exists):
    A0  box IoU        @ 0.30   the naive rule; expected to collapse
    A   box containment @ 0.50  the honest baseline
    B   mask containment @ 0.50 the mechanism under test

All arms consume ONE detection pass per image. Masks are computed once per image and reused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .association import (Detection, associate_by_box_containment,
                          associate_by_box_iou, associate_by_mask_containment,
                          verdict_from_assignment)
from .metrics import PersonOutcome, match_persons, paired_bootstrap, stratified, summarise
from .perception import Perception

ARMS = ("A0_box_iou", "A_box_containment", "B_mask_containment")


def _run_arms(dets: list[Detection], iou_thr: float, cont_thr: float) -> dict[str, list]:
    return {
        "A0_box_iou": associate_by_box_iou(dets, iou_thr),
        "A_box_containment": associate_by_box_containment(dets, cont_thr),
        "B_mask_containment": associate_by_mask_containment(dets, cont_thr),
    }


def run(comp_dir: Path, weights: Path, device: str, iou_thr: float, cont_thr: float,
        strata, labels, limit: int | None = None) -> dict:
    records = json.loads((comp_dir / "composites.json").read_text())
    if limit:
        records = records[:limit]
    per = Perception(weights, device)

    outcomes: dict[str, list[PersonOutcome]] = {a: [] for a in ARMS}
    assoc_hits = {a: [0, 0] for a in ARMS}   # [correct, total] over matched GT helmets

    for n, rec in enumerate(records):
        img = np.array(Image.open(comp_dir / "images" / f"{rec['name']}.jpg").convert("RGB"))
        dets = per.detect(img)
        per.attach_person_masks(img, dets)

        pred_persons = [i for i, d in enumerate(dets) if d.cls == "person"]
        gt = rec["persons"]
        g2p = match_persons([dets[i].box for i in pred_persons],
                            [tuple(p["box"]) for p in gt])

        arms = _run_arms(dets, iou_thr, cont_thr)
        n_vis = len(gt)

        for arm, verdicts in arms.items():
            v_by_det = {v.person_idx: verdict_from_assignment(dets, v) for v in verdicts}
            for gi, g in enumerate(gt):
                matched = gi in g2p
                pred = None
                if matched:
                    det_idx = pred_persons[g2p[gi]]
                    pred = v_by_det[det_idx].compliant
                outcomes[arm].append(PersonOutcome(
                    composite=rec["name"], gid=g["gid"], n_visible=n_vis,
                    overlap=rec["overlap"], gt_compliant=g["compliant"],
                    matched=matched, pred_compliant=pred))

            # --- direct association accuracy: which person did each helmet get attached to? ---
            det2gt = {pred_persons[pi]: gi for gi, pi in g2p.items()}
            helmets = [i for i, d in enumerate(dets) if d.cls == "helmet"]
            owner_of_det = {}
            for v in verdicts:
                for j in v.assigned:
                    owner_of_det[j] = v.person_idx
            for j in helmets:
                # whose helmet is it really? the GT person whose helmet box this overlaps most
                best, best_gi = 0.0, None
                for gi, g in enumerate(gt):
                    for hb in g["helmet_boxes"]:
                        from .association import box_iou
                        v = box_iou(dets[j].box, tuple(hb))
                        if v > best:
                            best, best_gi = v, gi
                if best_gi is None or best < 0.5:
                    continue
                assoc_hits[arm][1] += 1
                owner_det = owner_of_det.get(j)
                if owner_det is not None and det2gt.get(owner_det) == best_gi:
                    assoc_hits[arm][0] += 1

        if (n + 1) % 50 == 0:
            print(f"  scored {n+1}/{len(records)}", flush=True)

    result = {
        "n_composites": len(records),
        "latency": per.latency_summary(),
        "arms": {},
    }
    for arm in ARMS:
        c, t = assoc_hits[arm]
        result["arms"][arm] = {
            "overall": summarise(outcomes[arm]),
            "by_crowding": stratified(outcomes[arm], strata, labels, key="n_visible"),
            "by_overlap": stratified(outcomes[arm], strata, labels, key="overlap"),
            "helmet_association_accuracy": (c / t) if t else None,
            "helmet_association_n": t,
        }

    result["headline"] = {
        "B_vs_A_strict": paired_bootstrap(outcomes["A_box_containment"],
                                          outcomes["B_mask_containment"], conditioned=False),
        "B_vs_A_conditioned": paired_bootstrap(outcomes["A_box_containment"],
                                               outcomes["B_mask_containment"], conditioned=True),
    }

    # The mechanism claim is that the gap GROWS with crowding. Report the per-stratum delta so
    # a flat profile refutes it visibly rather than being averaged away by the headline.
    result["delta_by_crowding"] = {}
    for lab in labels:
        a = [o for o in outcomes["A_box_containment"]
             if _in(o.n_visible, strata, labels, lab)]
        b = [o for o in outcomes["B_mask_containment"]
             if _in(o.n_visible, strata, labels, lab)]
        if a and b:
            result["delta_by_crowding"][lab] = paired_bootstrap(a, b, n_boot=1000)
    return result


def _in(n: int, strata, labels, lab: str) -> bool:
    for (lo, hi), l in zip(strata, labels):
        if l == lab:
            return lo <= n <= hi
    return False
