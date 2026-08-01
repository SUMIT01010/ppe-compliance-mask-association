"""Oracle-detector ablation: run the arms with PERFECT detections.

Why this exists as science, not as a workaround. The full experiment feeds both arms the same
YOLO detections, which controls for detector quality but does not remove it - a missed helmet
hurts both arms and compresses the measurable gap. Substituting ground-truth boxes for the
detector removes detection from the loop entirely and asks the mechanism question in its purest
form:

    when every person, head and helmet is known exactly, does mask containment assign PPE to the
    right person more often than box containment does, and does the gap grow with crowding?

This is an UPPER BOUND on the association gain, not the deployment number. If ARM B shows no
advantage here, it cannot show one with a real detector either, and the hypothesis is dead
regardless of how well the detector is trained. If it does show one, the trained-detector run
says how much of it survives contact with imperfect perception.

SAM still runs for real: it is prompted with the ground-truth person box and must segment the
composite image with no knowledge of how it was built.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .association import Detection
from .evaluate import ARMS, _run_arms
from .metrics import PersonOutcome, paired_bootstrap, stratified, summarise
from .association import verdict_from_assignment


def _oracle_detections(gt: list[dict]) -> tuple[list[Detection], list[int]]:
    """Build a detection list straight from ground truth.

    Returns (detections, person_det_index_per_gt) so each GT person maps to exactly one person
    detection - the matching step of the full pipeline becomes the identity, by construction.
    """
    dets: list[Detection] = []
    person_idx = []
    for g in gt:
        person_idx.append(len(dets))
        dets.append(Detection("person", tuple(g["box"]), 1.0))
    for g in gt:
        for hb in g["head_boxes"]:
            dets.append(Detection("head", tuple(hb), 1.0))
        for hb in g["helmet_boxes"]:
            dets.append(Detection("helmet", tuple(hb), 1.0))
    return dets, person_idx


def run_oracle(comp_dir: Path, device: str, iou_thr: float, cont_thr: float,
               strata, labels, limit: int | None = None,
               sam_weights: str = "mobile_sam.pt") -> dict:
    records = json.loads((comp_dir / "composites.json").read_text())
    if limit:
        records = records[:limit]

    from ultralytics import SAM
    sam = SAM(sam_weights)

    outcomes: dict[str, list[PersonOutcome]] = {a: [] for a in ARMS}
    assoc = {a: [0, 0] for a in ARMS}

    for n, rec in enumerate(records):
        img = np.array(Image.open(comp_dir / "images" / f"{rec['name']}.jpg").convert("RGB"))
        gt = rec["persons"]
        dets, pidx = _oracle_detections(gt)

        r = sam(img, bboxes=[list(d.box) for d in dets if d.cls == "person"],
                verbose=False, device=device)[0]
        if r.masks is not None:
            m = r.masks.data.cpu().numpy().astype(bool)
            for i, mask in zip(pidx, m):
                dets[i].mask = mask

        arms = _run_arms(dets, iou_thr, cont_thr)
        det2gt = {d: gi for gi, d in enumerate(pidx)}

        for arm, verdicts in arms.items():
            vmap = {v.person_idx: verdict_from_assignment(dets, v) for v in verdicts}
            for gi, g in enumerate(gt):
                outcomes[arm].append(PersonOutcome(
                    composite=rec["name"], gid=g["gid"], n_visible=len(gt),
                    overlap=rec["overlap"], gt_compliant=g["compliant"],
                    matched=True, pred_compliant=vmap[pidx[gi]].compliant))

            # helmet association: every helmet detection's true owner is known exactly here
            owner = {}
            for v in verdicts:
                for j in v.assigned:
                    owner[j] = v.person_idx
            k = len(gt)
            j = k
            for gi, g in enumerate(gt):
                j += len(g["head_boxes"])
                for _ in g["helmet_boxes"]:
                    assoc[arm][1] += 1
                    if det2gt.get(owner.get(j), -1) == gi:
                        assoc[arm][0] += 1
                    j += 1

        if (n + 1) % 50 == 0:
            print(f"  oracle {n+1}/{len(records)}", flush=True)

    out = {"n_composites": len(records), "mode": "oracle_detections", "arms": {}}
    for arm in ARMS:
        c, t = assoc[arm]
        out["arms"][arm] = {
            "overall": summarise(outcomes[arm]),
            "by_crowding": stratified(outcomes[arm], strata, labels, key="n_visible"),
            "helmet_association_accuracy": (c / t) if t else None,
            "helmet_association_n": t,
        }
    out["headline"] = paired_bootstrap(outcomes["A_box_containment"],
                                       outcomes["B_mask_containment"])
    out["delta_by_crowding"] = {}
    for lab in labels:
        from .evaluate import _in
        a = [o for o in outcomes["A_box_containment"] if _in(o.n_visible, strata, labels, lab)]
        b = [o for o in outcomes["B_mask_containment"] if _in(o.n_visible, strata, labels, lab)]
        if a and b:
            out["delta_by_crowding"][lab] = paired_bootstrap(a, b, n_boot=1000)
    return out
