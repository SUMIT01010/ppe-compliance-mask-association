"""Scoring: match predictions to composite ground truth, then stratify and test.

Two recall definitions are reported and they answer different questions:

  strict      - over ALL ground-truth non-compliant persons, counting detector misses as
                failures. This is the deployment-relevant number: a non-compliant worker who was
                never detected is just as unprotected as one who was mis-associated.
  conditioned - over only those persons the detector actually found. This isolates the
                ASSOCIATION step, which is the mechanism under test.

Both arms consume identical detections, so detector misses are identical across arms and cannot
bias the A-vs-B delta either way.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .association import box_iou


@dataclass
class PersonOutcome:
    composite: str
    gid: int
    n_visible: int          # persons in this composite -> crowding stratum
    overlap: float
    gt_compliant: bool
    matched: bool           # did the detector find this person at all
    pred_compliant: bool | None   # None = undetermined


def match_persons(pred_boxes: list[tuple], gt_boxes: list[tuple], thr: float = 0.5) -> dict[int, int]:
    """Greedy IoU matching, highest first. Returns {gt_index: pred_index}."""
    pairs = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            iou = box_iou(g, p)
            if iou >= thr:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, out = set(), set(), {}
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        out[gi] = pi
        used_g.add(gi)
        used_p.add(pi)
    return out


def stratum_of(n: int, strata: list[tuple[int, int]], labels: list[str]) -> str:
    for (lo, hi), lab in zip(strata, labels):
        if lo <= n <= hi:
            return lab
    return labels[-1]


def _recall(outcomes: list[PersonOutcome], conditioned: bool) -> float | None:
    """Non-compliance recall: of the persons who are genuinely non-compliant, what fraction did
    we flag as non-compliant? Undetermined counts as a MISS - failing to raise the alarm is the
    dangerous error, and 'we could not tell' does not protect anyone."""
    pool = [o for o in outcomes if not o.gt_compliant and (o.matched or not conditioned)]
    if not pool:
        return None
    hit = sum(1 for o in pool if o.pred_compliant is False)
    return hit / len(pool)


def _f1_compliant(outcomes: list[PersonOutcome]) -> float | None:
    """F1 on the COMPLIANT class. Guards against an arm that maximises non-compliance recall by
    calling everyone non-compliant - that policy scores 1.0 recall and is useless."""
    tp = sum(1 for o in outcomes if o.gt_compliant and o.pred_compliant is True)
    fp = sum(1 for o in outcomes if not o.gt_compliant and o.pred_compliant is True)
    fn = sum(1 for o in outcomes if o.gt_compliant and o.pred_compliant is not True)
    if tp == 0:
        return 0.0 if (fp or fn) else None
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def summarise(outcomes: list[PersonOutcome]) -> dict:
    n = len(outcomes)
    return {
        "n_persons": n,
        "n_noncompliant": sum(1 for o in outcomes if not o.gt_compliant),
        "n_matched": sum(1 for o in outcomes if o.matched),
        "noncompliance_recall_strict": _recall(outcomes, conditioned=False),
        "noncompliance_recall_conditioned": _recall(outcomes, conditioned=True),
        "compliant_f1": _f1_compliant(outcomes),
        "undetermined_rate": (sum(1 for o in outcomes if o.pred_compliant is None) / n) if n else None,
        "verdict_accuracy": (sum(1 for o in outcomes if o.pred_compliant is o.gt_compliant) / n) if n else None,
    }


def stratified(outcomes: list[PersonOutcome], strata, labels, key="n_visible") -> dict:
    out = {}
    for o in outcomes:
        k = stratum_of(o.n_visible, strata, labels) if key == "n_visible" else f"{o.overlap:.2f}"
        out.setdefault(k, []).append(o)
    return {k: summarise(v) for k, v in sorted(out.items())}


def paired_bootstrap(a: list[PersonOutcome], b: list[PersonOutcome],
                     n_boot: int = 2000, seed: int = 42, conditioned: bool = False) -> dict:
    """CI on the ARM B - ARM A recall delta, resampling COMPOSITES (not persons).

    Resampling composites keeps persons from the same scene together, which is required: their
    outcomes are correlated through the scene they share, and resampling persons independently
    would understate the variance. Pairing means both arms see the same resampled scenes, so the
    detector's contribution cancels out of the delta.
    """
    by_comp: dict[str, tuple[list, list]] = {}
    for o in a:
        by_comp.setdefault(o.composite, ([], []))[0].append(o)
    for o in b:
        by_comp.setdefault(o.composite, ([], []))[1].append(o)
    keys = sorted(by_comp)
    rng = random.Random(seed)

    base_a, base_b = _recall(a, conditioned), _recall(b, conditioned)
    if base_a is None or base_b is None:
        return {"delta": None}

    deltas = []
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        sa = [o for k in pick for o in by_comp[k][0]]
        sb = [o for k in pick for o in by_comp[k][1]]
        ra, rb = _recall(sa, conditioned), _recall(sb, conditioned)
        if ra is not None and rb is not None:
            deltas.append(rb - ra)
    if not deltas:
        return {"delta": None}
    d = np.array(deltas)
    return {
        "arm_a": base_a, "arm_b": base_b, "delta": base_b - base_a,
        "ci95_lo": float(np.percentile(d, 2.5)), "ci95_hi": float(np.percentile(d, 97.5)),
        "p_delta_gt_0": float((d > 0).mean()), "n_boot": len(deltas),
    }
