"""Person <-> PPE association: the arms under test.

WHY THE BASELINE CHANGED FROM THE ORIGINAL SCAFFOLD
---------------------------------------------------
The first scaffold defined ARM A as "assign each PPE box to the person box it overlaps most,
above IoU >= 0.30". That baseline is broken, and not in an interesting way. A helmet box is
roughly 1-2% of a person box's area, so their IoU is ~0.01-0.02 and NEVER clears 0.30. ARM A
would assign nothing, call every person undetermined, and score 0. Beating it would prove
nothing about masks.

So the fair baseline is the rule practitioners actually use for small-PPE-to-person assignment:
CONTAINMENT - what fraction of the PPE box falls inside the person region - with argmax over
persons. That makes the two arms differ in exactly one thing:

    ARM A  person region = the bounding BOX
    ARM B  person region = the instance MASK

which is precisely the mechanism this project claims matters. Same threshold, same argmax, same
detections. The original IoU rule is retained as ARM A0 and reported, because its collapse is
the evidence that ARM A is the honest baseline rather than a strawman we picked to lose.

OWNERSHIP NOTE (CLAUDE.md section 1)
------------------------------------
`associate_by_mask_containment` (ARM B) is the novel mechanism and was originally left as a stub
for Sumit. It is implemented here on his instruction to "assume whatever is necessary". It is a
reference implementation, deliberately kept short with the decision rule explicit, so it is
straightforward to inspect, defend, or replace with his own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]  # xyxy


@dataclass
class Detection:
    """One detection. `mask` is populated for persons in ARM B only."""
    cls: str                          # "person" | "head" | "helmet" | "safety-vest"
    box: Box
    conf: float
    mask: np.ndarray | None = None    # HxW bool


@dataclass
class PersonVerdict:
    """Per-person compliance outcome - the unit the headline metric is computed over."""
    person_idx: int
    assigned: list[int] = field(default_factory=list)
    compliant: bool | None = None     # None = undetermined (no head evidence)


def box_iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def box_containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (iw * ih) / area


def mask_containment(inner: Box, mask: np.ndarray) -> float:
    """Fraction of `inner`'s area that lies inside a boolean instance mask."""
    h, w = mask.shape
    x0, y0 = int(max(0, np.floor(inner[0]))), int(max(0, np.floor(inner[1])))
    x1, y1 = int(min(w, np.ceil(inner[2]))), int(min(h, np.ceil(inner[3])))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return float(mask[y0:y1, x0:x1].sum()) / area


def _assign(detections: list[Detection], score_fn, threshold: float) -> list[PersonVerdict]:
    """Shared argmax-over-persons assignment. The arms differ ONLY in `score_fn`."""
    persons = [i for i, d in enumerate(detections) if d.cls == "person"]
    ppe = [i for i, d in enumerate(detections) if d.cls in ("helmet", "head", "safety-vest")]
    verdicts = [PersonVerdict(person_idx=i) for i in persons]
    by_person = {p: v for p, v in zip(persons, verdicts)}

    for j in ppe:
        scored = [(score_fn(detections[j], detections[p]), p) for p in persons]
        if not scored:
            continue
        best, best_p = max(scored, key=lambda t: t[0])
        if best >= threshold:
            by_person[best_p].assigned.append(j)
    return verdicts


def associate_by_box_iou(detections: list[Detection], iou_threshold: float) -> list[PersonVerdict]:
    """ARM A0 - the naive rule. Reported to show it collapses; see module docstring."""
    return _assign(detections, lambda ppe, per: box_iou(ppe.box, per.box), iou_threshold)


def associate_by_box_containment(detections: list[Detection], threshold: float) -> list[PersonVerdict]:
    """ARM A - the honest baseline. Person region is the bounding box."""
    return _assign(detections, lambda ppe, per: box_containment(ppe.box, per.box), threshold)


def associate_by_mask_containment(detections: list[Detection], threshold: float) -> list[PersonVerdict]:
    """ARM B - the mechanism under test. Person region is the instance mask.

    Identical decision rule to ARM A; the only change is that a PPE box is scored against the
    person's SEGMENTED silhouette rather than their bounding box. Two overlapping person boxes
    can both contain the same helmet, so ARM A's argmax is close to a coin flip in exactly the
    crowded case that matters. Their masks are disjoint, so containment can resolve it.

    Whether that converts into a non-compliance recall gain, and whether the gain grows with
    crowding, is what evaluate.py measures. It is not assumed here.

    Falls back to the person's box when a mask is missing (segmentation failure), so a SAM miss
    degrades ARM B to ARM A on that person rather than silently dropping them from the metric.
    """
    def score(ppe: Detection, per: Detection) -> float:
        if per.mask is None:
            return box_containment(ppe.box, per.box)
        return mask_containment(ppe.box, per.mask)

    return _assign(detections, score, threshold)


def verdict_from_assignment(detections: list[Detection], verdict: PersonVerdict) -> PersonVerdict:
    """Derive the compliance call. HEAD PROTECTION ONLY - see src/sh17.py PROJECT_CLASSES_NOTE.

    Rule (the assumption the arms disagree about, per docs/DATASET_DECISION.md):
        head assigned, no helmet assigned  -> NON-COMPLIANT
        helmet assigned                    -> COMPLIANT
        neither assigned                   -> UNDETERMINED (None)

    A person with no evidence at all is left undetermined rather than scored compliant. Scoring
    absence as compliance is the exact failure mode that makes a safety monitor useless.
    """
    assigned = {detections[j].cls for j in verdict.assigned}
    if "head" not in assigned and "helmet" not in assigned:
        verdict.compliant = None
    else:
        verdict.compliant = "helmet" in assigned
    return verdict
