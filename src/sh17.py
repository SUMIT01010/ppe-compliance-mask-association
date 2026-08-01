"""SH17 dataset facts, label IO, and the split protocol.

Everything in here is *measured*, not assumed. The class index -> name map was recovered by
aligning YOLO line order against VOC object order over 400 random images (purity 1.000 on all
17 classes); see docs/TECHNICAL_JOURNEY.md. Do not hand-edit it from the paper.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# --- SH17 native 17-class map (verified empirically) ----------------------
SH17_CLASSES = {
    0: "person", 1: "ear", 2: "ear-mufs", 3: "face", 4: "face-guard",
    5: "face-mask-medical", 6: "foot", 7: "tools", 8: "glasses", 9: "gloves",
    10: "helmet", 11: "hands", 12: "head", 13: "medical-suit", 14: "shoes",
    15: "safety-suit", 16: "safety-vest",
}

# --- The 4 classes this project trains on ---------------------------------
# Everything else in SH17 is irrelevant to a head-protection compliance call and only slows
# training. `safety-vest` is kept in the DETECTOR (it is nearly free) but is deliberately NOT
# part of the compliance rule - see PROJECT_CLASSES_NOTE below.
PROJECT_CLASSES = ["person", "head", "helmet", "safety-vest"]
NAME_TO_PID = {n: i for i, n in enumerate(PROJECT_CLASSES)}
SH17_TO_PID = {sid: NAME_TO_PID[name] for sid, name in SH17_CLASSES.items() if name in NAME_TO_PID}

PID_PERSON, PID_HEAD, PID_HELMET, PID_VEST = 0, 1, 2, 3

PROJECT_CLASSES_NOTE = """
Compliance is HEAD-PROTECTION ONLY (helmet). Vest was dropped from the compliance rule after
counting the held-out pool: only 14 single-person held-out images contain a safety-vest, versus
35 compliant / 712 non-compliant helmet sources. A vest-inclusive rule would compute its headline
on a sample too small to support it. Vest stays a detector class so the number is reportable, but
it is not in the verdict.
"""


@dataclass(frozen=True)
class Ann:
    """One annotation, in absolute xyxy pixel coords of the image it came from."""
    pid: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


def read_yolo_labels(path: Path, w: int, h: int, remap: bool = True) -> list[Ann]:
    """Read a YOLO .txt into absolute-pixel Anns, keeping only the 4 project classes."""
    out: list[Ann] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        sid = int(parts[0])
        if remap:
            if sid not in SH17_TO_PID:
                continue
            pid = SH17_TO_PID[sid]
        else:
            pid = sid
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append(Ann(pid,
                       (cx - bw / 2) * w, (cy - bh / 2) * h,
                       (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def write_yolo_labels(path: Path, anns: list[Ann], w: int, h: int) -> None:
    lines = []
    for a in anns:
        cx, cy = (a.x0 + a.x1) / 2 / w, (a.y0 + a.y1) / 2 / h
        bw, bh = (a.x1 - a.x0) / w, (a.y1 - a.y0) / h
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(bw, 1.0), min(bh, 1.0)
        lines.append(f"{a.pid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# --- Split protocol -------------------------------------------------------
# SH17 ships train_files.txt (6479) / val_files.txt (1620), disjoint. We keep the authors'
# boundary and never cross it:
#
#   SH17 train  -> det_train (90%) + det_val (10%)   ... detector fitting + checkpoint selection
#   SH17 val    -> HELDOUT (100%)                    ... composite sources, real-image eval, test mAP
#
# The composite experiment therefore runs on persons the detector has never seen. Splitting the
# heldout pool further would leave too few compliant sources (35) to build composites from.

def stable_split(stem: str, frac: float, salt: str = "p7") -> bool:
    """Deterministic per-stem split that does not depend on file ordering or list length."""
    d = hashlib.md5(f"{salt}:{stem}".encode()).hexdigest()
    return (int(d[:8], 16) / 0xFFFFFFFF) < frac


def load_official_splits(archive: Path) -> tuple[list[str], list[str]]:
    trn = [Path(x).stem for x in (archive / "train_files.txt").read_text().split()]
    val = [Path(x).stem for x in (archive / "val_files.txt").read_text().split()]
    return sorted(trn), sorted(val)
