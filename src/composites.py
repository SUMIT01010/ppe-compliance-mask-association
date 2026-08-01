"""Synthetic crowded scenes with EXACT person<->PPE ground truth.

The problem this solves (docs/EVAL_DESIGN.md): SH17 annotates person / head / helmet as
independent boxes and never says which head belongs to which person. So the corpus cannot score
an association method - there is no association label in it. Scoring against a box-IoU-derived
"ground truth" would define ARM A to be correct by construction.

The fix: start from images containing exactly ONE person. There, every PPE box provably belongs
to that person. Composite k such crops into one scene and the links carry through exactly, at any
crowding level, for free.

DESIGN CHOICE - rectangular paste, not mask paste.
Pasting SAM-segmented silhouettes would look better and would be a BUG: a silhouette pasted onto
a different background leaves a sharp figure/ground boundary that SAM can re-segment almost
trivially, inflating ARM B (the arm under test) for a reason that has nothing to do with the
mechanism. A rectangular crop carries its own background with it, so ARM B's segmentation step
has to do real work. The paste is less photorealistic and more conservative toward our own
hypothesis, which is the correct direction to err.

Both arms consume the SAME detections from the SAME detector on the SAME composites, so any
compositing artifact hits both arms identically and cannot manufacture a margin between them.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .sh17 import PID_HEAD, PID_HELMET, PID_PERSON, Ann, read_yolo_labels

CANVAS_W, CANVAS_H = 1280, 960   # canvas HEIGHT is fixed; width grows with the crowd
MAX_CANVAS_W = 2560              # hard cap so inference cost stays bounded
MIN_VISIBILITY = 0.30   # persons occluded below this are dropped from GT: not fairly detectable
PERSON_H_FRAC = 0.72    # target person height as a fraction of canvas height


@dataclass
class GTPerson:
    """One pasted person, with its links known by construction."""
    gid: int
    box: tuple[float, float, float, float]        # amodal person box on the canvas
    visible_box: tuple[float, float, float, float]
    visibility: float
    head_boxes: list[tuple[float, float, float, float]]
    helmet_boxes: list[tuple[float, float, float, float]]
    compliant: bool          # helmet present in the source image => wearing head protection
    source_stem: str


@dataclass
class Composite:
    name: str
    k: int                   # persons pasted (before visibility filtering)
    overlap: float           # TARGET step overlap used for layout
    persons: list[GTPerson]
    realized_overlap: float = 0.0
    """Mean over persons of their highest box-IoU with any other person.

    The layout steps by person_box_width*(1-overlap), but source crops differ in aspect (SH17's
    single-person pool is portrait-heavy), so the target and the achieved overlap diverge. This
    is the number to plot the overlap sweep against; `overlap` is only the knob that produced it.
    """

    @property
    def n_visible(self) -> int:
        return len(self.persons)


def _source_pool(split_dir: Path, manifest: dict) -> tuple[list[str], list[str]]:
    """Single-person heldout images that have a head box, partitioned by compliance."""
    compliant, noncompliant = [], []
    for stem, m in manifest.items():
        if m["split"] != "heldout" or m["n_person"] != 1:
            continue
        anns = read_yolo_labels(split_dir / "labels" / f"{stem}.txt", m["w"], m["h"], remap=False)
        cls = {a.pid for a in anns}
        if PID_HEAD not in cls:
            continue          # no head => compliance undeterminable => not a usable source
        (compliant if PID_HELMET in cls else noncompliant).append(stem)
    return sorted(compliant), sorted(noncompliant)


def _backgrounds(manifest: dict, split_dir: Path) -> list[str]:
    """Zero-person heldout images. A background containing a person would inject a phantom
    detection with no GT link and silently corrupt both arms."""
    return sorted(s for s, m in manifest.items()
                  if m["split"] == "heldout" and m["n_person"] == 0)


def _crop_source(img_dir: Path, lbl_dir: Path, stem: str, m: dict):
    """Crop the person + its PPE out of a single-person image. Returns (PIL crop, anns in crop)."""
    anns = read_yolo_labels(lbl_dir / f"{stem}.txt", m["w"], m["h"], remap=False)
    persons = [a for a in anns if a.pid == PID_PERSON]
    if not persons:
        return None
    p = max(persons, key=lambda a: a.area)
    ppe = [a for a in anns if a.pid in (PID_HEAD, PID_HELMET)]

    # Crop must cover the person AND its PPE - helmets routinely poke above the person box.
    x0 = min([p.x0] + [a.x0 for a in ppe])
    y0 = min([p.y0] + [a.y0 for a in ppe])
    x1 = max([p.x1] + [a.x1 for a in ppe])
    y1 = max([p.y1] + [a.y1 for a in ppe])
    mx, my = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
    x0, y0 = max(0.0, x0 - mx), max(0.0, y0 - my)
    x1, y1 = min(float(m["w"]), x1 + mx), min(float(m["h"]), y1 + my)
    if x1 - x0 < 24 or y1 - y0 < 48:
        return None

    im = Image.open(img_dir / f"{stem}.jpg").convert("RGB").crop((int(x0), int(y0), int(x1), int(y1)))
    out = [Ann(a.pid, a.x0 - x0, a.y0 - y0, a.x1 - x0, a.y1 - y0) for a in [p] + ppe]
    return im, out


def build_one(rng: random.Random, k: int, overlap: float, srcs: list[str],
              img_dir: Path, lbl_dir: Path, manifest: dict,
              bg_stem: str | None, name: str) -> tuple[Image.Image, Composite] | None:
    """Composite k source persons into one scene at a target adjacent-box overlap."""
    crops = []
    for s in srcs:
        c = _crop_source(img_dir, lbl_dir, s, manifest[s])
        if c is not None:
            crops.append((s, *c))
    if not crops:
        return None

    # Scale each crop so its PERSON box reaches the target height, then lay out left-to-right
    # stepping by person_width*(1-overlap) so adjacent person boxes overlap by `overlap`.
    plan = []
    for stem, im, anns in crops:
        p = next(a for a in anns if a.pid == PID_PERSON)
        ph = max(1.0, p.y1 - p.y0)
        sc = (CANVAS_H * PERSON_H_FRAC * rng.uniform(0.92, 1.04)) / ph
        plan.append((stem, im, anns, sc, (p.x1 - p.x0) * sc))

    # CANVAS WIDTH GROWS WITH THE CROWD - this is load-bearing, not cosmetics.
    # A fixed-width canvas forces persons to be shrunk as k rises, which would make "person got
    # smaller" vary in lockstep with "scene got more crowded". The crowding strata would then be
    # confounded with object scale and the central claim - that the ARM B margin grows with
    # crowding - would be untestable. Holding person height constant and widening the canvas
    # keeps crowding the only thing that changes across strata.
    span = sum(w * (1 - overlap) for *_, w in plan[:-1]) + (plan[-1][-1] if plan else 0)
    cw_canvas = int(min(MAX_CANVAS_W, max(CANVAS_W, span / 0.92)))
    if span > cw_canvas * 0.96:            # only beyond the hard cap do we shrink
        squeeze = cw_canvas * 0.96 / span
        plan = [(s, im, a, sc * squeeze, w * squeeze) for s, im, a, sc, w in plan]
        span = sum(w * (1 - overlap) for *_, w in plan[:-1]) + plan[-1][-1]

    # Background: a real zero-person SH17 scene, blurred hard. SH17's zero-person images are
    # mostly object close-ups (a jar, a hand, a tool); pasted sharp they read as absurd and add
    # structure the detector can trip on. Blurring leaves plausible out-of-focus context and no
    # competing objects.
    if bg_stem:
        canvas = (Image.open(img_dir / f"{bg_stem}.jpg").convert("RGB")
                  .resize((cw_canvas, CANVAS_H)).filter(ImageFilter.GaussianBlur(18)))
    else:
        canvas = Image.new("RGB", (cw_canvas, CANVAS_H), (110, 110, 112))

    x = rng.uniform(0, max(1.0, cw_canvas - span))
    owner = np.full((CANVAS_H, cw_canvas), -1, dtype=np.int16)
    order = list(range(len(plan)))
    rng.shuffle(order)                                # randomise depth so occlusion isn't ordered
    placed, gts = [], []

    for gid, (stem, im, anns, sc, pw) in enumerate(plan):
        p = next(a for a in anns if a.pid == PID_PERSON)
        cw, ch = max(1, round(im.width * sc)), max(1, round(im.height * sc))
        # Stand them on a common ground line, then clamp so the person box stays fully in frame -
        # a person clipped by the canvas edge has an unusable ground-truth box.
        py = CANVAS_H * 0.97 - p.y1 * sc + rng.uniform(-0.03, 0.01) * CANVAS_H
        py = min(max(py, -p.y0 * sc), CANVAS_H - p.y1 * sc)
        ox, oy = x - p.x0 * sc, py
        placed.append((gid, im.resize((cw, ch), Image.LANCZOS), ox, oy, sc, anns, stem))
        x += pw * (1 - overlap)

    for gid, im, ox, oy, sc, anns, stem in sorted(placed, key=lambda t: order[t[0]]):
        canvas.paste(im, (int(ox), int(oy)))
        p = next(a for a in anns if a.pid == PID_PERSON)
        bx0, by0 = ox + p.x0 * sc, oy + p.y0 * sc
        bx1, by1 = ox + p.x1 * sc, oy + p.y1 * sc
        xi0, yi0 = int(max(0, bx0)), int(max(0, by0))
        xi1, yi1 = int(min(cw_canvas, bx1)), int(min(CANVAS_H, by1))
        if xi1 > xi0 and yi1 > yi0:
            owner[yi0:yi1, xi0:xi1] = gid

    for gid, im, ox, oy, sc, anns, stem in placed:
        p = next(a for a in anns if a.pid == PID_PERSON)
        box = (ox + p.x0 * sc, oy + p.y0 * sc, ox + p.x1 * sc, oy + p.y1 * sc)
        xi0, yi0 = int(max(0, box[0])), int(max(0, box[1]))
        xi1, yi1 = int(min(cw_canvas, box[2])), int(min(CANVAS_H, box[3]))
        if xi1 <= xi0 or yi1 <= yi0:
            continue
        vis_mask = owner[yi0:yi1, xi0:xi1] == gid
        vis = float(vis_mask.mean())
        if vis < MIN_VISIBILITY:
            continue
        ys, xs = np.nonzero(vis_mask)
        vbox = (float(xi0 + xs.min()), float(yi0 + ys.min()),
                float(xi0 + xs.max()), float(yi0 + ys.max()))
        tf = lambda a: (ox + a.x0 * sc, oy + a.y0 * sc, ox + a.x1 * sc, oy + a.y1 * sc)
        heads = [tf(a) for a in anns if a.pid == PID_HEAD]
        helms = [tf(a) for a in anns if a.pid == PID_HELMET]
        gts.append(GTPerson(gid, box, vbox, vis, heads, helms, bool(helms), stem))

    if not gts:
        return None

    from .association import box_iou
    if len(gts) > 1:
        mx = [max(box_iou(a.box, b.box) for j, b in enumerate(gts) if j != i)
              for i, a in enumerate(gts)]
        realized = float(np.mean(mx))
    else:
        realized = 0.0
    return canvas, Composite(name, len(plan), overlap, gts, realized_overlap=realized)


def build_dataset(out: Path, img_dir: Path, lbl_dir: Path, manifest: dict,
                  n: int = 600, seed: int = 42, p_compliant: float = 0.35) -> dict:
    """Build the composite evaluation set, sweeping crowding x overlap."""
    rng = random.Random(seed)
    comp, noncomp = _source_pool(img_dir.parent, manifest)
    bgs = _backgrounds(manifest, img_dir.parent)
    print(f"sources: {len(comp)} compliant / {len(noncomp)} non-compliant; {len(bgs)} backgrounds")
    if not comp or not noncomp:
        raise RuntimeError("source pool empty - run 01_prepare_data.py first")

    (out / "images").mkdir(parents=True, exist_ok=True)
    K_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    OVERLAPS = [0.0, 0.15, 0.30, 0.45, 0.60]

    records = []
    for i in range(n):
        k = rng.choice(K_CHOICES)
        ov = rng.choice(OVERLAPS) if k > 1 else 0.0
        picks = []
        for _ in range(k):
            pool = comp if (rng.random() < p_compliant and comp) else noncomp
            # no repeat within a single composite - a person cannot stand next to themselves
            for _try in range(20):
                s = rng.choice(pool)
                if s not in picks:
                    picks.append(s)
                    break
        bg = rng.choice(bgs) if bgs else None
        name = f"comp_{i:05d}"
        r = build_one(rng, k, ov, picks, img_dir, lbl_dir, manifest, bg, name)
        if r is None:
            continue
        canvas, meta = r
        canvas.save(out / "images" / f"{name}.jpg", quality=92)
        records.append(asdict(meta))
        if (i + 1) % 100 == 0:
            print(f"  built {i+1}/{n}", flush=True)

    (out / "composites.json").write_text(json.dumps(records))
    npers = [len(r["persons"]) for r in records]
    print(f"built {len(records)} composites; persons/scene mean={np.mean(npers):.2f} "
          f"max={max(npers)}; total persons={sum(npers)}")
    return {"n": len(records), "persons": int(sum(npers))}
