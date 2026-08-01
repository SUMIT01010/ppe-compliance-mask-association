"""PPE compliance service.

Serves the pipeline the experiment validated: detect -> segment -> associate -> per-person
verdict. The association arm is selectable at request time so the deployed service can be A/B'd
against the same baseline the paper reports, rather than silently shipping one and hoping.

Drift monitoring here is deliberately small and behavioural rather than a metrics stack: the
signals that actually predict this model degrading in a new site are (a) crowding shifting
outside the range it was evaluated on, (b) the undetermined rate climbing, which means heads are
not being associated, and (c) detector confidence sagging. All three are computable without
labels, which is the point - a deployed safety monitor never gets labels.
"""

from __future__ import annotations

import io
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.association import (associate_by_box_containment,  # noqa: E402
                             associate_by_mask_containment, verdict_from_assignment)
from src.perception import Perception  # noqa: E402

WEIGHTS = Path(os.environ.get(
    "WEIGHTS", config.RUNS / "detector_demo" / "weights" / "best.pt"))

app = FastAPI(title="PPE Compliance (mask-containment association)", version="1.0")
_percep: Perception | None = None
_window: deque = deque(maxlen=500)   # rolling behavioural stats for drift


def percep() -> Perception:
    global _percep
    if _percep is None:
        _percep = Perception(WEIGHTS, config.get_device())
    return _percep


@app.get("/health")
def health():
    return {"status": "ok", "weights": str(WEIGHTS), "weights_present": WEIGHTS.exists(),
            "device": config.get_device()}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  arm: str = Query("mask", pattern="^(mask|box)$")):
    """Per-person head-protection compliance for one image.

    arm=mask -> ARM B (instance-mask containment), the method this project validated
    arm=box  -> ARM A (bounding-box containment), the baseline, for online comparison
    """
    raw = await file.read()
    img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

    t0 = time.perf_counter()
    p = percep()
    dets = p.detect(img)
    if arm == "mask":
        p.attach_person_masks(img, dets)
        verdicts = associate_by_mask_containment(dets, config.CONTAINMENT_THRESHOLD)
    else:
        verdicts = associate_by_box_containment(dets, config.CONTAINMENT_THRESHOLD)

    people = []
    for v in verdicts:
        v = verdict_from_assignment(dets, v)
        people.append({
            "box": [round(c, 1) for c in dets[v.person_idx].box],
            "confidence": round(dets[v.person_idx].conf, 3),
            "compliant": v.compliant,
            "status": ("non_compliant" if v.compliant is False
                       else "compliant" if v.compliant else "undetermined"),
            "assigned_ppe": sorted({dets[j].cls for j in v.assigned}),
        })

    latency_ms = (time.perf_counter() - t0) * 1000
    n_nc = sum(1 for x in people if x["compliant"] is False)
    n_und = sum(1 for x in people if x["compliant"] is None)
    _window.append({
        "n_persons": len(people), "undetermined": n_und,
        "mean_conf": float(np.mean([x["confidence"] for x in people])) if people else None,
        "latency_ms": latency_ms,
    })

    return JSONResponse({
        "arm": arm,
        "n_persons": len(people),
        "n_non_compliant": n_nc,
        "n_undetermined": n_und,
        "latency_ms": round(latency_ms, 1),
        "people": people,
    })


@app.get("/monitor")
def monitor():
    """Label-free drift signals over the last N requests. Compare against the evaluation-time
    reference band; a sustained excursion is the cue to re-validate, not an automatic alarm."""
    if not _window:
        return {"n": 0, "note": "no traffic yet"}
    crowd = [w["n_persons"] for w in _window]
    confs = [w["mean_conf"] for w in _window if w["mean_conf"] is not None]
    und = sum(w["undetermined"] for w in _window)
    tot = sum(w["n_persons"] for w in _window)
    return {
        "n_requests": len(_window),
        "persons_per_frame_mean": round(float(np.mean(crowd)), 2),
        "persons_per_frame_p95": float(np.percentile(crowd, 95)),
        "undetermined_rate": round(und / tot, 4) if tot else None,
        "mean_detector_confidence": round(float(np.mean(confs)), 3) if confs else None,
        "latency_ms_p95": round(float(np.percentile([w["latency_ms"] for w in _window], 95)), 1),
        "reference": {
            "note": "evaluation-time values from outputs/experiment.json; drift = sustained "
                    "departure from these, especially undetermined_rate rising",
        },
    }
