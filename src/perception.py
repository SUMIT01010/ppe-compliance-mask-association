"""Detector + segmenter. Produces the detections BOTH arms consume.

The single most important property of this module: `detect()` is called ONCE per image and its
output is shared by every arm. Masks are attached afterwards. If each arm ran its own detector,
any difference between arms would confound association quality with detection noise, and the
experiment would measure nothing.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .association import Detection
from .sh17 import PROJECT_CLASSES


class Perception:
    def __init__(self, weights: str | Path, device: str, sam_weights: str = "mobile_sam.pt",
                 conf: float = 0.25, imgsz: int = 960):
        from ultralytics import YOLO
        self.yolo = YOLO(str(weights))
        self.device, self.conf, self.imgsz = device, conf, imgsz
        self._sam = None
        self._sam_weights = sam_weights
        self.timings: dict[str, list[float]] = {"detect": [], "segment": []}

    @property
    def sam(self):
        if self._sam is None:
            from ultralytics import SAM
            self._sam = SAM(self._sam_weights)
        return self._sam

    def detect(self, image: np.ndarray) -> list[Detection]:
        t = time.perf_counter()
        r = self.yolo(image, conf=self.conf, imgsz=self.imgsz, device=self.device,
                      verbose=False)[0]
        self.timings["detect"].append(time.perf_counter() - t)
        out = []
        if r.boxes is None:
            return out
        for b in r.boxes:
            cid = int(b.cls.item())
            out.append(Detection(cls=PROJECT_CLASSES[cid],
                                 box=tuple(float(v) for v in b.xyxy[0].tolist()),
                                 conf=float(b.conf.item())))
        return out

    def attach_person_masks(self, image: np.ndarray, dets: list[Detection]) -> None:
        """Populate `.mask` on person detections, in place. ARM B only.

        Prompting SAM with the person's own box is the honest setup: it uses exactly the
        information ARM A already has (that box) and adds only segmentation.
        """
        persons = [d for d in dets if d.cls == "person"]
        if not persons:
            return
        boxes = [list(d.box) for d in persons]
        t = time.perf_counter()
        r = self.sam(image, bboxes=boxes, verbose=False, device=self.device)[0]
        self.timings["segment"].append(time.perf_counter() - t)
        if r.masks is None:
            return
        m = r.masks.data.cpu().numpy().astype(bool)
        for d, mask in zip(persons, m):
            d.mask = mask

    def latency_summary(self) -> dict:
        def p(xs, q):
            return float(np.percentile(xs, q) * 1000) if xs else None
        return {
            "detect_ms_p50": p(self.timings["detect"], 50),
            "detect_ms_p95": p(self.timings["detect"], 95),
            "segment_ms_p50": p(self.timings["segment"], 50),
            "segment_ms_p95": p(self.timings["segment"], 95),
            "n_detect_calls": len(self.timings["detect"]),
            "n_segment_calls": len(self.timings["segment"]),
        }
