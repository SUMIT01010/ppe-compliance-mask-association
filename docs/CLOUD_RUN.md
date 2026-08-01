# Running this at full scale

Everything below is a config change, not a code change. The local defaults are sized for an
afternoon on an M-series Mac; nothing about the experiment design differs between scales.

## 1. Detector

```bash
SCALE=full DEVICE=cuda python scripts/02_train_detector.py
```

| | `SCALE=demo` (default) | `SCALE=full` |
|---|---|---|
| model | yolov8s | yolov8m |
| imgsz | 768 | 960 |
| epochs | 25 | 80 |
| batch | 12 | 16 |

`imgsz` is the one that matters here, not model size: 39,764 of SH17's 75,994 annotations are
under 1% of image area, and `helmet`/`head` are the classes the compliance rule depends on.
Raising input resolution buys more on this corpus than a bigger backbone does.

`DEVICE` is read by `config.get_device()` and is never hardcoded at a call site, so the same
scripts run `mps` locally and `cuda` on a node.

## 2. Composites

```bash
python scripts/03_build_composites.py 5000
```

The evaluation set is cheap (CPU/PIL only) — 700 composites take about four minutes. The binding
constraint is not compute, it is **source diversity**: only 35 held-out single-person images
contain a helmet. Past roughly 1,000 composites you are resampling the same 35 compliant people
into new arrangements, which grows the number of *scenes* but not the number of distinct
*persons*. Report both counts; do not let a large composite count imply a large sample.

Raising it properly means annotating more compliant single-person images, not generating more
composites.

## 3. Experiment

```bash
DEVICE=cuda python scripts/04_run_experiment.py runs/detector_full/weights/best.pt
```

Cost is dominated by MobileSAM, which runs once per composite for ARM B. ARM A0 and ARM A are
effectively free — they reuse the same detections.

## 4. Service

```bash
docker build -f app/Dockerfile -t ppe-compliance .
docker run -p 8000:8000 -v $(pwd)/runs/detector_demo/weights:/app/weights ppe-compliance
```

The image is CPU-only on purpose (see the comment at the top of `app/Dockerfile`). For a GPU
node, swap the torch index-url to a cu121 wheel and set `DEVICE=cuda`.
