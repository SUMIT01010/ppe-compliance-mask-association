"""Train the 4-class YOLOv8 detector that BOTH arms share.

Scale is a single env flag (docs/CLOUD_RUN.md flips it):

    SCALE=demo  (default)  yolov8s @ 768, 25 epochs   - fits an afternoon on this Mac's MPS
    SCALE=full             yolov8m @ 960, 80 epochs   - what to run on a CUDA box

Detector quality is NOT the finding. The finding is the ARM B - ARM A gap, and both arms consume
the same detections, so a weaker detector costs absolute recall but leaves the paired delta
intact. mAP is reported so nobody confuses the two.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SCALE = os.environ.get("SCALE", "demo")
PRESET = {
    # BATCH SIZE HERE IS A MEMORY DECISION, NOT A THROUGHPUT ONE.
    # First attempt (yolov8s @ 768, batch 12) allocated 8.33 GB on a 16 GB unified-memory Mac
    # that already had ~6 GB of apps resident. It ran at 6.5 s/it - and that was swap thrash,
    # not compute: free RAM was down to ~65 MB and the load average hit 45. On unified memory
    # the MPS allocation and the app working set are the SAME pool, so an oversized batch does
    # not just slow training, it makes the whole machine unusable.
    # Keep the allocation near ~3 GB. On a discrete-GPU box raise batch freely (see SCALE=full).
    "demo": dict(model="yolov8n.pt", imgsz=640, epochs=20, batch=8),
    "full": dict(model="yolov8m.pt", imgsz=960, epochs=80, batch=16),
}[SCALE]

if __name__ == "__main__":
    from ultralytics import YOLO

    device = config.get_device()
    data = config.DATASETS / "sh17_p7" / "data.yaml"
    print(f"SCALE={SCALE} device={device} {PRESET}")

    model = YOLO(PRESET["model"])
    model.train(
        data=str(data),
        imgsz=PRESET["imgsz"],
        epochs=PRESET["epochs"],
        batch=PRESET["batch"],
        device=device,
        workers=4,
        seed=config.SEED,
        project=str(config.RUNS),
        name=f"detector_{SCALE}",
        exist_ok=True,
        patience=12,
        # Small objects (helmet, head) are the hard part of this corpus: 39.7k of 76k SH17
        # annotations are under 1% of image area. Keep scale/mosaic augmentation mild so tiny
        # boxes are not augmented out of existence.
        scale=0.4,
        mosaic=0.6,
        close_mosaic=8,
        pretrained=True,
        val=True,
        plots=True,
        # amp=False skips ultralytics' AMP sanity check, which DOWNLOADS yolo11n.pt on first run.
        # On a flaky connection that download blocks the whole train() call before a single
        # iteration, presenting as a hang at 0% CPU with an empty log. MPS gains little from AMP
        # anyway - float32 is the recommended dtype there.
        amp=False,
    )
    print("best:", config.RUNS / f"detector_{SCALE}" / "weights" / "best.pt")
