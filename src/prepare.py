"""One-time corpus prep: downscale SH17 and remap 17 classes -> the 4 this project uses.

Why this exists: SH17 ships originals at a median max-side of 5866px / 1.7MB. Decoding those
every epoch dominates training time and buys nothing at imgsz<=960. We pay the resize once.

Labels are YOLO-normalised, so they are resize-invariant - only the class id is remapped.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .sh17 import (SH17_TO_PID, load_official_splits, read_yolo_labels,
                   stable_split, write_yolo_labels)

Image.MAX_IMAGE_PIXELS = None  # SH17 has legitimate 8192px originals; the bomb guard is noise here

MAX_SIDE = 1280   # keeps person crops usable as composite sources; imgsz<=960 downstream
JPEG_Q = 92


def _one(args: tuple[str, str, str, str]) -> tuple[str, int, int, int] | None:
    stem, src_img, src_lbl, dst_root = args
    dst = Path(dst_root)
    try:
        im = Image.open(src_img)
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # fast DCT-domain downscale for JPEG
        im = im.convert("RGB")
        w0, h0 = im.size
        scale = min(1.0, MAX_SIDE / max(w0, h0))
        w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        if (w, h) != (w0, h0):
            im = im.resize((w, h), Image.LANCZOS)
        im.save(dst / "images" / f"{stem}.jpg", quality=JPEG_Q)
    except Exception as e:  # a handful of SH17 files are known-odd; skip rather than abort the run
        return None

    anns = read_yolo_labels(Path(src_lbl), w, h, remap=True)
    write_yolo_labels(dst / "labels" / f"{stem}.txt", anns, w, h)
    n_person = sum(1 for a in anns if a.pid == 0)
    return (stem, w, h, n_person)


def prepare(archive: Path, out: Path, det_val_frac: float = 0.10, workers: int = 8) -> dict:
    trn, heldout = load_official_splits(archive)
    det_val = [s for s in trn if stable_split(s, det_val_frac)]
    det_train = [s for s in trn if s not in set(det_val)]

    splits = {"train": det_train, "val": det_val, "heldout": heldout}
    manifest: dict[str, dict] = {}

    img_dir = archive / "images"
    by_stem = {p.stem: p for p in img_dir.iterdir() if not p.name.startswith(".")}

    for split, stems in splits.items():
        d = out / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        jobs = [(s, str(by_stem[s]), str(archive / "labels" / f"{s}.txt"), str(d))
                for s in stems if s in by_stem]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, j) for j in jobs]
            for f in as_completed(futs):
                r = f.result()
                if r is None:
                    continue
                stem, w, h, npers = r
                manifest[stem] = {"split": split, "w": w, "h": h, "n_person": npers}
                done += 1
                if done % 500 == 0:
                    print(f"  [{split}] {done}/{len(jobs)}", flush=True)
        print(f"[{split}] wrote {done}/{len(jobs)} images", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=0))

    yaml = (f"path: {out.resolve()}\n"
            "train: train/images\nval: val/images\ntest: heldout/images\n"
            "names:\n  0: person\n  1: head\n  2: helmet\n  3: safety-vest\n")
    (out / "data.yaml").write_text(yaml)
    return manifest
