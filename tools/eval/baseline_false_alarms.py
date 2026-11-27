"""
Baseline false-alarm test.

Runs the CURRENT models (fire/smoke and gun/knife, the same files app.py uses) over every recorded
webcam frame in datasets/VigilenseWebcam/<session>/ and counts how many frames trigger each class.
None of these frames is meant to contain fire, smoke or a weapon, so every trigger is a false alarm.

Run from the inner vigilense folder (app.py stopped):
    python baseline_false_alarms.py

It only reads files. Per-detection results are saved to datasets/VigilenseWebcam/baseline_detections.csv
"""
import csv
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

THREAT = "weights/fire_smoke_v1.pt"
WEAPON = "weights/weapons_v2.pt"
BASE = Path("datasets/VigilenseWebcam")
LEVELS = (0.35, 0.5, 0.7)          # confidence levels to compare
FLOOR = min(LEVELS)                # the models are run at the lowest level, then counted at each level
CLASSES = ["fire", "smoke", "gun", "knife"]


def find(path):
    p = Path(path)
    for candidate in (p, Path(str(p) + p.suffix)):      # also accepts the Windows double extension (.pt.pt)
        if candidate.exists():
            return candidate
    sys.exit(f"Model file not found: {p}")


DEVICE = 0 if torch.cuda.is_available() else "cpu"
models = {"fire/smoke": YOLO(str(find(THREAT))), "gun/knife": YOLO(str(find(WEAPON)))}
print("Inference device:", DEVICE)

folders = sorted(d for d in BASE.iterdir() if d.is_dir()) if BASE.exists() else []
if not folders:
    sys.exit(f"No recordings found in {BASE}")

table, rows = {}, []
for folder in folders:
    files = sorted(folder.glob("*.jpg"))
    hit = {c: [0] * len(LEVELS) for c in CLASSES}       # frames with at least one detection at each level
    for i in range(0, len(files), 16):
        chunk = files[i:i + 16]
        for model in models.values():
            results = model.predict([str(f) for f in chunk], conf=FLOOR, device=DEVICE, verbose=False)
            for f, r in zip(chunk, results):
                best = {}
                for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    label = model.names[int(c)]
                    best[label] = max(s, best.get(label, 0.0))
                    rows.append([folder.name, f.name, label, round(s, 3)])
                for label, s in best.items():
                    if label in hit:
                        for k, level in enumerate(LEVELS):
                            if s >= level:
                                hit[label][k] += 1
    table[folder.name] = (len(files), hit)
    print("done:", folder.name)

print("\nFrames that trigger each class, counted at confidence >= 0.35 / 0.5 / 0.7")
print(f"{'folder':18s} {'frames':>6s} " + " ".join(f"{c:>12s}" for c in CLASSES))
for name, (n, hit) in table.items():
    print(f"{name:18s} {n:6d} " + " ".join(f"{'/'.join(str(x) for x in hit[c]):>12s}" for c in CLASSES))

out = BASE / "baseline_detections.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["folder", "frame", "class", "confidence"])
    w.writerows(rows)
print("\nSaved:", out)