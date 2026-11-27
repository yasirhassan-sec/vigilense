"""
Show what the CURRENT gun/knife model reacts to on the recorded webcam frames.

It finds the strongest knife detections (confidence 0.5 or higher, the app's default) and draws
the box on each frame, so you can see what the model thinks is a knife.

Run from the inner vigilense folder (app.py stopped):
    python show_knife_detections.py                  (all recordings)
    python show_knife_detections.py face_wall        (one or more named recordings)

The sheet is saved to datasets/VigilenseWebcam/knife_alarms_sheet.jpg (ignored by Git). Reads files only.
"""
import math
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

WEAPON = "weights/weapons_v1.pt"
BASE = Path("datasets/VigilenseWebcam")
MIN_CONF = 0.5
TOP = 24

weights = Path(WEAPON)
if not weights.exists():
    weights = Path(str(weights) + weights.suffix)      # Windows double extension
if not weights.exists():
    sys.exit(f"Model file not found: {WEAPON}")

model = YOLO(str(weights))
device = 0 if torch.cuda.is_available() else "cpu"

folders = [BASE / name for name in sys.argv[1:]] or sorted(d for d in BASE.iterdir() if d.is_dir())
found, per_folder = [], {}
for folder in folders:
    files = sorted(folder.glob("*.jpg"))
    count = 0
    for i in range(0, len(files), 16):
        chunk = files[i:i + 16]
        results = model.predict([str(f) for f in chunk], conf=MIN_CONF, device=device, verbose=False)
        for f, r in zip(chunk, results):
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                if model.names[int(c)] == "knife":
                    found.append((s, folder.name, f, b))
                    count += 1
    per_folder[folder.name] = (len(files), count)

print("Knife detections at confidence >= 0.5 (frames may count more than once):")
for name, (n, c) in per_folder.items():
    print(f"  {name:16s} {c:4d} detections in {n} frames")
if not found:
    sys.exit("No knife detections found, so there is nothing to draw.")

found.sort(key=lambda t: -t[0])
top = found[:TOP]
W, H, cols = 220, 180, 6
sheet = Image.new("RGB", (cols * W, math.ceil(len(top) / cols) * H), "white")
draw = ImageDraw.Draw(sheet)
print(f"\nStrongest {len(top)}:")
for i, (s, folder, f, box) in enumerate(top):
    im = Image.open(f).convert("RGB")
    ImageDraw.Draw(im).rectangle(box, outline="red", width=4)       # box drawn on a copy, files are untouched
    im.thumbnail((W - 6, H - 22))
    x, y = (i % cols) * W, (i // cols) * H
    sheet.paste(im, (x + 3, y + 18))
    draw.text((x + 3, y + 3), f"{i + 1}: {s:.2f} {f.stem[-14:]}", fill="black")
    print(f"  {i + 1:2d}. {s:.2f}  {folder}/{f.name}")

out = BASE / "knife_alarms_sheet.jpg"
sheet.save(out)
print("\nSaved:", out)