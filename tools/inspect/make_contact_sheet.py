"""
Make one picture that shows many frames from a webcam recording, so they can be checked at a glance.

Examples (run from the inner vigilense folder):
    python make_contact_sheet.py face_wall
    python make_contact_sheet.py empty_room 36

The sheet is saved inside datasets/VigilenseWebcam/ (ignored by Git, so face pictures are never committed).
"""
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

if len(sys.argv) < 2:
    sys.exit("Usage: python make_contact_sheet.py <session_name> [how_many_frames]")

base = Path("datasets/VigilenseWebcam")
folder = base / sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
files = sorted(folder.glob("*.jpg"))
if not files:
    sys.exit(f"No .jpg files found in {folder}")

random.seed(0)                                            # same frames every time
pick = sorted(random.sample(files, min(n, len(files))))   # sorted, so time order is kept

W, H, cols = 220, 165, 6
rows = math.ceil(len(pick) / cols)
sheet = Image.new("RGB", (cols * W, rows * H), "white")
draw = ImageDraw.Draw(sheet)
for i, f in enumerate(pick):
    im = Image.open(f).convert("RGB")
    im.thumbnail((W - 6, H - 22))
    x, y = (i % cols) * W, (i // cols) * H
    sheet.paste(im, (x + 3, y + 18))
    draw.text((x + 3, y + 3), f.stem[-12:], fill="black")

out = base / f"{sys.argv[1]}_sheet.jpg"
sheet.save(out)
print(f"{len(files)} frames in folder, showing {len(pick)}. Saved: {out}")