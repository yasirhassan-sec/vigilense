"""
Contact sheet of one class from a dataset in datasets/VigilenseExtra/raw/<name>, with the boxes drawn in red.
Use it to check what a class really contains before deciding how to use it.

Examples (run from the inner vigilense folder):
    python inspect_dataset_class.py vape_a 0          (class id 0, 36 random samples)
    python inspect_dataset_class.py vape_a 4 48       (class id 4, 48 samples)

Reads files only. The sheet and a CSV list of the sampled images are saved in datasets/VigilenseExtra/sheets/.
"""
import math
import random
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

if len(sys.argv) < 3:
    sys.exit("Usage: python inspect_dataset_class.py <dataset_name> <class_id> [how_many]")
name, cid = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 36

root = Path("datasets/VigilenseExtra/raw") / name
if not root.exists():
    sys.exit(f"{root} not found")
yaml_path = root / "data.yaml"
names = yaml.safe_load(yaml_path.read_text()).get("names", []) if yaml_path.exists() else []
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
class_name = names[int(cid)] if int(cid) < len(names) else "?"

samples, areas, odd_lines = [], [], 0
for label in root.glob("*/labels/*.txt"):
    image_dir = label.parent.parent / "images"
    image = next((image_dir / (label.stem + e) for e in (".jpg", ".jpeg", ".png", ".webp")
                  if (image_dir / (label.stem + e)).exists()), None)
    if image is None:
        continue
    boxes = []
    for line in label.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            odd_lines += 1 if p else 0
            continue
        if p[0] == cid:
            boxes.append([float(v) for v in p[1:]])
            areas.append(float(p[3]) * float(p[4]))
    if boxes:
        samples.append((image, boxes))

areas.sort()
print(f"Class {cid} ({class_name}): {len(samples)} images, {len(areas)} boxes")
if areas:
    print(f"Box area as fraction of image -> median {areas[len(areas) // 2]:.3f}, "
          f"10th pct {areas[len(areas) // 10]:.3f}, 90th pct {areas[len(areas) * 9 // 10]:.3f}")
if odd_lines:
    print(f"Note: {odd_lines} label lines do not have 5 numbers (for example polygons) and were skipped.")
if not samples:
    sys.exit("Nothing to draw.")

random.seed(0)
if len(samples) > limit:
    samples = random.sample(samples, limit)

W, H, cols = 220, 180, 6
sheet = Image.new("RGB", (cols * W, math.ceil(len(samples) / cols) * H), "white")
draw = ImageDraw.Draw(sheet)
for i, (path, boxes) in enumerate(samples):
    im = Image.open(path).convert("RGB")
    im.thumbnail((W - 10, H - 30))
    x, y = (i % cols) * W, (i // cols) * H
    sheet.paste(im, (x + 5, y + 22))
    for cx, cy, bw, bh in boxes:
        draw.rectangle((x + 5 + (cx - bw / 2) * im.width, y + 22 + (cy - bh / 2) * im.height,
                        x + 5 + (cx + bw / 2) * im.width, y + 22 + (cy + bh / 2) * im.height),
                       outline="red", width=2)
    draw.text((x + 5, y + 5), f"Sample {i + 1}", fill="black")

out_dir = Path("datasets/VigilenseExtra/sheets")
out_dir.mkdir(parents=True, exist_ok=True)
sheet_path = out_dir / f"{name}_class{cid}_samples.jpg"
sheet.save(sheet_path)
with open(out_dir / f"{name}_class{cid}_samples.csv", "w") as f:
    f.write("sample,image\n")
    for i, (path, _) in enumerate(samples):
        f.write(f"{i + 1},{path}\n")
print("Saved:", sheet_path)