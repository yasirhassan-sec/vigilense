import sys, random, math
from pathlib import Path
from PIL import Image, ImageDraw

cid = sys.argv[1]                                   # class id to inspect
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 96
root = Path("datasets/VigilenseWeapons/raw/gun_knife")

samples, areas = [], []
for label in root.glob("*/labels/*.txt"):
    image_dir = label.parent.parent / "images"
    image = next((image_dir / (label.stem + e) for e in [".jpg", ".jpeg", ".png", ".webp"]
                  if (image_dir / (label.stem + e)).exists()), None)
    if image is None:
        continue
    boxes = []
    for line in label.read_text().splitlines():
        p = line.split()
        if len(p) == 5 and p[0] == cid:
            boxes.append([float(v) for v in p[1:]])
            areas.append(float(p[3]) * float(p[4]))
    if boxes:
        samples.append((image, boxes))

areas.sort()
print(f"Class {cid}: {len(samples)} images, {len(areas)} boxes")
if areas:
    print(f"Box area as fraction of image -> median {areas[len(areas)//2]:.3f}, "
          f"10th pct {areas[len(areas)//10]:.3f}, 90th pct {areas[len(areas)*9//10]:.3f}")

random.seed(0)
if len(samples) > limit:
    samples = random.sample(samples, limit)

W, H, cols = 220, 180, 6
rows = math.ceil(len(samples) / cols)
sheet = Image.new("RGB", (cols * W, rows * H), "white")
draw = ImageDraw.Draw(sheet)
for i, (path, boxes) in enumerate(samples):
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    im.thumbnail((W - 10, H - 30))
    x, y = (i % cols) * W, (i // cols) * H
    sheet.paste(im, (x + 5, y + 22))
    for cx, cy, bw, bh in boxes:
        draw.rectangle((x + 5 + (cx - bw / 2) * im.width, y + 22 + (cy - bh / 2) * im.height,
                        x + 5 + (cx + bw / 2) * im.width, y + 22 + (cy + bh / 2) * im.height),
                       outline="red", width=2)
    draw.text((x + 5, y + 5), f"Sample {i + 1}", fill="black")

out = f"class{cid}_samples.jpg"
sheet.save(out)
print("Saved:", out)
with open(f"class{cid}_samples.csv", "w") as f:
    f.write("sample,image\n")
    for i, (path, _) in enumerate(samples):
        f.write(f"{i + 1},{path}\n")
print("Saved:", f"class{cid}_samples.csv")