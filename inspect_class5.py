from pathlib import Path
from PIL import Image, ImageDraw
import math

root = Path("datasets/VigilenseWeapons/raw/gun_knife")
samples = []

for label in root.glob("*/labels/*.txt"):
    image_dir = label.parent.parent / "images"
    image = next(
        (image_dir / (label.stem + ext)
         for ext in [".jpg", ".jpeg", ".png", ".webp"]
         if (image_dir / (label.stem + ext)).exists()),
        None
    )
    if image is None:
        continue

    boxes = []
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == "5":
            boxes.append([float(v) for v in parts[1:]])

    if boxes:
        samples.append((image, boxes))

W, H = 220, 180
cols = 6
rows = math.ceil(len(samples) / cols)
sheet = Image.new("RGB", (cols * W, rows * H), "white")
draw = ImageDraw.Draw(sheet)

for i, (path, boxes) in enumerate(samples):
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    im.thumbnail((W - 10, H - 30))
    x, y = (i % cols) * W, (i // cols) * H
    sheet.paste(im, (x + 5, y + 22))

    scale_x = im.width / iw
    scale_y = im.height / ih
    for cx, cy, bw, bh in boxes:
        x1 = x + 5 + (cx - bw/2) * iw * scale_x
        y1 = y + 22 + (cy - bh/2) * ih * scale_y
        x2 = x + 5 + (cx + bw/2) * iw * scale_x
        y2 = y + 22 + (cy + bh/2) * ih * scale_y
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)

    draw.text((x + 5, y + 5), f"Sample {i+1}", fill="black")

sheet.save("class5_samples.jpg")
print("Class 5 images:", len(samples))
print("Saved: class5_samples.jpg")
