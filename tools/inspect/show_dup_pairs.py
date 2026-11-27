import csv, math
from pathlib import Path
from PIL import Image, ImageDraw

rows = list(csv.DictReader(open("near_dups.csv")))[:24]
W, H, cols = 200, 150, 2                     # 2 pairs per row
PW = 2 * W + 30
sheet = Image.new("RGB", (cols * PW, math.ceil(len(rows) / cols) * (H + 20)), "white")
draw = ImageDraw.Draw(sheet)

for i, r in enumerate(rows):
    x0, y0 = (i % cols) * PW, (i // cols) * (H + 20)
    draw.text((x0 + 5, y0 + 3), f"Pair {i + 1}: train (left) | other split (right)", fill="black")
    for k, key in enumerate(("train_image", "other_image")):
        im = Image.open(r[key]).convert("RGB")
        im.thumbnail((W, H - 5))
        sheet.paste(im, (x0 + k * (W + 10), y0 + 18))
    print(i + 1, Path(r["train_image"]).name[:35], "|", Path(r["other_image"]).name[:35])

sheet.save("dup_pairs.jpg")
print("Saved: dup_pairs.jpg,", len(rows), "pairs")