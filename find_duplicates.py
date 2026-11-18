import random, math
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

root = Path("datasets/VigilenseWeapons/raw/gun_knife")
items = []   # (split, image path, classes)
for label in root.glob("*/labels/*.txt"):
    split = label.parent.parent.name
    image_dir = label.parent.parent / "images"
    image = next((image_dir / (label.stem + e) for e in [".jpg", ".jpeg", ".png", ".webp"]
                  if (image_dir / (label.stem + e)).exists()), None)
    if image is None:
        continue
    classes = sorted({l.split()[0] for l in label.read_text().splitlines() if len(l.split()) == 5})
    items.append((split, image, classes))
n = len(items)

def features(path):
    im = Image.open(path).convert("RGB")
    g = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()                       # 256-bit shape hash
    thumb = np.asarray(im.resize((8, 8), Image.BILINEAR), dtype=np.int16).flatten()  # colour check
    return np.packbits(bits), thumb

print("hashing", n, "images ...")
H = np.zeros((n, 32), dtype=np.uint8)
T = np.zeros((n, 192), dtype=np.int16)
for k, (_, p, _) in enumerate(items):
    H[k], T[k] = features(p)

table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
parent = list(range(n))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a

cross = []                                   # pairs whose images are in different splits
for i in range(n - 1):
    d = table[np.bitwise_xor(H[i + 1:], H[i])].sum(axis=1)
    for j in np.nonzero(d <= 16)[0] + i + 1:
        if np.abs(T[i] - T[j]).mean() <= 15:                       # same shape AND same colours
            parent[find(i)] = find(j)
            if items[i][0] != items[j][0]:
                cross.append((i, j))

groups = defaultdict(list)
for k in range(n):
    groups[find(k)].append(k)

sizes = Counter(min(len(v), 6) for v in groups.values())
print(f"\ngroups: {len(groups)}  (size 1: {sizes[1]}, size 2: {sizes[2]}, size 3-5: {sizes[3]+sizes[4]+sizes[5]}, size 6+: {sizes[6]})")
print("largest group:", max(len(v) for v in groups.values()), "images")

for s in ("valid", "test"):
    total = sum(1 for it in items if it[0] == s)
    leaked = sum(1 for v in groups.values() if any(items[k][0] == "train" for k in v)
                 for k in v if items[k][0] == s)
    print(f"{s}: {leaked} of {total} images share a group with a train image")

with open("groups.csv", "w") as f:
    f.write("group,split,classes,image\n")
    for gid, v in enumerate(groups.values()):
        for k in v:
            f.write(f"{gid},{items[k][0]},{'+'.join(items[k][2])},{items[k][1]}\n")
print("Saved: groups.csv")

random.seed(1)
pairs = random.sample(cross, min(24, len(cross)))
W, H_, cols = 200, 150, 2
PW = 2 * W + 30
sheet = Image.new("RGB", (cols * PW, math.ceil(len(pairs) / cols) * (H_ + 20)), "white")
draw = ImageDraw.Draw(sheet)
for q, (i, j) in enumerate(pairs):
    x0, y0 = (q % cols) * PW, (q // cols) * (H_ + 20)
    draw.text((x0 + 5, y0 + 3), f"Pair {q + 1}: {items[i][0]} | {items[j][0]}", fill="black")
    for k, idx in enumerate((i, j)):
        im = Image.open(items[idx][1]).convert("RGB")
        im.thumbnail((W, H_ - 5))
        sheet.paste(im, (x0 + k * (W + 10), y0 + 18))
sheet.save("dup_pairs2.jpg")
print("Saved: dup_pairs2.jpg (random pairs from", len(cross), "cross-split matches)")