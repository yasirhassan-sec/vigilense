import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

root = Path("datasets/VigilenseWeapons/raw/gun_knife")
boxes = defaultdict(Counter)      # class -> split -> box count
images = defaultdict(Counter)     # class -> split -> image count
empty = Counter()                 # split -> label files with no boxes
items = []                        # (split, image path)

for label in root.glob("*/labels/*.txt"):
    split = label.parent.parent.name
    image_dir = label.parent.parent / "images"
    image = next((image_dir / (label.stem + e) for e in [".jpg", ".jpeg", ".png", ".webp"]
                  if (image_dir / (label.stem + e)).exists()), None)
    if image is None:
        continue
    items.append((split, image))
    seen = set()
    for line in label.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            boxes[p[0]][split] += 1
            seen.add(p[0])
    for c in seen:
        images[c][split] += 1
    if not seen:
        empty[split] += 1

print("PART A: images / boxes per class and split")
print(f"{'class':>5s} {'train':>12s} {'valid':>12s} {'test':>12s}")
for c in sorted(boxes):
    print(f"{c:>5s} " + " ".join(f"{images[c][s]:>5d}/{boxes[c][s]:<6d}" for s in ("train", "valid", "test")))
print("files with no boxes (backgrounds):", dict(empty), "\n")

print("PART B: near-duplicate images across splits (this takes a minute or two)")
def dhash(path):
    im = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view(np.uint64)[0]

hashes = np.array([dhash(p) for _, p in items], dtype=np.uint64)
splits = np.array([s for s, _ in items])
table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

has_dup_in_train = set()
pairs = []
for i in range(len(hashes) - 1):
    x = np.bitwise_xor(hashes[i + 1:], hashes[i])
    d = table[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)
    for j in np.nonzero(d <= 4)[0] + i + 1:
        a, b = (i, j) if splits[i] == "train" else (j, i)
        if splits[a] == "train" and splits[b] != "train":
            has_dup_in_train.add(b)
            if len(pairs) < 30:
                pairs.append((items[a][1], items[b][1]))

for s in ("valid", "test"):
    total = int((splits == s).sum())
    leaked = sum(1 for k in has_dup_in_train if splits[k] == s)
    print(f"{s}: {leaked} of {total} images have a near-duplicate in train")

with open("near_dups.csv", "w") as f:
    f.write("train_image,other_image\n")
    for a, b in pairs:
        f.write(f"{a},{b}\n")
print("Saved: near_dups.csv (first 30 pairs)")