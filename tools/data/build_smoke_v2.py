"""
Build the fire/smoke dataset fire_smoke_v2 on THIS laptop.

  Sources:  D-Fire (datasets/VigilenseExtra/raw/dfire) + your webcam sessions as background images.
  Why:      the old model saw only D-Fire, so a face against a wall looked like smoke. Here the webcam
            scenes are added as 'nothing here' images (train only), and two sessions are held out
            (face_wall, phone_hand_b) so the false-alarm test is fair.

D-Fire has many near-identical frames (video grabs). All images are hashed, near-duplicates are put in one
group, and each WHOLE group goes to train, valid or test (80/10/10), so no test image has a twin in train.

    python build_smoke_v2.py --check      (show class counts and which id is fire/smoke, build nothing)
    python build_smoke_v2.py              (build)
    python build_smoke_v2.py --swap       (use if --check says the fire/smoke ids are the other way round)

It never changes the source folders and refuses to overwrite an existing output folder.
"""
import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("datasets/VigilenseExtra/raw/dfire")
WEBCAM = Path("datasets/VigilenseWebcam")
OUT = Path("datasets/VigilenseFire/fire_smoke_v2")
IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NAMES = ["fire", "smoke"]
TARGET = {"train": 0.80, "valid": 0.10, "test": 0.10}

p = argparse.ArgumentParser()
p.add_argument("--src", default=str(SRC))
p.add_argument("--holdout", nargs="*", default=["face_wall", "phone_hand_b"],
               help="webcam sessions kept OUT of training (used later for the false-alarm test)")
p.add_argument("--bg-repeat", type=int, default=3, help="copies of each webcam background frame in train")
p.add_argument("--swap", action="store_true", help="D-Fire id 0 = fire and id 1 = smoke (default: 0 = smoke, 1 = fire)")
p.add_argument("--check", action="store_true")
args = p.parse_args()
SRC = Path(args.src)


def label_of(img):
    parts = list(img.parts)
    if "images" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


# D-Fire README: class 0 = smoke, class 1 = fire (fire has MORE boxes: 14,692 vs 11,865). Checked below.
ID_TO_NAME = {"0": "fire", "1": "smoke"} if args.swap else {"0": "smoke", "1": "fire"}

if not SRC.exists():
    sys.exit(f"{SRC} not found. Run download_dfire.py first.")
items, counts = [], Counter()
for img in SRC.rglob("*"):
    if img.suffix.lower() not in IMG:
        continue
    lbl = label_of(img)
    if lbl is None or not lbl.exists():
        continue
    lines = []
    for line in lbl.read_text().splitlines():
        q = line.split()
        if len(q) != 5 or q[0] not in ("0", "1"):
            continue
        counts[q[0]] += 1
        lines.append(f"{NAMES.index(ID_TO_NAME[q[0]])} " + " ".join(q[1:]))
    items.append({"img": img, "lines": lines,
                  "classes": sorted({NAMES[int(l.split()[0])] for l in lines}) or ["bg"]})
if not items:
    sys.exit(f"No image+label pairs found under {SRC} (need folders called 'images' and 'labels').")

print(f"D-Fire images with labels: {len(items)} | empty (background): {sum(1 for it in items if not it['lines'])}")
print(f"Boxes per raw class id: 0 -> {counts['0']}, 1 -> {counts['1']}")
more = "1" if counts["1"] > counts["0"] else "0"
fire_id = [k for k, v in ID_TO_NAME.items() if v == "fire"][0]
print(f"Mapping used: {ID_TO_NAME}.  The README says fire has more boxes than smoke; "
      f"id {more} has more here, and the mapping calls id {fire_id} fire -> "
      + ("consistent." if more == fire_id else "NOT consistent: look at a few images, then re-run with/without --swap."))

sessions = sorted(d for d in WEBCAM.iterdir() if d.is_dir()) if WEBCAM.exists() else []
bg = [(s.name, f) for s in sessions if s.name not in args.holdout
      for f in sorted(s.glob("*")) if f.suffix.lower() in IMG]
print(f"Webcam background frames: {len(bg)} x{args.bg_repeat} in train "
      f"(sessions: {', '.join(s.name for s in sessions if s.name not in args.holdout)}) | held out: {args.holdout}")
if args.check:
    sys.exit("\n--check only: nothing was built.")
if OUT.exists():
    sys.exit(f"{OUT} already exists. Delete or rename it first. Nothing was changed.")

# ---------------------------------------------------------------- near-duplicate groups
n = len(items)
print(f"\nHashing {n} images (several minutes) ...")
H = np.zeros((n, 4), dtype=np.uint64)
T = np.zeros((n, 192), dtype=np.int16)
for k, it in enumerate(items):
    im = Image.open(it["img"]).convert("RGB")
    g = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    H[k] = np.packbits((g[:, 1:] > g[:, :-1]).flatten()).view(np.uint64)
    T[k] = np.asarray(im.resize((8, 8), Image.BILINEAR), dtype=np.int16).flatten()
    if k % 2000 == 0:
        print(f"  {k}/{n}")

parent = list(range(n))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


print("Comparing images ...")
for i in range(n - 1):
    d = np.bitwise_count(H[i + 1:] ^ H[i]).sum(axis=1)
    for j in np.nonzero(d <= 16)[0] + i + 1:
        if np.abs(T[i] - T[j]).mean() <= 15:
            parent[find(i)] = find(j)
    if i % 2000 == 0:
        print(f"  {i}/{n}")
comps = defaultdict(list)
for i in range(n):
    comps[find(i)].append(i)
print(f"Groups: {len(comps)} (from {n} images)")

random.seed(7)
total = Counter(c for it in items for c in it["classes"])
cur = {s: Counter() for s in TARGET}
split_of = {}
for idx in sorted(comps.values(), key=lambda v: (-len(v), random.random())):
    need = Counter(c for i in idx for c in items[i]["classes"])
    best = min(TARGET, key=lambda s: sum(cnt * (cur[s][c] + 1) / (TARGET[s] * total[c]) for c, cnt in need.items()))
    for i in idx:
        split_of[i] = best
    cur[best].update(need)


def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_label(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


print("\nWriting", OUT, "...")
stats = {s: Counter() for s in TARGET}
OUT.mkdir(parents=True)
with open(OUT.parent / "fire_smoke_v2_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["new_split", "group", "classes", "file", "original"])
    for i, it in enumerate(items):
        s = split_of[i]
        stem = f"dfire__{it['img'].stem}"
        put(it["img"], OUT / s / "images" / (stem + it["img"].suffix.lower()))
        write_label(OUT / s / "labels" / (stem + ".txt"), it["lines"])
        stats[s]["images"] += 1
        for l in it["lines"]:
            stats[s][NAMES[int(l.split()[0])]] += 1
        if not it["lines"]:
            stats[s]["background"] += 1
        w.writerow([s, find(i), "+".join(it["classes"]), stem, it["img"]])
    for sess, f_ in bg:
        for r in range(args.bg_repeat):
            stem = f"webcam__{sess}__{f_.stem}__r{r}"
            put(f_, OUT / "train" / "images" / (stem + f_.suffix.lower()))
            write_label(OUT / "train" / "labels" / (stem + ".txt"), [])
            stats["train"]["images"] += 1
            stats["train"]["background"] += 1
        w.writerow(["train", "", "bg", f"webcam__{sess}__{f_.stem}", f_])

(OUT / "data.yaml").write_text(f"path: {OUT.resolve().as_posix()}\n"
                               "train: train/images\nval: valid/images\ntest: test/images\n"
                               "nc: 2\nnames: ['fire', 'smoke']\n")
print(f"\n{'split':6s} {'images':>7s} {'fire':>7s} {'smoke':>7s} {'backgr.':>7s}")
for s in TARGET:
    print(f"{s:6s} {stats[s]['images']:7d} {stats[s]['fire']:7d} {stats[s]['smoke']:7d} {stats[s]['background']:7d}")
print(f"\nSaved {OUT / 'data.yaml'}\nNext:  python train.py --data {OUT.as_posix()}/data.yaml --name fire_smoke_v2")
print("OK")
