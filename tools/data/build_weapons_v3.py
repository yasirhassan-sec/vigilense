"""
Build the merged gun/knife dataset weapons_v3 on THIS laptop.

  gun   = every firearm: pistol/handgun, revolver, rifle, automatic rifle, shotgun, SMG, sniper, ...
  knife = knife, blade, machete, dagger, ...

Sources (all YOLO format):
  1. datasets/VigilenseWeapons/raw/gun_knife   (Roboflow, mostly handguns + knives)
  2. datasets/DangerousItems/raw               (Zenodo: rifle, gun, knife, machete; baseball bat dropped)
  3. any extra dataset you add with --extra <folder>  (class names read from its data.yaml)
  4. webcam sessions in datasets/VigilenseWebcam as background (no weapon) images, train only

All images from all sources are hashed together. Near-duplicates (same photo, other crop/watermark)
and frames of the same video are put in one group, and each WHOLE group goes to train, valid or test
(80/10/10). So no test image has a copy in train, also not across datasets.

    python build_weapons_v3.py --check                 (show class mapping and counts, build nothing)
    python build_weapons_v3.py                         (build)
    python build_weapons_v3.py --extra datasets/VigilenseExtra/raw/rifles_a --holdout face_wall phone_hand_b

It never deletes or changes the source datasets, and refuses to overwrite an existing output folder.
"""
import argparse
import csv
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

OUT = Path("datasets/VigilenseWeapons/weapons_v3")
WEBCAM = Path("datasets/VigilenseWebcam")
IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NAMES = ["gun", "knife"]
TARGET = {"train": 0.80, "valid": 0.10, "test": 0.10}

# Fixed id maps for the two datasets whose class ids we already checked by eye.
FIXED = {
    "gun_knife": {"0": "knife", "1": "gun", "2": "gun", "3": "gun", "4": "knife", "5": "gun"},
    "dangerous": {"0": "knife", "1": "knife", "2": None, "3": "gun", "4": "gun"},  # machete, knife, bat, rifle, gun
}
GUN_WORDS = ["gun", "pistol", "handgun", "revolver", "rifle", "shotgun", "smg", "submachine", "firearm",
             "sniper", "carbine", "ak47", "ak-47", "ak 47", "m4", "m16", "uzi", "glock", "bazooka",
             "launcher", "rpg"]
KNIFE_WORDS = ["knife", "knives", "blade", "machete", "dagger", "sword", "cleaver"]
DROP_WORDS = ["person", "people", "human", "hand", "face", "no_weapon", "no weapon", "background",
              "bat", "phone", "bottle", "stick"]

p = argparse.ArgumentParser()
p.add_argument("--extra", nargs="*", default=[], help="extra YOLO dataset folders (each with data.yaml)")
p.add_argument("--map", nargs="*", default=[], help="override a class name: --map 'weapon=gun' 'tool=drop'")
p.add_argument("--holdout", nargs="*", default=["face_wall", "phone_hand_b"],
               help="webcam sessions kept OUT of training (used later for the false-alarm test)")
p.add_argument("--check", action="store_true", help="only print mappings and counts")
args = p.parse_args()
OVERRIDE = {k.strip().lower(): (None if v.strip().lower() == "drop" else v.strip().lower())
            for k, v in (m.split("=", 1) for m in args.map)}


def by_name(name):
    n = name.strip().lower()
    if n in OVERRIDE:
        return OVERRIDE[n], True
    if any(w in n for w in DROP_WORDS) and not any(w in n for w in GUN_WORDS + KNIFE_WORDS):
        return None, True
    if any(w in n for w in KNIFE_WORDS):
        return "knife", True
    if any(w in n for w in GUN_WORDS):
        return "gun", True
    return None, False


def label_of(img):
    parts = list(img.parts)
    if "images" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def orig_split(path):
    s = [x.lower() for x in path.parts]
    for key, names in (("train", ("train", "training")), ("valid", ("val", "valid", "validation")),
                       ("test", ("test", "testing"))):
        if any(x in names for x in s):
            return key
    return "?"


sources = []
if Path("datasets/VigilenseWeapons/raw/gun_knife").exists():
    sources.append(("gun_knife", Path("datasets/VigilenseWeapons/raw/gun_knife"), FIXED["gun_knife"]))
else:
    print("WARNING: datasets/VigilenseWeapons/raw/gun_knife not found (skipped)")
if Path("datasets/DangerousItems/raw").exists():
    sources.append(("dangerous", Path("datasets/DangerousItems/raw"), FIXED["dangerous"]))
else:
    print("WARNING: datasets/DangerousItems/raw not found - run download_dangerous_items.py (skipped)")

unknown = []
for folder in args.extra:
    folder = Path(folder)
    y = folder / "data.yaml"
    if not y.exists():
        sys.exit(f"{y} not found")
    names = yaml.safe_load(y.read_text())["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    m = {}
    for i, n in enumerate(names):
        target, known = by_name(n)
        if not known:
            unknown.append(f"{folder.name}: '{n}'")
        m[str(i)] = target
    sources.append((re.sub(r"\W+", "_", folder.name), folder, m))

print("\nClass mapping (dataset id -> VigiLense class, None = dropped):")
for name, folder, m in sources:
    print(f"  {name:12s}", {k: v for k, v in m.items()})
if unknown:
    sys.exit("\nUnknown class names: " + ", ".join(unknown) +
             "\nTell the script what they are, e.g.  --map 'weapon=gun'  or  --map 'tool=drop'")
if not sources:
    sys.exit("No source datasets found.")

# ---------------------------------------------------------------- collect images and remapped labels
items = []      # dict(src, img, split, lines, classes)
dropped_boxes = Counter()
for name, folder, m in sources:
    for img in folder.rglob("*"):
        if img.suffix.lower() not in IMG:
            continue
        lbl = label_of(img)
        if lbl is None or not lbl.exists():
            continue
        lines, classes = [], set()
        for line in lbl.read_text().splitlines():
            q = line.split()
            if len(q) < 5:
                continue
            target = m.get(str(int(float(q[0]))))
            if target is None:
                dropped_boxes[(name, q[0])] += 1
                continue
            if len(q) == 5:
                box = q[1:]
            else:                                        # polygon -> bounding box
                xs, ys = [float(v) for v in q[1::2]], [float(v) for v in q[2::2]]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                box = [f"{(x1 + x2) / 2:.6f}", f"{(y1 + y2) / 2:.6f}", f"{x2 - x1:.6f}", f"{y2 - y1:.6f}"]
            lines.append(" ".join([str(NAMES.index(target))] + box))
            classes.add(target)
        items.append({"src": name, "img": img, "split": orig_split(img.relative_to(folder)),
                      "lines": lines, "classes": sorted(classes) or ["bg"]})

print("\nImages per source:", dict(Counter(it["src"] for it in items)))
boxes = Counter()
for it in items:
    for l in it["lines"]:
        boxes[(it["src"], NAMES[int(l.split()[0])])] += 1
for name, _, _ in sources:
    print(f"  {name:12s} gun boxes {boxes[(name, 'gun')]:6d} | knife boxes {boxes[(name, 'knife')]:6d}")
if dropped_boxes:
    print("Dropped boxes (source, class id):", dict(dropped_boxes))

sessions = sorted(d for d in WEBCAM.iterdir() if d.is_dir()) if WEBCAM.exists() else []
bg = [(s.name, f) for s in sessions if s.name not in args.holdout
      for f in sorted(s.glob("*")) if f.suffix.lower() in IMG]
print(f"Webcam background frames for train: {len(bg)} "
      f"(sessions: {', '.join(s.name for s in sessions if s.name not in args.holdout)}) | held out: {args.holdout}")
if args.check:
    sys.exit("\n--check only: nothing was built.")
if OUT.exists():
    sys.exit(f"{OUT} already exists. Delete or rename it first. Nothing was changed.")

# ---------------------------------------------------------------- near-duplicate groups over ALL sources
n = len(items)
print(f"\nHashing {n} images (a few minutes) ...")
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


def union(a, b):
    parent[find(a)] = find(b)


print("Comparing images ...")
for i in range(n - 1):
    d = np.bitwise_count(H[i + 1:] ^ H[i]).sum(axis=1)
    for j in np.nonzero(d <= 16)[0] + i + 1:
        if np.abs(T[i] - T[j]).mean() <= 15:
            union(i, j)
    if i % 2000 == 0:
        print(f"  {i}/{n}")


def video_name(path):
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", Path(path).stem)
    stem = re.sub(r"_(jpg|jpeg|png)$", "", stem)
    return re.sub(r"[_-]?\d+$", "", stem)


by_video = defaultdict(list)
for i, it in enumerate(items):
    v = video_name(it["img"])
    if v.lower() not in {"", "knife", "pistol", "gun", "image", "img"} and not re.fullmatch(r"[0-9a-f]+", v):
        by_video[(it["src"], v)].append(i)
for idx in by_video.values():
    if len(idx) >= 3:
        for k in idx[1:]:
            union(idx[0], k)
room = [i for (src, v), idx in by_video.items()
        if src == "gun_knife" and (v.endswith("frame") or v.startswith("pistola_z3v")) for i in idx]
for k in room[1:]:
    union(room[0], k)

comps = defaultdict(list)
for i in range(n):
    comps[find(i)].append(i)

# leakage of the original splits (for the report)
for name, _, _ in sources:
    for s in ("valid", "test"):
        tot = [i for i in range(n) if items[i]["src"] == name and items[i]["split"] == s]
        if tot:
            has_train = {find(i) for i in range(n) if items[i]["split"] == "train"}
            leak = sum(1 for i in tot if find(i) in has_train)
            print(f"  original {name} {s}: {leak}/{len(tot)} images ({100 * leak / len(tot):.0f}%) have a copy in some train split")

# ---------------------------------------------------------------- group-aware 80/10/10 split
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


def write_label(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


print("\nWriting", OUT, "...")
stats = {s: Counter() for s in TARGET}
OUT.mkdir(parents=True)
with open(OUT.parent / "weapons_v3_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["new_split", "group", "source", "old_split", "classes", "file", "original"])
    for i, it in enumerate(items):
        s = split_of[i]
        stem = f"{it['src']}__{it['img'].stem}"
        put(it["img"], OUT / s / "images" / (stem + it["img"].suffix.lower()))
        write_label(OUT / s / "labels" / (stem + ".txt"), it["lines"])
        stats[s]["images"] += 1
        for l in it["lines"]:
            stats[s][NAMES[int(l.split()[0])]] += 1
        if not it["lines"]:
            stats[s]["background"] += 1
        w.writerow([s, find(i), it["src"], it["split"], "+".join(it["classes"]), stem, it["img"]])
    for sess, f_ in bg:
        stem = f"webcam__{sess}__{f_.stem}"
        put(f_, OUT / "train" / "images" / (stem + f_.suffix.lower()))
        write_label(OUT / "train" / "labels" / (stem + ".txt"), [])
        stats["train"]["images"] += 1
        stats["train"]["background"] += 1
        w.writerow(["train", "", "webcam", "", "bg", stem, f_])

(OUT / "data.yaml").write_text(f"path: {OUT.resolve().as_posix()}\n"
                               "train: train/images\nval: valid/images\ntest: test/images\n"
                               "nc: 2\nnames: ['gun', 'knife']\n")
print(f"\n{'split':6s} {'images':>7s} {'gun':>7s} {'knife':>7s} {'backgr.':>7s}")
for s in TARGET:
    print(f"{s:6s} {stats[s]['images']:7d} {stats[s]['gun']:7d} {stats[s]['knife']:7d} {stats[s]['background']:7d}")
print(f"\nGroups: {len(comps)} | saved {OUT / 'data.yaml'} and {OUT.parent / 'weapons_v3_manifest.csv'}")
print("OK")
