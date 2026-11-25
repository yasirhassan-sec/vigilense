"""
Build the weapons_v2 training set on THIS laptop (no Colab).

  clean_v1 (group-aware split, gun/knife)  +  webcam frames with NO weapon as background images
  -> datasets/VigilenseWeapons/weapons_v2_bg/  (train / valid / test + data.yaml)

Background images get an EMPTY label file, which teaches the model "this is not a knife/gun".
Whole webcam sessions go to train, except the hold-out sessions, which are kept unseen so the
false-alarm test after training is honest.

Run from the inner vigilense folder:
    python prepare_weapons_v2.py                         (hold out face_wall)
    python prepare_weapons_v2.py --holdout face_wall empty_room
It only reads clean_v1 and the webcam folders; it never overwrites an existing output folder.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

SRC = Path("datasets/VigilenseWeapons/clean_v1")
OUT = Path("datasets/VigilenseWeapons/weapons_v2_bg")
WEBCAM = Path("datasets/VigilenseWebcam")
IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

p = argparse.ArgumentParser()
p.add_argument("--holdout", nargs="*", default=["face_wall"],
               help="webcam sessions NOT used for training (kept for the false-alarm test)")
args = p.parse_args()

for split in ("train", "valid", "test"):
    if not (SRC / split / "images").exists():
        sys.exit(f"{SRC / split / 'images'} not found. Run build_clean_dataset.py first (it must print OK).")
if OUT.exists():
    sys.exit(f"{OUT} already exists. Delete it or rename it first. Nothing was changed.")


def put(src, dst):
    """Hard link when possible (no extra disk space), otherwise copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


counts = {}
for split in ("train", "valid", "test"):
    n = 0
    for img in (SRC / split / "images").iterdir():
        if img.suffix.lower() not in IMG:
            continue
        lbl = SRC / split / "labels" / (img.stem + ".txt")
        put(img, OUT / split / "images" / img.name)
        if lbl.exists():
            put(lbl, OUT / split / "labels" / lbl.name)
        else:
            (OUT / split / "labels" / (img.stem + ".txt")).write_text("")
        n += 1
    counts[split] = n

sessions = sorted(d for d in WEBCAM.iterdir() if d.is_dir()) if WEBCAM.exists() else []
added = {}
for s in sessions:
    if s.name in args.holdout:
        continue
    files = [f for f in s.glob("*") if f.suffix.lower() in IMG]
    for f in files:
        name = f"bg_{s.name}_{f.stem}"
        put(f, OUT / "train" / "images" / (name + f.suffix.lower()))
        (OUT / "train" / "labels" / (name + ".txt")).write_text("")      # empty = background
    added[s.name] = len(files)

(OUT / "data.yaml").write_text(
    f"path: {OUT.resolve().as_posix()}\n"
    "train: train/images\nval: valid/images\ntest: test/images\n"
    "nc: 2\nnames: ['gun', 'knife']\n")

print("Images copied from clean_v1:", counts)
print("Background frames added to train:", added, "| total", sum(added.values()))
print("Held out for the false-alarm test:", args.holdout)
print("Saved:", OUT / "data.yaml")
