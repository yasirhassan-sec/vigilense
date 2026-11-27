"""
Why is knife recall lower on the weapons_v3 TEST split? Finds every ground-truth knife the model misses
and breaks recall down by source, box size and image group size (big group = frames of one video).

    python analyze_knife_misses.py
    python analyze_knife_misses.py --split test --conf 0.25 --model weights/weapons_v3.pt

Saves runs/knife_misses.csv (every knife, hit or miss) and runs/knife_misses_sheet.jpg (up to 30 misses,
green = real knife, red = model prediction). Read-only on the dataset.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

KNIFE = 1  # class ids: 0 gun, 1 knife


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def size_bin(frac):
    return ("<0.5% of image" if frac < 0.005 else "0.5-2%" if frac < 0.02 else "2-10%" if frac < 0.10 else ">10%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="weights/weapons_v3.pt")
    p.add_argument("--data", default="datasets/VigilenseWeapons/weapons_v3")
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()

    root = Path(a.data)
    manifest = root.parent / "weapons_v3_manifest.csv"
    group_of, group_size = {}, Counter()
    if manifest.exists():
        with open(manifest, newline="") as f:
            for r in csv.DictReader(f):
                group_of[r["file"]] = r["group"]
                group_size[r["group"]] += 1
    else:
        print("manifest not found - group sizes will be 'unknown'")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(a.model)
    imgs = sorted((root / a.split / "images").iterdir())
    rows, miss_examples = [], []
    for i in range(0, len(imgs), 16):
        chunk = imgs[i:i + 16]
        for img, r in zip(chunk, model.predict([str(x) for x in chunk], conf=a.conf, device=device, verbose=False)):
            lbl = root / a.split / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            h, w = r.orig_shape
            preds = [(b.tolist(), float(s)) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy) if int(c) == KNIFE]
            for line in lbl.read_text().splitlines():
                q = line.split()
                if len(q) != 5 or int(q[0]) != KNIFE:
                    continue
                cx, cy, bw, bh = (float(v) for v in q[1:])
                gt = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                best = max((iou(gt, pb) for pb, _ in preds), default=0.0)
                hit = best >= 0.5
                g = group_of.get(img.stem, "")
                rows.append([img.name, img.name.split("__")[0], round(bw * bh, 5), size_bin(bw * bh),
                             group_size.get(g, 0), int(hit), round(best, 2)])
                if not hit and len(miss_examples) < 30:
                    miss_examples.append((img, gt, [pb for pb, _ in preds]))

    out = Path("runs")
    out.mkdir(exist_ok=True)
    with open(out / "knife_misses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "source", "box_area_fraction", "size_bin", "group_size", "found", "best_iou"])
        w.writerows(rows)

    def table(title, keyfn, order=None):
        d = defaultdict(lambda: [0, 0])
        for r in rows:
            k = keyfn(r)
            d[k][0] += 1
            d[k][1] += r[5]
        print(f"\n{title}")
        for k in (order or sorted(d)):
            if k in d:
                n, h = d[k]
                print(f"  {str(k):18s} knives {n:4d} | found {h:4d} | recall {100 * h / n:5.1f}%")

    print(f"\nKnife recall on {a.split} at conf {a.conf}, IoU 0.5 (model {a.model}): "
          f"{sum(r[5] for r in rows)}/{len(rows)}")
    table("By source", lambda r: r[1])
    table("By box size (share of image area)", lambda r: r[3], ["<0.5% of image", "0.5-2%", "2-10%", ">10%"])
    table("By group size (images that look like the same scene/video)",
          lambda r: "1-2" if r[4] <= 2 else "3-10" if r[4] <= 10 else "11+" if r[4] > 10 else "?",
          ["1-2", "3-10", "11+"])

    tiles = []
    for img, gt, preds in miss_examples:
        im = cv2.imread(str(img))
        if im is None:
            continue
        cv2.rectangle(im, tuple(map(int, gt[:2])), tuple(map(int, gt[2:])), (0, 200, 0), 3)
        for pb in preds:
            cv2.rectangle(im, tuple(map(int, pb[:2])), tuple(map(int, pb[2:])), (0, 0, 255), 2)
        tiles.append(cv2.resize(im, (320, 240)))
    if tiles:
        while len(tiles) % 5:
            tiles.append(tiles[0] * 0)
        sheet = cv2.vconcat([cv2.hconcat(tiles[k:k + 5]) for k in range(0, len(tiles), 5)])
        cv2.imwrite(str(out / "knife_misses_sheet.jpg"), sheet)
        print(f"\nSaved {out / 'knife_misses_sheet.jpg'} ({len(miss_examples)} misses) and {out / 'knife_misses.csv'}")


if __name__ == "__main__":
    main()
