"""
Evaluate a weapon model on the weapons_v3 valid and test splits, overall and per source dataset.
Uses the same Python/Ultralytics as training (not the 'yolo' command, which may come from another install).

    python eval_by_source.py                       (weights/weapons_v3.pt)
    python eval_by_source.py --model weights/weapons_v3.pt

Prints P, R, mAP50, mAP50-95 per class and saves runs/eval_weapons_v3_by_source.csv
"""
import argparse
import csv
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="weights/weapons_v3.pt")
p.add_argument("--data", default="datasets/VigilenseWeapons/weapons_v3")
a = p.parse_args()

root = Path(a.data).resolve()
lists = Path("runs/eval_lists").resolve()
lists.mkdir(parents=True, exist_ok=True)
device = 0 if torch.cuda.is_available() else "cpu"
print("Ultralytics", ultralytics.__version__, "| model", a.model)

model = YOLO(a.model)
rows = []
for split in ("valid", "test"):
    images = sorted((root / split / "images").iterdir())
    groups = {"all": images}
    for img in images:
        groups.setdefault(img.name.split("__")[0], []).append(img)
    for name, imgs in groups.items():
        lst = lists / f"{split}_{name}.txt"
        lst.write_text("\n".join(i.as_posix() for i in imgs) + "\n")
        yml = lists / f"{split}_{name}.yaml"
        yml.write_text(f"train: {lst.as_posix()}\nval: {lst.as_posix()}\nnc: 2\nnames: ['gun', 'knife']\n")
        m = model.val(data=str(yml), split="val", batch=16, device=device, plots=False, verbose=False,
                      project=str(Path("runs").resolve()), name="eval_tmp", exist_ok=True)
        rows.append([split, name, len(imgs), "all", m.box.mp, m.box.mr, m.box.map50, m.box.map])
        for k, idx in enumerate(m.box.ap_class_index):
            rows.append([split, name, len(imgs), m.names[int(idx)], m.box.p[k], m.box.r[k], m.box.ap50[k], m.box.ap[k]])

print(f"\n{'split':6s} {'source':11s} {'images':>6s} {'class':6s} {'P':>6s} {'R':>6s} {'mAP50':>6s} {'mAP50-95':>8s}")
for r in rows:
    print(f"{r[0]:6s} {r[1]:11s} {r[2]:6d} {r[3]:6s} {r[4]:6.3f} {r[5]:6.3f} {r[6]:6.3f} {r[7]:8.3f}")
out = Path("runs") / f"eval_{Path(a.model).stem}_by_source.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "source", "images", "class", "precision", "recall", "mAP50", "mAP50_95"])
    w.writerows([r[:4] + [round(x, 4) for x in r[4:]] for r in rows])
print("\nSaved:", out)
