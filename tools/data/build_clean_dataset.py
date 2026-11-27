import csv, shutil, sys
from collections import Counter
from pathlib import Path

PLAN = Path("split_plan.csv")
OUT = Path("datasets/VigilenseWeapons/clean_v1")
NAMES = ["gun", "knife"]
MAP = {"1": 0, "2": 0, "3": 0, "5": 0,      # handgun classes -> gun
       "0": 1, "4": 1}                      # knife classes   -> knife

if OUT.exists():
    sys.exit(f"{OUT} already exists. Delete it or pick another name. Nothing was changed.")

rows = list(csv.DictReader(open(PLAN)))
boxes = Counter()
images = Counter()

for r in rows:
    img = Path(r["image"])
    split = r["new_split"]
    lbl = img.parent.parent / "labels" / (img.stem + ".txt")

    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, OUT / split / "images" / img.name)

    new_lines = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            new_class = MAP[p[0]]
            new_lines.append(" ".join([str(new_class)] + p[1:]))
            boxes[(split, NAMES[new_class])] += 1
    text = "\n".join(new_lines) + ("\n" if new_lines else "")
    (OUT / split / "labels" / (img.stem + ".txt")).write_text(text)
    images[split] += 1

yaml = (f"path: {OUT.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 2\nnames: ['gun', 'knife']\n")
(OUT / "data.yaml").write_text(yaml)

print("images:", dict(images))
for split in ("train", "valid", "test"):
    print(f"{split:6s} gun boxes: {boxes[(split, 'gun')]:5d} | knife boxes: {boxes[(split, 'knife')]:5d}")
gun = sum(v for (s, c), v in boxes.items() if c == "gun")
knife = sum(v for (s, c), v in boxes.items() if c == "knife")
print(f"total gun {gun} (expected 7801) | total knife {knife} (expected 1883)")
print("OK" if (gun, knife) == (7801, 1883) else "MISMATCH - tell me before doing anything else")
print("Saved:", OUT)