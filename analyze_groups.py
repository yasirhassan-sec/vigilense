import re
from collections import Counter, defaultdict
from pathlib import Path

root = Path("datasets/VigilenseWeapons/raw/gun_knife")
base_splits = defaultdict(set)         # exact source image -> splits it appears in
group_splits = defaultdict(Counter)    # video/source group -> images per split
group_classes = defaultdict(Counter)   # video/source group -> boxes per class
files = 0

for label in root.glob("*/labels/*.txt"):
    split = label.parent.parent.name
    base = re.sub(r"\.rf\.[0-9a-f]+$", "", label.stem)           # source image name
    group = re.sub(r"[_-]?\d+$", "", re.sub(r"_(jpg|jpeg|png)$", "", base))
    files += 1
    base_splits[base].add(split)
    group_splits[group][split] += 1
    for line in label.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            group_classes[group][p[0]] += 1

dup = sum(1 for s in base_splits.values() if len(s) > 1)
leaky = [g for g, c in group_splits.items() if len(c) > 1]
print(f"label files: {files}, unique source images: {len(base_splits)}")
print(f"source images appearing in more than one split: {dup}")
print(f"groups: {len(group_splits)}, groups present in more than one split: {len(leaky)}\n")
print(f"{'group':40s} {'train':>6s} {'valid':>6s} {'test':>6s}  boxes per class")
for g, c in sorted(group_splits.items(), key=lambda kv: -sum(kv[1].values()))[:40]:
    cls = ", ".join(f"{k}:{v}" for k, v in sorted(group_classes[g].items()))
    print(f"{g[:40]:40s} {c['train']:6d} {c['valid']:6d} {c['test']:6d}  {cls}")