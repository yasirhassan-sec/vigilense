import csv, random, re
from collections import Counter, defaultdict
from pathlib import Path

random.seed(7)
TARGET = {"train": 0.80, "valid": 0.10, "test": 0.10}
NOT_VIDEO = {"", "knife", "pistol"}            # name prefixes that are not video frames

rows = list(csv.DictReader(open("groups.csv")))
n = len(rows)
parent = list(range(n))

def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a

def union(a, b):
    parent[find(a)] = find(b)

# 1) image-similarity groups from find_duplicates.py
first = {}
for i, r in enumerate(rows):
    g = r["group"]
    if g in first:
        union(i, first[g])
    else:
        first[g] = i

# 2) frames of the same video
def video_name(path):
    stem = Path(path).stem
    base = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    base = re.sub(r"_(jpg|jpeg|png)$", "", base)
    return re.sub(r"[_-]?\d+$", "", base)

by_name = defaultdict(list)
for i, r in enumerate(rows):
    name = video_name(r["image"])
    if name not in NOT_VIDEO and not re.fullmatch(r"[0-9a-f]+", name):
        by_name[name].append(i)

for name, idx in by_name.items():
    if len(idx) >= 3:
        for k in idx[1:]:
            union(idx[0], k)

# 3) the white-wall room videos: keep them all together
room = [i for name, idx in by_name.items()
        if name.endswith("frame") or name.startswith("pistola_z3v") for i in idx]
for k in room[1:]:
    union(room[0], k)

comps = defaultdict(list)
for i in range(n):
    comps[find(i)].append(i)

def classes_of(r):
    return r["classes"].split("+") if r["classes"] else ["bg"]

total = Counter(c for r in rows for c in classes_of(r))
cur = {s: Counter() for s in TARGET}
split_of = {}

# biggest groups first, each whole group goes to the split that is furthest below its target
for idx in sorted(comps.values(), key=lambda v: (-len(v), random.random())):
    need = Counter(c for i in idx for c in classes_of(rows[i]))
    def score(s):
        return sum(cnt * (cur[s][c] + 1) / (TARGET[s] * total[c]) for c, cnt in need.items())
    best = min(TARGET, key=score)
    for i in idx:
        split_of[i] = best
    cur[best].update(need)

print("total images:", n, "| groups after merging:", len(comps))
print("images per split:", {s: sum(1 for v in split_of.values() if v == s) for s in TARGET})
print("\nimages per class (train / valid / test):")
for c in sorted(total):
    row = [cur[s][c] for s in TARGET]
    t = sum(row)
    print(f"  class {c:>3s}: " + " / ".join(f"{x:5d} ({100 * x / t:4.1f}%)" for x in row))
print("\nlargest groups (images, split):",
      [(len(v), split_of[v[0]]) for v in sorted(comps.values(), key=len, reverse=True)[:8]])

with open("split_plan.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "old_split", "new_split", "group", "classes"])
    for i, r in enumerate(rows):
        w.writerow([r["image"], r["split"], split_of[i], find(i), r["classes"]])
print("Saved: split_plan.csv")