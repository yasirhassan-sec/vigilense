"""
Download one Roboflow Universe dataset into datasets/VigilenseExtra/raw/<name>. It never overwrites anything.

Example (the vape dataset):
    python download_roboflow.py my-workspace-83t16 vapes_object_detection 7 vape_a

<workspace> and <project> are the two parts after universe.roboflow.com/ in the dataset's web address.
The Roboflow API key is typed in when asked (hidden, and not stored anywhere).
"""
import sys
from collections import Counter
from getpass import getpass
from pathlib import Path

from roboflow import Roboflow

if len(sys.argv) != 5:
    sys.exit("Usage: python download_roboflow.py <workspace> <project> <version> <name>")
workspace, project, version, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

out = Path("datasets/VigilenseExtra/raw") / name
if out.exists():
    sys.exit(f"{out} already exists. Nothing was downloaded.")

rf = Roboflow(api_key=getpass("Enter your Roboflow API key: "))
rf.workspace(workspace).project(project).version(version).download("yolov8", location=str(out))

print("\nSaved to:", out)
yaml = out / "data.yaml"
print(yaml.read_text() if yaml.exists() else "data.yaml not found")

for split in ("train", "valid", "test"):
    d = out / split / "images"
    print(f"{split:6s} images: {len(list(d.glob('*'))) if d.exists() else 0}")

counts = Counter()
for label in out.glob("*/labels/*.txt"):
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            counts[parts[0]] += 1
print("boxes per class id:", dict(sorted(counts.items(), key=lambda kv: int(kv[0]))))