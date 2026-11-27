"""
Compare gun/knife models on webcam sessions that contain NO weapon (every detection = false alarm).

    python compare_false_alarms.py
    python compare_false_alarms.py --models weights/weapons_v1.pt weights/weapons_v2.pt weights/weapons_v3.pt --sessions face_wall phone_hand_b

Use only sessions that NO model was trained on (default: face_wall, phone_hand_b), otherwise the
comparison is not fair. Prints frames triggered and the false-alarm rate at confidence 0.35 / 0.5 / 0.7,
and saves the table (with frame totals) to datasets/VigilenseWebcam/false_alarm_comparison.csv.
Stop app.py first so the GPU is free. It only reads files.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

BASE = Path("datasets/VigilenseWebcam")
LEVELS = (0.35, 0.5, 0.7)

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+",
               default=["weights/weapons_v1.pt", "weights/weapons_v2.pt", "weights/weapons_v3.pt"])
p.add_argument("--sessions", nargs="+", default=["face_wall", "phone_hand_b"])
a = p.parse_args()

device = 0 if torch.cuda.is_available() else "cpu"
models = [m for m in a.models if Path(m).exists()]
for m in set(a.models) - set(models):
    print("skipped (file not found):", m)
sessions = [s for s in a.sessions if (BASE / s).is_dir() and any((BASE / s).glob("*.jpg"))]
for s in set(a.sessions) - set(sessions):
    print("skipped (no frames recorded):", s)
if not models or not sessions:
    sys.exit("Nothing to compare.")

rows = []
for m in models:
    model = YOLO(m)
    for s in sessions:
        files = sorted((BASE / s).glob("*.jpg"))
        hit = {c: [0] * len(LEVELS) for c in ("gun", "knife")}
        anyhit = [0] * len(LEVELS)
        for i in range(0, len(files), 16):
            for r in model.predict([str(f) for f in files[i:i + 16]], conf=min(LEVELS), device=device, verbose=False):
                best = {}
                for c, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    name = model.names[int(c)]
                    best[name] = max(conf, best.get(name, 0.0))
                for k, lv in enumerate(LEVELS):
                    for name, conf in best.items():
                        if name in hit and conf >= lv:
                            hit[name][k] += 1
                    if any(conf >= lv for conf in best.values()):
                        anyhit[k] += 1
        n = len(files)
        for k, lv in enumerate(LEVELS):
            rows.append([Path(m).stem, s, n, lv, hit["gun"][k], hit["knife"][k], anyhit[k], round(100 * anyhit[k] / n, 1)])
        print(f"done: {Path(m).stem} on {s} ({n} frames)")

print(f"\n{'model':12s} {'session':14s} {'frames':>6s} {'conf':>5s} {'gun':>5s} {'knife':>6s} {'any':>5s} {'rate %':>7s}")
for r in rows:
    print(f"{r[0]:12s} {r[1]:14s} {r[2]:6d} {r[3]:5.2f} {r[4]:5d} {r[5]:6d} {r[6]:5d} {r[7]:7.1f}")
out = BASE / "false_alarm_comparison.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "session", "frames", "conf", "gun_frames", "knife_frames", "any_frames", "false_alarm_rate_pct"])
    w.writerows(rows)
print("\nSaved:", out)
