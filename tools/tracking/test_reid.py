"""
Does appearance matching tell people apart on your clip?  (first step of the re-ID work)

    python test_reid.py data\\weapon.mp4

It tracks people with ByteTrack (like test_tracking.py), computes an appearance embedding for every tracked person
with the ReID model Ultralytics ships (yolo26n-reid.onnx, downloaded on first use), and prints how similar the IDs
are to each other (cosine similarity, 1.0 = identical look).

What to look for: if two IDs are the SAME real person (for example #1 and #3 in data\\weapon.mp4), their similarity
should be clearly higher than between different people (#1 and #2). Send me the printed matrix.
Crops cut by the left/right picture edge are skipped, because a half-visible person gives a poor embedding.
The first run may print an 'AutoUpdate' message while it installs onnxruntime. Just run the command again.
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.utils.reid import ReID
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--model", default="weights/yolo26n.pt")
    p.add_argument("--reid", default="yolo26n-reid.onnx")
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--track-buffer", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--edge", type=int, default=6, help="skip a person whose box touches the left/right picture edge")
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    device = 0 if torch.cuda.is_available() else "cpu"
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    cfg.track_buffer = a.track_buffer
    tracker = BYTETracker(cfg)
    det = YOLO(a.model)
    reid = ReID(a.reid, device=device)

    feats = defaultdict(list)          # track id -> list of embeddings
    runs, n, skipped = 0, 0, 0
    while runs < a.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % a.every:
            continue
        h, w = frame.shape[:2]
        if w > a.width:
            frame = cv2.resize(frame, (a.width, int(h * a.width / w)))
        H, W = frame.shape[:2]
        r = det.predict(frame, classes=[0], conf=0.25, device=device, verbose=False)[0]
        rows = tracker.update(r.boxes[r.boxes.conf >= 0.4].cpu().numpy(), frame)
        runs += 1
        keep = []
        for x1, y1, x2, y2, tid, *_ in rows:
            if x1 <= a.edge or x2 >= W - a.edge or (x2 - x1) < 20 or (y2 - y1) < 40:
                skipped += 1
                continue
            keep.append((int(tid), x1, y1, x2, y2))
        if keep:
            xywh = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] for _, x1, y1, x2, y2 in keep], dtype=np.float32)
            for (tid, *_), e in zip(keep, reid(frame, xywh)):
                if e is not None:
                    feats[tid].append(np.asarray(e, dtype=np.float32).ravel())

    ids = sorted(feats)
    print(f"\ndetector runs: {runs} | person crops skipped (edge/too small): {skipped}")
    print("embeddings per ID:", {i: len(feats[i]) for i in ids})
    if len(ids) < 2:
        raise SystemExit("Need at least two IDs with usable crops to compare.")

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-9)

    mean = {i: unit(np.mean([unit(e) for e in feats[i]], axis=0)) for i in ids}
    print("\nSimilarity between IDs (cosine, 1.0 = identical):")
    print("      " + "".join(f"  #{j:<4d}" for j in ids))
    for i in ids:
        print(f"  #{i:<3d}" + "".join(f"  {float(mean[i] @ mean[j]):5.2f}" for j in ids))
    print("\nSame ID, first half of its frames vs second half (how stable one person looks to the model):")
    for i in ids:
        if len(feats[i]) >= 4:
            k = len(feats[i]) // 2
            a1 = unit(np.mean([unit(e) for e in feats[i][:k]], axis=0))
            a2 = unit(np.mean([unit(e) for e in feats[i][k:]], axis=0))
            print(f"  #{i}: {float(a1 @ a2):.2f}")
    print("\nIf #1 and #3 are the same man, their number above should be clearly higher than #1/#2 and #3/#2.")


if __name__ == "__main__":
    main()
