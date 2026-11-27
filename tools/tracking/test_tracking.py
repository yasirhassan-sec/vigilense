"""
Quick check of person tracking (ByteTrack, same settings as backend/app_tracking.py) on one video, no web app.

    python test_tracking.py data\\weapon.mp4
    python test_tracking.py data\\weapon.mp4 --every 5 --max-frames 400 --track-buffer 15

--every 5 mimics the app, which runs the detector about 5 times per second on a 25 fps video.
Prints how many frames had people and how many different IDs were handed out (far more IDs than real people
means the tracker keeps losing people), and saves runs/track_test.mp4 with the ID drawn on each person.
"""
import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--model", default="weights/yolo26n.pt")
    p.add_argument("--every", type=int, default=5, help="run the detector on every Nth video frame")
    p.add_argument("--max-frames", type=int, default=400, help="stop after this many detector runs")
    p.add_argument("--track-buffer", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    cfg.track_buffer = a.track_buffer
    tracker = BYTETracker(cfg)
    model = YOLO(a.model)
    device = 0 if torch.cuda.is_available() else "cpu"

    out_path = Path("runs") / "track_test.mp4"
    out_path.parent.mkdir(exist_ok=True)
    writer, ids_seen, frames_with_people, runs, n = None, set(), 0, 0, 0
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
        r = model.predict(frame, classes=[0], conf=0.25, device=device, verbose=False)[0]
        is_person = r.boxes.conf >= 0.4
        rows = tracker.update(r.boxes[is_person].cpu().numpy(), frame)
        runs += 1
        if len(rows):
            frames_with_people += 1
        for x1, y1, x2, y2, tid, score, cls, idx in rows:
            ids_seen.add(int(tid))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), 2)
            cv2.putText(frame, f"#{int(tid)}", (int(x1) + 4, int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer:
        writer.release()
    print(f"detector runs: {runs} | runs with a tracked person: {frames_with_people} | different IDs: {len(ids_seen)}")
    print("IDs:", sorted(ids_seen))
    print(f"Saved {out_path}  (open it and check that one person keeps one ID)")


if __name__ == "__main__":
    main()
