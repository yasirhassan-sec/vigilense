"""
Record clean webcam frames (no boxes drawn) for the VigiLense training data.

Examples (run from the inner vigilense folder, with app.py STOPPED):
    python capture_frames.py face_wall
    python capture_frames.py empty_room --count 60
    python capture_frames.py phone_2ft --every 0.7 --count 150

Frames go to datasets/VigilenseWebcam/<session>/ (this folder is ignored by Git).
Press q in the preview window to stop early.
"""
import argparse
import time
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("session", help="short name for this recording, e.g. face_wall, empty_room, phone_2ft")
parser.add_argument("--cam", type=int, default=0, help="camera number (default 0)")
parser.add_argument("--every", type=float, default=1.0, help="seconds between saved frames")
parser.add_argument("--count", type=int, default=120, help="stop after this many frames")
args = parser.parse_args()

out = Path("datasets/VigilenseWebcam") / args.session
out.mkdir(parents=True, exist_ok=True)
already = len(list(out.glob("*.jpg")))      # numbering continues, so old frames are never overwritten

cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    raise SystemExit("Cannot open the camera. Stop app.py (Ctrl+C) or any other program using it, then try again.")

print(f"Recording '{args.session}': {args.count} frames, one every {args.every} s. Folder: {out}")
saved, last = 0, 0.0
while saved < args.count:
    ok, frame = cap.read()
    if not ok:
        print("Camera stopped sending frames.")
        break

    now = time.time()
    if now - last >= args.every:
        last = now
        saved += 1
        cv2.imwrite(str(out / f"{args.session}_{already + saved:04d}.jpg"), frame)   # clean frame, no text

    preview = frame.copy()                                   # the text is drawn on a copy only
    cv2.putText(preview, f"{args.session}: {saved}/{args.count} saved (q to stop)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("capture", preview)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {saved} frames to {out}")