"""
VigiLense - minimal end-to-end skeleton

  video / webcam  ->  YOLO (persons + fire/smoke)  ->  SQLite incident log + snapshot
                  ->  FastAPI web page (live view, incidents)

Run from the project root (C:\\vigilense) inside the venv:
    python backend\\app.py --source data\\fire.mp4
    python backend\\app.py --source 0                  (laptop webcam)
Then open http://127.0.0.1:8000 in your browser.
"""

import argparse
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # C:\vigilense
EVIDENCE = ROOT / "evidence"
DB_PATH = ROOT / "evidence" / "incidents.db"

# ---------------------------------------------------------------- settings
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="data/test.mp4", help="video file path, RTSP url, or 0 for webcam")
parser.add_argument("--camera", default="CAM-1", help="camera name shown in incidents")
parser.add_argument("--person-model", default="weights/yolo26n.pt")
parser.add_argument("--threat-model", default="weights/fire_smoke_v1.pt")
parser.add_argument("--conf", type=float, default=0.35, help="min confidence for fire/smoke")
parser.add_argument("--every", type=int, default=3, help="run detection on every Nth frame (CPU saver)")
parser.add_argument("--cooldown", type=float, default=10.0, help="seconds before the same alert repeats")
parser.add_argument("--width", type=int, default=640, help="resize frames to this width")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()


def resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() and path.with_suffix(".pt.pt").exists():  # Windows double-extension fix
        path = path.with_suffix(".pt.pt")
    return path


# ---------------------------------------------------------------- database
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


EVIDENCE.mkdir(exist_ok=True)
with db() as con:
    con.execute(
        """CREATE TABLE IF NOT EXISTS incidents (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               time TEXT, camera TEXT, label TEXT, confidence REAL,
               persons INTEGER, snapshot TEXT)"""
    )


def log_incident(label, conf, persons, frame):
    ts = datetime.now()
    name = f"{ts:%Y%m%d_%H%M%S}_{args.camera}_{label}.jpg"
    cv2.imwrite(str(EVIDENCE / name), frame)
    with db() as con:
        con.execute(
            "INSERT INTO incidents (time, camera, label, confidence, persons, snapshot) VALUES (?,?,?,?,?,?)",
            (ts.isoformat(timespec="seconds"), args.camera, label, round(conf, 3), persons, name),
        )
    print(f"[ALERT] {ts:%H:%M:%S} {args.camera}: {label} ({conf:.2f}), persons={persons}")


# ---------------------------------------------------------------- detection loop
state = {"frame": None, "fps": 0.0, "persons": 0, "threats": [], "running": True}
lock = threading.Lock()


def detection_loop():
    person_model = YOLO(str(resolve(args.person_model)))
    threat_model = YOLO(str(resolve(args.threat_model)))
    print("Threat classes:", threat_model.names)

    src = args.source
    if src.isdigit():
        src = int(src)                      # webcam index
    elif not src.startswith(("rtsp", "http")):
        src = str(resolve(src))             # local video file
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open source {src}")
        state["running"] = False
        return

    last_alert = {}
    n, persons, threats, t0 = 0, 0, [], time.time()
    while state["running"]:
        ok, frame = cap.read()
        if not ok:
            if isinstance(src, str) and not src.startswith(("rtsp", "http")):  # loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        h, w = frame.shape[:2]
        if w > args.width:
            frame = cv2.resize(frame, (args.width, int(h * args.width / w)))

        if n % args.every == 0:
            p = person_model.predict(frame, classes=[0], conf=0.4, verbose=False)[0]
            t = threat_model.predict(frame, conf=args.conf, verbose=False)[0]
            persons = len(p.boxes)
            threats = [
                (t.names[int(c)], float(s), b.tolist())
                for c, s, b in zip(t.boxes.cls, t.boxes.conf, t.boxes.xyxy)
            ]
            person_boxes = [b.tolist() for b in p.boxes.xyxy]

            annotated = frame.copy()
            for x1, y1, x2, y2 in person_boxes:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), 2)
            for label, conf, (x1, y1, x2, y2) in threats:
                color = (0, 0, 255) if label == "fire" else (160, 160, 160)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (int(x1), max(15, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # one incident per class, then wait `cooldown` seconds before repeating
            best = {}
            for label, conf, _ in threats:
                best[label] = max(conf, best.get(label, 0))
            for label, conf in best.items():
                if time.time() - last_alert.get(label, 0) > args.cooldown:
                    last_alert[label] = time.time()
                    log_incident(label, conf, persons, annotated)

            elapsed = time.time() - t0
            with lock:
                state.update(frame=annotated, persons=persons,
                             threats=[(l, round(c, 2)) for l, c, _ in threats],
                             fps=round(n / elapsed, 1) if elapsed else 0)
        n += 1
    cap.release()


# ---------------------------------------------------------------- web app
app = FastAPI(title="VigiLense")
app.mount("/evidence", StaticFiles(directory=EVIDENCE), name="evidence")


@app.get("/api/status")
def status():
    with lock:
        return {"camera": args.camera, "fps": state["fps"], "persons": state["persons"],
                "threats": state["threats"], "running": state["running"]}


@app.get("/api/incidents")
def incidents(limit: int = 50):
    with db() as con:
        rows = con.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.get("/video")
def video():
    def gen():
        while state["running"]:
            with lock:
                f = state["frame"]
            if f is not None:
                ok, jpg = cv2.imencode(".jpg", f)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.1)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><html><head><meta charset="utf-8"><title>VigiLense</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#0f1720;color:#e5e7eb;margin:0;padding:16px}
 h1{margin:0 0 12px;font-size:22px} .grid{display:grid;grid-template-columns:2fr 1fr;gap:16px}
 .card{background:#17212c;border-radius:10px;padding:12px} img.live{width:100%;border-radius:6px}
 table{width:100%;border-collapse:collapse;font-size:13px} td,th{padding:6px;border-bottom:1px solid #2a3542;text-align:left}
 .fire{color:#f87171;font-weight:600} .smoke{color:#cbd5e1;font-weight:600} img.thumb{width:90px;border-radius:4px}
 #st{font-size:14px;margin-top:8px}
</style></head><body>
<h1>VigiLense &mdash; live monitor</h1>
<div class="grid">
 <div class="card"><img class="live" src="/video"><div id="st">loading...</div></div>
 <div class="card"><b>Incidents</b><table id="inc"><tr><th>Time</th><th>Event</th><th>Snapshot</th></tr></table></div>
</div>
<script>
async function refresh(){
  const s = await (await fetch('/api/status')).json();
  document.getElementById('st').innerText =
    `${s.camera} | ${s.fps} fps | persons: ${s.persons} | threats: ${s.threats.map(t=>t[0]+' '+t[1]).join(', ')||'none'}`;
  const rows = await (await fetch('/api/incidents?limit=20')).json();
  document.getElementById('inc').innerHTML = '<tr><th>Time</th><th>Event</th><th>Snapshot</th></tr>' +
    rows.map(r=>`<tr><td>${r.time.replace('T',' ')}<br>${r.camera}</td>
      <td class="${r.label}">${r.label} ${r.confidence}<br>persons: ${r.persons}</td>
      <td><a href="/evidence/${r.snapshot}" target="_blank"><img class="thumb" src="/evidence/${r.snapshot}"></a></td></tr>`).join('');
}
setInterval(refresh, 1500); refresh();
</script></body></html>"""


if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    print(f"Open http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
