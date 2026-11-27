"""
VigiLense - multi-camera skeleton

  camera 1 ┐                                   ┌─> SQLite incident log + snapshots
  camera 2 ├─> YOLO (persons + fire/smoke + weapons) ────┼─> priority score per camera
  camera 3 ┘   one batched pass for all cams   └─> web page: camera grid + auto "main view"

Run from the project root (C:\\vigilense) inside the venv:
    python backend\\app.py --source data\\cam1.mp4 data\\cam2.mp4 data\\cam3.mp4
    python backend\\app.py --source data\\fire.mp4            (one camera still works)
    python backend\\app.py --source 0 data\\fire.mp4           (webcam + a video)
Then open http://127.0.0.1:8000

False-alarm controls (on by default):
  --confirm 3 --window 5   a threat must be seen in 3 of the last 5 loops before it alerts (--confirm 1 = off)
  --no-lookalike           turn off the phone/remote/mouse filter for gun/knife boxes
  --smoke-conf 0.7         smoke needs this confidence (fire uses --conf); lower it to see fainter smoke
"""

import argparse
import sqlite3
from collections import defaultdict, deque
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # C:\vigilense
EVIDENCE = ROOT / "evidence"
DB_PATH = EVIDENCE / "incidents.db"

# ---------------------------------------------------------------- settings
parser = argparse.ArgumentParser()
parser.add_argument("--source", nargs="+", default=["data/fire.mp4"],
                    help="one or more sources: video files, RTSP urls, or webcam numbers (0, 1)")
parser.add_argument("--names", nargs="+", help="camera names, same order as --source (default CAM-1, CAM-2, ...)")
parser.add_argument("--person-model", default="weights/yolo26n.pt")
parser.add_argument("--threat-model", default="weights/fire_smoke_v2.pt")
parser.add_argument("--weapon-model", default="weights/weapons_v4.pt", help="gun/knife model (skipped if the file is missing)")
parser.add_argument("--conf", type=float, default=0.35, help="min confidence for fire/smoke")
parser.add_argument("--weapon-conf", type=float, default=0.5, help="min confidence for gun/knife")
parser.add_argument("--smoke-conf", type=float, default=0.7,
                    help="min confidence for smoke only (dark shadows score 0.4-0.7 as smoke); fire still uses --conf")
parser.add_argument("--every", type=int, default=1, help="skip frames: process each camera every Nth loop")
parser.add_argument("--cooldown", type=float, default=10.0, help="seconds before the same alert repeats on a camera")
parser.add_argument("--width", type=int, default=640, help="resize frames to this width")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--confirm", type=int, default=3,
                    help="a threat must appear in this many of the last --window loops before it alerts (1 = off)")
parser.add_argument("--window", type=int, default=5, help="number of recent loops used by --confirm")
parser.add_argument("--no-lookalike", action="store_true",
                    help="do not drop gun/knife boxes that sit on a phone/remote/mouse found by the person model")
parser.add_argument("--lookalike-overlap", type=float, default=0.5,
                    help="drop a gun/knife box when this share of the smaller box overlaps a look-alike object")
parser.add_argument("--device", default=None,
                    help="'0' for the first GPU or 'cpu' (default: GPU if available, otherwise CPU)")
args = parser.parse_args()
if not 1 <= args.confirm <= args.window:
    parser.error("--confirm must be between 1 and --window")

# Use the GPU when there is one, otherwise the CPU, so the app also runs on laptops without NVIDIA.
DEVICE = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
print("Inference device:", DEVICE)

NAMES = args.names or [f"CAM-{i + 1}" for i in range(len(args.source))]
if len(NAMES) != len(args.source):
    parser.error("--names must have the same count as --source")

# How important each event is. Highest-scoring camera becomes the main view.
PRIORITY = {"gun": 120, "knife": 90, "fire": 100, "smoke": 60}
PER_PERSON = 2  # small bonus per visible person
WEAPONS = {"gun", "knife"}

# COCO objects (person model) that the weapon model confuses with guns/knives.
# Scissors are left out on purpose: they are sharp and should still raise "knife".
LOOKALIKE = {64: "mouse", 65: "remote", 67: "cell phone"}


def overlap(a, b):
    """Intersection divided by the area of the smaller box (1.0 = one box lies inside the other)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def one_label_per_object(weapons, thr=0.6):
    """If gun and knife boxes cover the same object, keep only the more confident one."""
    kept = []
    for w in sorted(weapons, key=lambda x: -x[1]):
        if all(iou(w[2], k[2]) < thr for k in kept):
            kept.append(w)
    return kept


def resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() and Path(str(path) + Path(str(path)).suffix).exists():  # Windows double-extension fix
        path = Path(str(path) + path.suffix)
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


def log_incident(camera, label, conf, persons, frame):
    ts = datetime.now()
    name = f"{ts:%Y%m%d_%H%M%S}_{camera}_{label}.jpg"
    cv2.imwrite(str(EVIDENCE / name), frame)
    with db() as con:
        con.execute(
            "INSERT INTO incidents (time, camera, label, confidence, persons, snapshot) VALUES (?,?,?,?,?,?)",
            (ts.isoformat(timespec="seconds"), camera, label, round(conf, 3), persons, name),
        )
    print(f"[ALERT] {ts:%H:%M:%S} {camera}: {label} ({conf:.2f}), persons={persons}")


# ---------------------------------------------------------------- cameras
class Camera:
    """Reads one source in its own thread and always keeps only the newest frame."""

    def __init__(self, idx, name, source):
        self.idx, self.name = idx, name
        self.src = int(source) if source.isdigit() else (
            source if source.startswith(("rtsp", "http")) else str(resolve(source)))
        self.is_file = isinstance(self.src, str) and not self.src.startswith(("rtsp", "http"))
        self.raw = None            # newest frame from the source
        self.annotated = None      # newest frame with boxes drawn
        self.persons = 0
        self.threats = []          # [(label, conf), ...]
        self.score = 0
        self.fps = 0.0
        self.online = False
        self.last_alert = {}
        self.history = defaultdict(lambda: deque(maxlen=args.window))   # label -> recent loops seen (True/False)
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            print(f"ERROR: {self.name} cannot open source {self.src}")
            return
        self.online = True
        delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25) if self.is_file else 0
        while True:
            ok, frame = cap.read()
            if not ok:
                if self.is_file:                       # loop video files
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.online = False                    # camera dropped: retry every 2 s
                time.sleep(2)
                cap = cv2.VideoCapture(self.src)
                self.online = cap.isOpened()
                continue
            h, w = frame.shape[:2]
            if w > args.width:
                frame = cv2.resize(frame, (args.width, int(h * args.width / w)))
            with self.lock:
                self.raw = frame
            if delay:
                time.sleep(delay)                      # play files at real speed

    def latest(self):
        with self.lock:
            return None if self.raw is None else self.raw.copy()


cams = [Camera(i, n, s) for i, (n, s) in enumerate(zip(NAMES, args.source))]
state = {"main": 0, "loop_fps": 0.0}


# ---------------------------------------------------------------- detection loop
def draw(frame, person_boxes, threats, name, score, confirmed=(), dropped=(), lookalikes=()):
    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), 2)
    for obj, conf, (x1, y1, x2, y2) in lookalikes:                 # phone / remote / mouse: thin grey
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (170, 170, 170), 1)
        cv2.putText(frame, f"{obj} {conf:.2f}", (int(x1), min(frame.shape[0] - 5, int(y2) + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
    for label, conf, (x1, y1, x2, y2) in dropped:                  # weapon box ignored because it is a phone etc.
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (120, 120, 120), 1)
        cv2.putText(frame, f"{label}? {conf:.2f} ignored", (int(x1), max(15, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    for label, conf, (x1, y1, x2, y2) in threats:
        color = {"fire": (0, 0, 255), "gun": (0, 0, 200), "knife": (0, 120, 255)}.get(label, (160, 160, 160))
        ok = label in confirmed
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2 if ok else 1)
        cv2.putText(frame, f"{label} {conf:.2f}" + ("" if ok else " (checking)"), (int(x1), max(15, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if ok else 1)
    cv2.putText(frame, f"{name}  score {score}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def detection_loop():
    person_model = YOLO(str(resolve(args.person_model)))
    threat_model = YOLO(str(resolve(args.threat_model)))
    weapon_path = resolve(args.weapon_model)
    weapon_model = YOLO(str(weapon_path)) if weapon_path.exists() else None
    if weapon_model is None:
        print(f"WARNING: weapon model not found ({weapon_path}); running without gun/knife detection")
    print("Threat classes:", threat_model.names, "| weapons:", weapon_model.names if weapon_model else "off",
          "| cameras:", ", ".join(NAMES))
    print(f"Weapon model: {weapon_path.name if weapon_model else 'off'} | confirm {args.confirm} of {args.window} loops"
          f" | look-alike filter {'off' if args.no_lookalike else 'on (phone, remote, mouse)'}")
    person_classes = [0] if args.no_lookalike else [0, *LOOKALIKE]

    n, t0 = 0, time.time()
    while True:
        batch = [(c, c.latest()) for c in cams]
        batch = [(c, f) for c, f in batch if f is not None]
        if not batch or n % args.every:
            n += 1
            time.sleep(0.01)
            continue
        frames = [f for _, f in batch]

        p_res = person_model.predict(
            frames, classes=person_classes, conf=0.25,
            device=DEVICE, verbose=False
        )

        t_res = threat_model.predict(
            frames, conf=args.conf,
            device=DEVICE, verbose=False
        )

        w_res = weapon_model.predict(
            frames, conf=args.weapon_conf,
            device=DEVICE, verbose=False
        ) if weapon_model else [None] * len(batch)
        for (cam, frame), p, t, w in zip(batch, p_res, t_res, w_res):
            persons, lookalikes = [], []
            for c, s, b in zip(p.boxes.cls, p.boxes.conf, p.boxes.xyxy):
                c, s, b = int(c), float(s), b.tolist()
                if c == 0 and s >= 0.4:
                    persons.append(b)
                elif c in LOOKALIKE:
                    lookalikes.append((LOOKALIKE[c], s, b))
            threats = [(t.names[int(c)], float(s), b.tolist())
                       for c, s, b in zip(t.boxes.cls, t.boxes.conf, t.boxes.xyxy)
                       if not (t.names[int(c)] == "smoke" and float(s) < args.smoke_conf)]
            dropped = []
            if w is not None:
                weapons = one_label_per_object([(w.names[int(c)], float(s), b.tolist())
                                                for c, s, b in zip(w.boxes.cls, w.boxes.conf, w.boxes.xyxy)])
                for wb in weapons:
                    if any(overlap(wb[2], lb[2]) >= args.lookalike_overlap for lb in lookalikes):
                        dropped.append(wb)                             # it is a phone/remote/mouse, not a weapon
                    else:
                        threats.append(wb)
            best = {}
            for label, conf, _ in threats:
                best[label] = max(conf, best.get(label, 0))

            # multi-frame confirmation: a class counts only if seen in --confirm of the last --window loops
            for label in set(cam.history) | set(best):
                cam.history[label].append(label in best)
            confirmed = {l: c for l, c in best.items() if sum(cam.history[l]) >= args.confirm}

            score = max([PRIORITY.get(l, 0) for l in confirmed] + [0]) + PER_PERSON * len(persons)
            annotated = draw(frame, persons, threats, cam.name, score, confirmed, dropped, lookalikes)
            for label, conf in confirmed.items():                  # one alert per class per cooldown
                if time.time() - cam.last_alert.get(label, 0) > args.cooldown:
                    cam.last_alert[label] = time.time()
                    log_incident(cam.name, label, conf, len(persons), annotated)

            with cam.lock:
                cam.annotated, cam.persons, cam.score = annotated, len(persons), score
                cam.threats = [(l, round(c, 2)) for l, c in confirmed.items()]

        # dynamic camera switching: show the highest-priority camera, keep the current one on ties
        top = max(cams, key=lambda c: c.score)
        if top.score > cams[state["main"]].score:
            state["main"] = top.idx

        n += 1
        state["loop_fps"] = round(n / (time.time() - t0), 1)


# ---------------------------------------------------------------- web app
app = FastAPI(title="VigiLense")
app.mount("/evidence", StaticFiles(directory=EVIDENCE), name="evidence")


@app.get("/api/status")
def status():
    return {
        "main": state["main"],
        "loop_fps": state["loop_fps"],
        "cameras": [{"id": c.idx, "name": c.name, "online": c.online, "persons": c.persons,
                     "threats": c.threats, "score": c.score} for c in cams],
    }


@app.get("/api/incidents")
def incidents(limit: int = 50):
    with db() as con:
        rows = con.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return JSONResponse([dict(r) for r in rows])


def mjpeg(get_cam):
    while True:
        cam = get_cam()
        with cam.lock:
            f = cam.annotated if cam.annotated is not None else cam.raw
        if f is not None:
            ok, jpg = cv2.imencode(".jpg", f)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.1)


@app.get("/video/main")
def video_main():
    return StreamingResponse(mjpeg(lambda: cams[state["main"]]),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video/{cam_id}")
def video(cam_id: int):
    if not 0 <= cam_id < len(cams):
        raise HTTPException(404, "no such camera")
    return StreamingResponse(mjpeg(lambda: cams[cam_id]), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
def index():
    tiles = "".join(
        f'<div class="tile" id="t{c.idx}"><img src="/video/{c.idx}"><div class="cap" id="c{c.idx}">{c.name}</div></div>'
        for c in cams)
    return """<!doctype html><html><head><meta charset="utf-8"><title>VigiLense</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#0f1720;color:#e5e7eb;margin:0;padding:16px}
 h1{margin:0 0 12px;font-size:22px} .layout{display:grid;grid-template-columns:2fr 1fr;gap:16px}
 .card{background:#17212c;border-radius:10px;padding:12px}
 img{width:100%;border-radius:6px;display:block;background:#000;object-fit:contain}
 img.mainview{height:52vh} .tile img{aspect-ratio:16/9}
 #mainCap{font-size:15px;margin-top:8px;font-weight:600}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}
 .tile{border:3px solid transparent;border-radius:8px;padding:2px} .tile.alert{border-color:#ef4444}
 .tile.main{outline:2px solid #38bdf8} .cap{font-size:12px;padding:4px 2px}
 table{width:100%;border-collapse:collapse;font-size:13px} td,th{padding:6px;border-bottom:1px solid #2a3542;text-align:left}
 .gun,.knife{color:#fb923c;font-weight:700} .fire{color:#f87171;font-weight:600} .smoke{color:#cbd5e1;font-weight:600} img.thumb{width:90px;height:60px}
 #inc{display:block;max-height:80vh;overflow-y:auto}
</style></head><body>
<h1>VigiLense &mdash; live monitor</h1>
<div class="layout">
 <div class="card">
  <img class="mainview" src="/video/main"><div id="mainCap">Main view</div>
  <div class="grid">""" + tiles + """</div>
 </div>
 <div class="card"><b>Incidents</b><table id="inc"></table></div>
</div>
<script>
async function refresh(){
  const s = await (await fetch('/api/status')).json();
  const m = s.cameras[s.main];
  document.getElementById('mainCap').innerText =
    `Main view: ${m.name} (priority ${m.score}) | ${s.loop_fps} detection loops/s`;
  for (const c of s.cameras){
    const t = document.getElementById('t'+c.id);
    t.className = 'tile' + (c.threats.length ? ' alert' : '') + (c.id === s.main ? ' main' : '');
    document.getElementById('c'+c.id).innerText =
      `${c.name} ${c.online ? '' : '(offline) '}| persons ${c.persons} | ${c.threats.map(x=>x[0]+' '+x[1]).join(', ') || 'clear'}`;
  }
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