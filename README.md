# VigiLense — AI Surveillance & Incident Response System

Final Year Project, Bachelor of Computer Engineering, Bahria University Islamabad.

**Team:** Yasir Hassan (detection) · Khurram Ali Khan (backend & tracking) · Hassnain Khan (dashboard & ops)
**Supervisor:** Dr. Syed Khawar Hussain Shah

VigiLense monitors multiple camera feeds, detects persons, weapons, fire and smoke, tracks people
across cameras, scores incidents by priority, switches to the most important camera, and logs alerts
with evidence on a web dashboard.

## Presentation video

A 6:47 project overview (1080p, narrated): [presentation/VigiLense-presentation.mp4](presentation/VigiLense-presentation.mp4).
Subtitles: [VigiLense-presentation.srt](presentation/VigiLense-presentation.srt) · Script: [NARRATION.md](presentation/NARRATION.md).

## Current status

| Part | Status |
|---|---|
| Environment (Python 3.11, PyTorch, Ultralytics YOLO26) | Done |
| Fire & smoke model `fire_smoke_v2` (YOLO26n, D-Fire + webcam backgrounds) | Done: test mAP50 0.802; 0 false alarms on 195 held-out webcam frames |
| Weapon model `weapons_v4` (gun, knife) | Done: test mAP50 0.863 (gun 0.896, knife 0.829); 0 false alarms on 195 held-out webcam frames. Small or far knives are the weak case (about 67% recall) |
| End-to-end app (`backend/app.py`) | Done: video → detection → confirm filter → SQLite incident + snapshot → live web page |
| Multi-camera (3+ feeds) with priority-based main view | Done |
| False-alarm controls | Done: confirm 3 of 5 loops, look-alike filter (phone/remote/mouse), smoke-only confidence `--smoke-conf` |
| People tracking (ByteTrack, one tracker per camera) and weapon holder | Done (prototype, `--no-track` turns it off) |
| Appearance re-ID across a camera and between cameras | Prototype (`--reid`); tuned on two demo clips only, longer tests pending |
| Gait recognition | Planned |
| React dashboard, notifications, evidence clips | Planned |
| End-to-end evaluation on independent footage | Planned (see Evaluation plan) |

Details: [MODEL_CARD.md](MODEL_CARD.md) (models, splits, false-alarm tests, limits) and
[progress.md](progress.md) (timeline from day one).

## Setup (Windows)

```powershell
git clone https://github.com/yasirhassan-sec/vigilense.git
cd vigilense
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

# 1) PyTorch first. CPU-only laptop:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#    NVIDIA GPU: use the CUDA command from https://pytorch.org/get-started/locally

# 2) Everything else
pip install -r requirements.txt
```

The two trained detection models are in the repository. Videos and datasets are not (too big):

```
weights/yolo26n.pt          # person model, downloads automatically on first use
weights/fire_smoke_v2.pt    # our fire & smoke model (in Git)
weights/weapons_v4.pt       # our gun & knife model (in Git)
yolo26n-reid.onnx           # re-ID model, downloads automatically on first use of --reid
data/fire.mp4               # any test video (not in Git)
```

## Run

```powershell
python backend\app.py --source data\cam1.mp4 data\cam2.mp4 data\cam3.mp4   # 3 cameras
python backend\app.py --source data\fire.mp4                              # one video
python backend\app.py --source 0                                          # webcam
python backend\app.py --source data\weapon.mp4 --reid                     # with appearance re-ID (G1, G2, ...)
```

Open http://127.0.0.1:8000. Useful options: `--names Gate Lobby Lab`, `--conf 0.35` (fire threshold),
`--smoke-conf 0.7` (smoke threshold), `--weapon-conf 0.5`, `--cooldown 10`, `--width 640`, `--device 0` (GPU),
`--no-track`, `--reid --reid-thr 0.62`. See [RUN_COMMANDS.md](RUN_COMMANDS.md) for every command.

Incidents are saved in `evidence/incidents.db` with snapshots in `evidence/`.

## Training

Models are trained on Google Colab (T4 GPU) with the notebooks in `notebooks/`.
Results and checkpoints are saved to Google Drive under `vigilense/runs/`.

## Folder structure

```
backend/     FastAPI app (app.py), re-ID gallery (reid_gallery.py)
tools/       data/ (download, build, split), eval/ (false-alarm and per-source tests), inspect/, tracking/ (test scripts)
tests/       unit tests (pytest)
train.py     local training script
models/      reserved for shared model code
dashboard/   React dashboard (planned)
notebooks/   training notebooks
Architecture/ diagrams
data/        videos and datasets (not in Git)
weights/     trained detection models (the two final ones are in Git)
evidence/    incident snapshots, clips, database (not in Git)
```

## Datasets

- D-Fire: Venâncio et al., "An automatic fire detection system based on deep convolutional neural networks
  for low-power, resource-constrained devices", Neural Computing and Applications, 2022.

## Team workflow

- One branch per feature, merge to `main` through pull requests (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- Run `git pull` before starting work. Never commit passwords, API keys or camera RTSP credentials
  (put them in `.env`, which is ignored).

## Evaluation plan

Done: group-based train/valid/test splits (no near-duplicate leakage), per-source results, false-alarm tests on two
held-out webcam sessions.

Still missing, in this order:
1. A freshly recorded knife clip (not from any dataset source) to measure real recall.
2. A real smoke clip (not a dark shadow) to check what `--smoke-conf 0.7` hides.
3. A longer multi-person clip with known identities to measure re-ID (identity switches and merges), and a second
   camera for cross-camera matching.
4. End-to-end numbers: alerts per hour on normal footage, time from event to alert, correct weapon holder.
