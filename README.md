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
| Fire & smoke model `fire_smoke_v1` (YOLO26n, D-Fire) | Done: test mAP50 0.780, precision 0.783, recall 0.716 |
| End-to-end skeleton (`backend/app.py`) | Done: video → detection → SQLite incident + snapshot → live web page |
| Multi-camera (3+ feeds) with priority-based main view | Done (first version): batched detection, camera grid, auto camera switching |
| Weapon model (gun, knife) | Planned |
| Tracking (ByteTrack/BoT-SORT) and cross-camera re-ID (OSNet) | Planned |
| React dashboard, alerts, evidence clips | Planned |

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

Model weights and videos are not in Git (too big). Get them from the shared Google Drive
(`vigilense/runs/fire_smoke_v1/weights/best.pt`) and place them like this:

```
weights/yolo26n.pt          # downloads automatically on first use
weights/fire_smoke_v1.pt    # our fire & smoke model
data/fire.mp4               # any test video
```

## Run

```powershell
python backend\app.py --source data\cam1.mp4 data\cam2.mp4 data\cam3.mp4   # 3 cameras
python backend\app.py --source data\fire.mp4                              # one video
python backend\app.py --source 0                                          # webcam
```

Open http://127.0.0.1:8000. Useful options: `--names Gate Lobby Lab`, `--conf 0.35` (fire/smoke threshold),
`--cooldown 10`, `--width 640`. See [RUN_COMMANDS.md](RUN_COMMANDS.md) for every command.

Incidents are saved in `evidence/incidents.db` with snapshots in `evidence/`.

## Training

Models are trained on Google Colab (T4 GPU) with the notebooks in `notebooks/`.
Results and checkpoints are saved to Google Drive under `vigilense/runs/`.

## Folder structure

```
backend/     FastAPI app, camera readers, detection, incident logic
models/      detection, tracking, re-ID code
dashboard/   React dashboard (planned)
notebooks/   Colab training notebooks
data/        videos and datasets (not in Git)
weights/     trained models (not in Git)
evidence/    incident snapshots, clips, database (not in Git)
```

## Datasets

- D-Fire: Venâncio et al., "An automatic fire detection system based on deep convolutional neural networks
  for low-power, resource-constrained devices", Neural Computing and Applications, 2022.

## Team workflow

- One branch per feature, merge to `main` through pull requests.
- Run `git pull` before starting work. Never commit passwords, API keys or camera RTSP credentials
  (put them in `.env`, which is ignored).
