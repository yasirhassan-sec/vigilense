# VigiLense — Run Commands

All commands are for **Windows PowerShell**, run from the project folder `C:\vigilense`.

---

## 1) Start of every session

Run these each time you open a new PowerShell window.

```powershell
cd C:\vigilense                      # go to the project folder
.venv\Scripts\activate               # turn on the Python environment -> you see (.venv)
git pull                             # get your teammates' latest changes from GitHub
```

---

## 2) Run the VigiLense app

After starting, open **http://127.0.0.1:8000** in your browser.

```powershell
python backend\app.py --source data\cam1.mp4 data\cam2.mp4 data\cam3.mp4   # 3 cameras (videos) at once
python backend\app.py --source data\fire.mp4                              # one video
python backend\app.py --source 0                                          # laptop webcam
python backend\app.py --source 0 data\fire.mp4 data\cam3.mp4              # mix webcam and videos
python backend\app.py --source "rtsp://user:pass@IP:554/stream"           # IP camera (later)
```

The page shows a **main view** (the camera with the highest priority: fire 100, smoke 60, +2 per person),
a **camera grid** (red border = threat, blue outline = current main view) and the **incident list**.

Optional settings (add to any command above):

```powershell
python backend\app.py --source data\cam1.mp4 data\cam2.mp4 --names Gate Lobby   # camera names (default CAM-1, CAM-2, ...)
python backend\app.py --source data\fire.mp4 --conf 0.25      # lower = catches more fire, more false alarms (default 0.35)
python backend\app.py --source data\fire.mp4 --every 2        # skip loops to save CPU (default 1 = every loop)
python backend\app.py --source data\fire.mp4 --cooldown 30    # seconds before the same alert repeats on a camera (default 10)
python backend\app.py --source data\fire.mp4 --width 480      # smaller frames = faster on CPU (default 640)
python backend\app.py --source data\fire.mp4 --port 8001      # use another port if 8000 is busy
```

**Stop the app:** click in the PowerShell window and press `Ctrl+C`.

---

## 3) Quick model tests (no web page)

Opens a window with boxes. Press `q` to stop. Results are saved in `runs\detect\predict...`.

```powershell
yolo predict model=weights\fire_smoke_v1.pt source="data\fire.mp4" show=True conf=0.3   # fire/smoke model on a video
yolo predict model=weights\fire_smoke_v1.pt source=0 show=True conf=0.3                 # fire/smoke model on webcam
yolo predict model=weights\yolo26n.pt source="data\fire.mp4" show=True classes=0        # person detection only
```

---

## 4) Check incidents and evidence

```powershell
dir evidence                                  # list saved snapshots and the database
start evidence                                # open the evidence folder in File Explorer
python -c "import sqlite3; [print(r) for r in sqlite3.connect('evidence/incidents.db').execute('SELECT * FROM incidents ORDER BY id DESC LIMIT 10')]"   # last 10 incidents
Remove-Item evidence\* -Recurse               # clear all incidents and snapshots (start fresh)
```

---

## 5) Save your work to GitHub

```powershell
git status                                    # see which files changed
git add .                                     # stage all changes (big files are ignored by .gitignore)
git commit -m "Describe what you changed"     # save a snapshot with a message
git push                                      # upload to GitHub
git log --oneline -5                          # see the last 5 commits
```

If the push is rejected with "fetch first":

```powershell
git pull --rebase                             # get others' changes, put yours on top
git push                                      # then push again
```

---

## 6) Install or fix the environment (only when needed)

```powershell
pip install -r requirements.txt               # install missing packages after a git pull
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # reinstall CPU PyTorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # check PyTorch (False = CPU, normal without NVIDIA GPU)
yolo checks                                   # check Ultralytics setup
```

---

## 7) File name problems

Windows can hide a doubled extension (e.g. `fire.mp4.mp4`). Turn on **View → Show → File name extensions** in File Explorer.

```powershell
dir data, weights                             # check real file names
Rename-Item data\fire.mp4.mp4 fire.mp4        # fix a doubled extension
```

---

## 8) Training (Google Colab, not the laptop)

1. Open `notebooks\train_fire_smoke_dfire.ipynb` in Google Colab (**File → Upload notebook**).
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the cells top to bottom.
4. Download `best.pt` from Google Drive (`vigilense/runs/<run name>/weights/`) into `weights\`.

---

## Common errors

| Error | Fix |
|---|---|
| `yolo` / `python` is not recognized | Run step 1: `cd C:\vigilense` and `.venv\Scripts\activate` |
| `ERROR: cannot open source` | Wrong video name or path. Run `dir data` to see the real name |
| `fatal: not a git repository` | You are not in the project folder. Run `cd C:\vigilense` |
| `address already in use` | The app is already running, or use `--port 8001` |
| Running scripts is disabled | Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
