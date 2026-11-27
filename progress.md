# VigiLense – Progress Report

**Project:** VigiLense: AI Surveillance & Incident Response System (FYP, Bahria University)
**Team:** Yasir Hassan, Hassnain Khan, Khurram Ali Khan
**Supervisor:** Dr. Syed Khawar Hussain Shah
**Last updated:** 28 Nov 2026

Hardware / stack: RTX 3050 (4 GB), Python 3.11, Ultralytics 8.4.177, YOLO26n, FastAPI, SQLite.

---

## 1. Detection models

### Fire & smoke (`weights/fire_smoke_v2.pt`)
- v2 trained on the D-Fire dataset (class 0 smoke, 1 fire) plus our own data; built with `build_smoke_v2.py`.
- Datasets are split by **groups of near-duplicate frames** (dHash + colour) so test images are not near-copies of training images.
- Two webcam sessions (`face_wall`, `phone_hand_b`) are **held out** and used only to measure false alarms.
- Result: smoke false alarms on the held-out sessions fell to **0 of 195 frames** (checked at several confidence levels with `compare_smoke_false_alarms.py`).
- The app uses a higher confidence for smoke only (`--smoke-conf`, default 0.7) to suppress dark-shadow false alarms.

### Weapons (`weights/weapons_v4.pt`, classes: gun, knife)
- v4 built with `build_weapons_v4.py` from Zenodo DangerousItems + two Roboflow sets (gun/knife and knife-detection), plus hard-negative/background frames.
- Training finished with `--resume` after it stopped at epoch 45.
- Result: **0 of 195** held-out webcam frames raised a false alarm at confidence 0.35–0.7.
- `analyze_knife_misses.py` shows where knives are missed (by source, size, group size).

### Model card
`MODEL_CARD.md` records the results, the split method, the false-alarm tables and the known limits.

## 2. Application (`backend/app.py`)
- Multi-camera FastAPI app with a priority table, a confirmation filter (3 of the last 5 detection loops), a look-alike filter (phone, remote, mouse) and SQLite incident logging.
- Defaults point to the v2 / v4 weights.
- Committed and pushed (weights force-added, commit `5398f0e`).

## 3. Tracking and re-identification (new, on `feature/tracking`)

| File | Purpose |
|---|---|
| `backend/app_tracking.py` | App with ByteTrack, one tracker per camera, people only. Incidents store the track ids and the holder of the weapon. |
| `backend/app_reid.py` | Same plus appearance re-ID (`--reid`, `--reid-thr`). |
| `backend/reid_gallery.py` | Global identity gallery shared by all cameras. |
| `test_tracking.py` | Runs a tracker on one video and saves an annotated clip. |
| `test_reid.py` | Prints the similarity between track ids. |

How it works:
1. ByteTrack gives each person a local id per camera. It uses overlap and motion only, so a person who leaves the picture and returns gets a **new id**.
2. The gallery compares what people look like (embeddings from Ultralytics `yolo26n-reid.onnx`, cosine similarity) and gives each person one global id (G1, G2, ...).
3. Two people visible together on the same camera never share an id.
4. **Merge step:** a person first seen from a bad angle can receive a second id. Two ids are merged only if they share a camera, were **never visible together**, and look alike (similarity ≥ match threshold − 0.07).
5. The weapon alert records who holds it (person box with the largest overlap with the weapon box).

Measured results (demo clips, threshold 0.62):
- Same man across tracks scored 0.73; different people scored 0.49 (`test_reid.py`).
- Before the merge step the WEAPON camera produced 4 ids for 2 people and the KNIFE camera 6.
- After it, the WEAPON camera shows 2 ids (G4 rifleman, G6 second man) and the KNIFE camera 2 ids. Every WEAPON alert says "held by G4".
- This was a one-minute run. A longer run (4–5 minutes, past the video loop) is still to be confirmed.

## 4. Known limits
- Small or far knives: recall about 67% for objects 0.5–2% of the frame.
- Stun devices are labelled as guns.
- Smoke on dark shadows can still trigger; faint real smoke has not been tested on a real clip.
- Small flames are weak (confidence about 0.37–0.55). D-Fire recall is about 0.70–0.74.
- Demo clips are not independent test data (`knife.mp4` may share a source with dataset images). `vape_a` was not used.
- Re-ID threshold was tuned on two demo clips only; other lighting or cameras may need another value.
- The re-ID model runs on CPU (ONNX Runtime has no CUDA provider here); the warning is harmless but slower.
- Crops touching the left or right picture edge are skipped, so a person who stays at the edge keeps a local id (`#n?`).
- The app logs any confirmed label; phone/vape are not in the models.

## 5. Next steps
1. Confirm the long re-ID run (WEAPON camera should still show only two G ids), then commit `feature/tracking`.
2. Review and merge tracking and re-ID into `backend/app.py` (shared file, so after team review).
3. Remaining proposal items: gait recognition, React dashboard, evidence clips and notifications, end-to-end evaluation.
4. Test smoke on a real smoke clip; record an independent knife clip for an honest test.
5. Guard so only labels in the priority table are logged.
