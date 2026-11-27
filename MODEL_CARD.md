# VigiLense detection models: results and known limits

Last updated 2026-11-28. All numbers below come from runs in this repository (YOLO26n, Ultralytics 8.4.177,
RTX 3050 Laptop GPU). Re-run the listed commands to reproduce them.

## Models used by `backend/app.py` (defaults)

| Task | File | Training data | Built with |
|---|---|---|---|
| Persons (+ phone/remote/mouse look-alikes) | `weights/yolo26n.pt` | COCO (pretrained) | n/a |
| Fire / smoke | `weights/fire_smoke_v2.pt` | D-Fire + 498 webcam background frames (x3) | `tools/data/build_smoke_v2.py`, `train.py` |
| Gun / knife | `weights/weapons_v4.pt` | Roboflow gun_knife + Zenodo DangerousItems + Roboflow knife-detection-hgvy2 + webcam backgrounds | `tools/data/build_weapons_v4.py`, `train.py` |

`fire_smoke_v2.pt` and `weapons_v4.pt` are committed (force-added, about 5 MB each); other `*.pt` files are ignored.

## How the data was split

Images from every source are hashed together. Near-duplicates and video frames are grouped, and each whole group
goes to train, valid or test (80/10/10). The original splits were leaky: 55-67% of several original valid/test
images had a copy in train. Two webcam sessions (`face_wall`, `phone_hand_b`) are never used for training and serve
as the false-alarm test.

## Results (test split, our group split)

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| weapons_v4 (all) | 0.863 | 0.610 | 0.873 | 0.807 |
| weapons_v4 gun | 0.896 | 0.673 | 0.883 | 0.849 |
| weapons_v4 knife | 0.829 | 0.547 | 0.863 | 0.765 |
| fire_smoke_v2 (all) | 0.802 | 0.468 | 0.797 | 0.717 |
| fire_smoke_v2 fire | 0.788 | 0.427 | 0.774 | 0.697 |
| fire_smoke_v2 smoke | 0.817 | 0.508 | 0.820 | 0.736 |

weapons_v4 per source (test, mAP50): dangerous 0.875, gun_knife 0.841 (knife 0.792, recall 0.693), knife_b 0.860.
Reproduce: `python tools/eval/eval_by_source.py --model weights/weapons_v4.pt --data datasets/VigilenseWeapons/weapons_v4`.

Do not compare v3 and v4 numbers directly: they use different splits (v3 test mAP50 was 0.805 on its own split).

## False alarms on held-out webcam frames (no weapon, fire or smoke in view)

| Model | face_wall (122 frames) | phone_hand_b (73 frames) |
|---|---|---|
| fire_smoke_v1 | 5 frames flagged as smoke (conf 0.25-0.5) | 0 |
| fire_smoke_v2 | 0 | 0 |
| weapons_v3 / weapons_v4 | 0 (conf 0.35-0.7) | 0 |

195 frames is a small sample: 0 flagged frames only shows the rate is probably below about 1.5%.
Reproduce: `python tools/eval/compare_false_alarms.py`, `python tools/eval/compare_smoke_false_alarms.py`.

## Known limits

- **Small or distant knives are missed.** Knife recall at conf 0.25 is about 79.5% overall, but only about 67% for knives
  covering 0.5-2% of the image (`python tools/eval/analyze_knife_misses.py`). A knife far from a CCTV camera is the weak case.
- **Smoke on dark shadows.** fire_smoke_v2 still scores shadows in night scenes at 0.4-0.7. The app therefore requires
  smoke confidence of at least 0.7 (`--smoke-conf`, fire still uses `--conf`). This also hides faint real smoke; it has
  not been tested on a real smoke clip.
- **Small fires are weak.** A small flame scores about 0.37-0.55, near the 0.35 floor, so a smaller or more distant flame
  could be missed.
- **Stun devices are labelled "gun".** A hand-held stun gun triggers a gun alert in the demo.
- **Detection recall on D-Fire is about 0.70-0.74.** Roughly 3 in 10 fire boxes and 1 in 4 smoke boxes are missed on the
  test split.
- **Demo clips are not an independent test.** `data/knife.mp4` may share a source with dataset images. Use a freshly
  recorded clip for any claim about generalisation.
- **Webcam false-alarm test is small and from one room.** Different lighting, cameras and scenes may behave differently.

## People tracking and re-ID (prototype)

ByteTrack gives each person a local ID per camera. `backend/reid_gallery.py` gives each real person one global ID
(G1, G2, ...) by comparing appearance embeddings from `yolo26n-reid.onnx` (cosine similarity, default threshold 0.62).
Two IDs are merged when they share a camera, look alike (similarity at least threshold minus 0.07) and were never
visible together. The weapon holder is the tracked person whose box overlaps the weapon box most (at least 0.3).

Demo results (two clips, `--reid-thr 0.62`):

| Clip | IDs before the merge rule | IDs after |
|---|---|---|
| WEAPON (2 people) | 4 (G2, G4, G5, G10) | 2 (G4, G6); alerts say "held by G4" every time |
| KNIFE | 6 | 2 |

Same man across tracks scored 0.73 and different people 0.49 in `tools/tracking/test_reid.py`.

Limits: tuned on two demo clips only; the post-merge result is from a one-minute run, so repeat on a longer, independent
clip with known identities. People who stay at the left or right picture edge are skipped (cut-off crops) and keep a
local ID. The re-ID model runs on the CPU in the current setup.

## Datasets reviewed and not used

- `vape_a` (Roboflow vapes_object_detection v7): catalogue product photos on white or black backgrounds, mixing vapes,
  e-liquid bottles, coils, grinders, glass pipes and cannabis packaging. Its class names do not match the sample sheets
  (the class named `vape_tank` shows e-liquid bottles). It does not look like CCTV footage, so it is not used for
  training. Phone false alarms are handled by the look-alike filter in `app.py`.
