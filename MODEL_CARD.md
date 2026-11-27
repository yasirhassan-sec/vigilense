# VigiLense detection models: results and known limits

Last updated 2026-11-27. All numbers below come from runs in this repository (YOLO26n, Ultralytics 8.4.177,
RTX 3050 Laptop GPU). Re-run the listed commands to reproduce them.

## Models used by `backend/app.py` (defaults)

| Task | File | Training data | Built with |
|---|---|---|---|
| Persons (+ phone/remote/mouse look-alikes) | `weights/yolo26n.pt` | COCO (pretrained) | n/a |
| Fire / smoke | `weights/fire_smoke_v2.pt` | D-Fire + 498 webcam background frames (x3) | `build_smoke_v2.py`, `train.py` |
| Gun / knife | `weights/weapons_v4.pt` | Roboflow gun_knife + Zenodo DangerousItems + Roboflow knife-detection-hgvy2 + webcam backgrounds | `build_weapons_v4.py`, `train.py` |

`*.pt` files are not in git (`.gitignore`). Share them separately.

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
Reproduce: `python eval_by_source.py --model weights/weapons_v4.pt --data datasets/VigilenseWeapons/weapons_v4`.

Do not compare v3 and v4 numbers directly: they use different splits (v3 test mAP50 was 0.805 on its own split).

## False alarms on held-out webcam frames (no weapon, fire or smoke in view)

| Model | face_wall (122 frames) | phone_hand_b (73 frames) |
|---|---|---|
| fire_smoke_v1 | 5 frames flagged as smoke (conf 0.25-0.5) | 0 |
| fire_smoke_v2 | 0 | 0 |
| weapons_v3 / weapons_v4 | 0 (conf 0.35-0.7) | 0 |

195 frames is a small sample: 0 flagged frames only shows the rate is probably below about 1.5%.
Reproduce: `python compare_false_alarms.py`, `python compare_smoke_false_alarms.py`.

## Known limits

- **Small or distant knives are missed.** Knife recall at conf 0.25 is about 79.5% overall, but only about 67% for knives
  covering 0.5-2% of the image (`python analyze_knife_misses.py`). A knife far from a CCTV camera is the weak case.
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

## Datasets reviewed and not used

- `vape_a` (Roboflow vapes_object_detection v7): catalogue product photos on white or black backgrounds, mixing vapes,
  e-liquid bottles, coils, grinders, glass pipes and cannabis packaging. Its class names do not match the sample sheets
  (the class named `vape_tank` shows e-liquid bottles). It does not look like CCTV footage, so it is not used for
  training. Phone false alarms are handled by the look-alike filter in `app.py`.
