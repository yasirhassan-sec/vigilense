"""
Train a YOLO26 detector on THIS laptop (replaces the Colab notebooks).

Example (RTX 3050 Laptop, 4 GB):
    python train.py --data datasets/VigilenseWeapons/weapons_v2_bg/data.yaml --name weapons_v2
Quick pipeline check first (3 epochs on 10% of the data, a few minutes):
    python train.py --data datasets/VigilenseWeapons/weapons_v2_bg/data.yaml --name weapons_v2_quick --quick
Continue a run that stopped (power cut, sleep, Ctrl+C):
    python train.py --name weapons_v2 --resume

Results: runs/detect/<name>/ (curves, confusion matrix, weights).
At the end the best model is copied to weights/<name>.pt and the TEST-split results are saved
to runs/detect/<name>/test_results.txt.
Keep the laptop plugged in and stop Windows from sleeping while it trains.
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="path to data.yaml")
    p.add_argument("--name", required=True, help="run name, e.g. weapons_v2")
    p.add_argument("--model", default="weights/yolo26n.pt", help="starting weights")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20, help="stop early after this many epochs without improvement")
    p.add_argument("--batch", type=int, default=16, help="batch 8 used 1.33 GB on the RTX 3050 (4 GB), so 16 should fit; use 8 if you get 'out of memory'")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--quick", action="store_true", help="3 epochs on 10%% of the data to test the pipeline")
    p.add_argument("--resume", action="store_true", help="continue runs/detect/<name>/weights/last.pt")
    a = p.parse_args()

    runs = Path("runs/detect").resolve()           # absolute, so Ultralytics does not nest runs/detect/runs/detect
    run_dir = runs / a.name
    if a.resume and not (run_dir / "weights" / "last.pt").exists():
        old = Path("runs/detect/runs/detect") / a.name   # location used by the first version of this script
        if (old / "weights" / "last.pt").exists():
            run_dir = old.resolve()
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(0) if device == 0 else "CPU (training will be very slow)")

    if a.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            sys.exit(f"Nothing to resume: {last} not found")
        model = YOLO(str(last))
        model.train(resume=True)
        run_dir = Path(model.trainer.save_dir)
    else:
        if not a.data or not Path(a.data).exists():
            sys.exit(f"data.yaml not found: {a.data}")
        start = Path(a.model)
        model = YOLO(str(start) if start.exists() else "yolo26n.pt")
        model.train(
            data=a.data,
            epochs=3 if a.quick else a.epochs,
            fraction=0.1 if a.quick else 1.0,
            patience=a.patience,
            imgsz=a.imgsz,
            batch=a.batch,
            workers=a.workers,
            device=device,
            amp=True,
            project=str(runs),
            name=a.name,
            exist_ok=True,
            plots=True,
            seed=0,
        )
        run_dir = Path(model.trainer.save_dir)

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"best.pt not found in {run_dir / 'weights'}")
    Path("weights").mkdir(exist_ok=True)
    shutil.copy2(best, Path("weights") / f"{a.name}.pt")
    print("Copied best model to", Path("weights") / f"{a.name}.pt")

    data = a.data or YOLO(str(best)).overrides.get("data")
    m = YOLO(str(best)).val(data=data, split="test", imgsz=a.imgsz, batch=a.batch, device=device,
                            project=str(runs), name=f"{a.name}_test", exist_ok=True, plots=True)
    lines = [f"TEST results for {a.name}",
             f"mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f} | precision {m.box.mp:.3f} | recall {m.box.mr:.3f}"]
    for k, n in enumerate(m.names.values()):
        lines.append(f"  {n:8s} P {m.box.p[k]:.3f}  R {m.box.r[k]:.3f}  mAP50 {m.box.ap50[k]:.3f}  mAP50-95 {m.box.ap[k]:.3f}")
    text = "\n".join(lines)
    print("\n" + text)
    (run_dir / "test_results.txt").write_text(text + "\n")


if __name__ == "__main__":      # required on Windows for the data-loader workers
    main()
