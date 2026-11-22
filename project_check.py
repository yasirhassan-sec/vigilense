"""
One-shot overall check of the VigiLense project. It only READS things and prints a summary.

Run from the inner vigilense folder (the one that contains backend/ and datasets/):
    python project_check.py
Then paste the whole output. It prints no passwords or API keys.
"""
import importlib
import platform
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

IMAGES = {".jpg", ".jpeg", ".png", ".webp"}

if not Path("backend/app.py").exists():
    sys.exit("Run this from the inner vigilense folder (the one that contains backend\\app.py).")


def run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"(could not run: {e})"


def n_images(folder):
    return sum(1 for p in folder.glob("*") if p.suffix.lower() in IMAGES) if folder.exists() else 0


print("== Environment ==")
print("python", platform.python_version())
for mod in ("torch", "ultralytics", "cv2", "roboflow", "fastapi"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod}: NOT AVAILABLE ({type(e).__name__})")
try:
    import torch
    ok = torch.cuda.is_available()
    print("CUDA available:", ok, f"| {torch.cuda.get_device_name(0)}" if ok else "")
except Exception:
    pass
print("opencv packages:", sorted(d.metadata["Name"] for d in distributions()
                                  if d.metadata["Name"] and "opencv" in d.metadata["Name"].lower()))

print("\n== Git ==")
print("branch:", run("git", "rev-parse", "--abbrev-ref", "HEAD"))
print("user.name:", run("git", "config", "user.name") or "(not set)")
status = run("git", "status", "--short").splitlines()
print(f"uncommitted changes: {len(status)}")
for line in status[:25]:
    print("  ", line)
print("last commits:")
print(run("git", "log", "--oneline", "-8"))
big = []
for f in run("git", "ls-files").splitlines():
    p = Path(f)
    if p.exists() and p.stat().st_size > 5 * 1024 * 1024:
        big.append(f"{f} ({p.stat().st_size // 1048576} MB)")
print("tracked files over 5 MB:", big or "none")

print("\n== Models (weights/) ==")
for p in sorted(Path("weights").glob("*")):
    print(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB")

print("\n== Data ==")
clean = Path("datasets/VigilenseWeapons/clean_v1")
if clean.exists():
    for split in ("train", "valid", "test"):
        counts = {"0": 0, "1": 0}
        for label in (clean / split / "labels").glob("*.txt"):
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0] in counts:
                    counts[parts[0]] += 1
        print(f"  clean_v1 {split:6s}: {n_images(clean / split / 'images'):5d} images | "
              f"gun boxes {counts['0']:5d} | knife boxes {counts['1']:5d}")
else:
    print("  clean_v1: NOT FOUND")
print("  raw gun_knife present:", Path("datasets/VigilenseWeapons/raw/gun_knife").exists())
web = Path("datasets/VigilenseWebcam")
if web.exists():
    for d in sorted(x for x in web.iterdir() if x.is_dir()):
        print(f"  webcam {d.name:18s}: {n_images(d)} frames")
extra = Path("datasets/VigilenseExtra/raw")
if extra.exists():
    for d in sorted(x for x in extra.iterdir() if x.is_dir()):
        print(f"  extra  {d.name:18s}: " + ", ".join(f"{s} {n_images(d / s / 'images')}" for s in ("train", "valid", "test")))

print("\n== Project features ==")
app = Path("backend/app.py").read_text(encoding="utf-8", errors="ignore")
print("  --device option in app.py:", "--device" in app)
print("  person tracking (.track) in app.py:", ".track(" in app)
code = " ".join(p.read_text(encoding="utf-8", errors="ignore").lower() for p in Path("backend").rglob("*.py"))
print("  re-id code in backend:", "reid" in code or "torchreid" in code or "re-id" in code)
dash = [p for p in Path("dashboard").rglob("*") if p.is_file() and p.name != ".gitkeep"] if Path("dashboard").exists() else []
print("  dashboard files:", len(dash))