"""
Download the Dangerous Items dataset (Omiotek 2025, Zenodo, CC BY 4.0) to THIS laptop.
1.4 GB. The download resumes after a dropped connection; just run the script again.

    python download_dangerous_items.py

Result: datasets/DangerousItems/raw/...   (5 classes: machete, knife, baseball bat, rifle, gun)
"""
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/16422779/files/Dangerous%20Items.zip?download=1"
MD5 = "f567c3b6e070b2c36578a09c5c7eb9e2"
ZIP = Path("datasets/DangerousItems/dangerous_items.zip")
RAW = Path("datasets/DangerousItems/raw")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def download():
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    part = ZIP.with_suffix(".part")
    for attempt in range(1, 31):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(URL, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = have + int(r.headers.get("Content-Length", 0))
                if have and r.status != 206:          # server ignored the resume request: start again
                    have = 0
                    part.unlink()
                with open(part, "ab") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        print(f"\r  {have / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
            print()
            part.rename(ZIP)
            return
        except Exception as e:                        # dropped connection: wait and resume
            print(f"\n  connection problem ({e.__class__.__name__}), retry {attempt}/30 in 10 s ...")
            time.sleep(10)
    sys.exit("Download failed 30 times. Run the script again later; it continues where it stopped.")


if RAW.exists() and any(RAW.rglob("*.txt")):
    sys.exit(f"{RAW} already has the dataset. Nothing to do.")
if not ZIP.exists():
    print("Downloading Dangerous Items (1.4 GB) ...")
    download()
print("Checking MD5 ...")
digest = md5(ZIP)
if digest != MD5:
    ZIP.unlink()
    sys.exit(f"Checksum mismatch ({digest}); the file was removed. Run the script again.")
print("Checksum OK. Unzipping ...")
with zipfile.ZipFile(ZIP) as z:
    z.extractall(RAW)
n_img = sum(1 for p in RAW.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
n_lbl = sum(1 for _ in RAW.rglob("*.txt"))
print(f"Done: {n_img} images, {n_lbl} label files in {RAW}")
print("You can delete", ZIP, "to save 1.4 GB.")
