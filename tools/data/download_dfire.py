"""
Download the D-Fire fire/smoke dataset (images + YOLO labels) from the authors' Google Drive folder.

    pip install gdown
    python download_dfire.py

Saves to datasets/VigilenseExtra/raw/dfire. Google Drive sometimes blocks large downloads
("quota exceeded" / "too many users"). If that happens the script tells you what to do by hand.
Source and licence: https://github.com/gaiasd/DFireDataset
"""
import sys
import zipfile
from pathlib import Path

OUT = Path("datasets/VigilenseExtra/raw/dfire")
FOLDER_ID = "1DWgsQLVgkkLM8m-VcugHNpD5WYDbjYp5"   # "D-Fire dataset (only images and labels)" from the authors' README


def main():
    try:
        import gdown
    except ImportError:
        sys.exit("Run first:  pip install gdown")
    if OUT.exists() and any(OUT.rglob("*.jpg")):
        sys.exit(f"{OUT} already has images - nothing to do.")
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(id=FOLDER_ID, output=str(OUT), quiet=False, use_cookies=False)
    except Exception as e:                                          # noqa: BLE001
        print("\nDownload failed:", e)
    for z in list(OUT.rglob("*.zip")):
        print("Unzipping", z.name)
        with zipfile.ZipFile(z) as f:
            f.extractall(z.parent)
    n = sum(1 for _ in OUT.rglob("*.jpg"))
    print(f"\nImages found in {OUT}: {n}")
    if n < 15000:
        print("Looks incomplete (D-Fire has about 21,000 images). Download it by hand:\n"
              "  1. Open https://github.com/gaiasd/DFireDataset and click 'D-Fire dataset (only images and labels)'.\n"
              "  2. Download the folder in your browser, unzip it into", OUT.resolve(), "\n"
              "  3. The folder must contain 'images' and 'labels' sub-folders (any depth).")
    else:
        print("OK - next: python build_smoke_v2.py --check")


if __name__ == "__main__":
    main()
