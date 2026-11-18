
from roboflow import Roboflow
from getpass import getpass
from pathlib import Path

print("Vigilense Weapon Dataset Downloader")

api_key = getpass("Enter your Roboflow API key: ")

rf = Roboflow(api_key=api_key)

project = rf.workspace(
    "siesas-workspace"
).project(
    "gun-and-knife-detection-abefg"
)

version = project.version(1)

destination = Path("datasets/VigilenseWeapons/raw/gun_knife")
destination.mkdir(parents=True, exist_ok=True)

dataset = version.download(
    "yolo26",
    location=str(destination),
    overwrite=True
)

print("\nDataset downloaded successfully!")
print("Location:", dataset.location)
print("Absolute location:", Path(dataset.location).resolve())
print("Location exists:", Path(dataset.location).exists())

print("\nDownloaded files:")
for item in Path(dataset.location).rglob("*"):
    if item.is_file():
        print(item)