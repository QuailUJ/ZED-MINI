"""Extract the official ZED 5.5 Python wrapper for the PyInstaller build."""
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / ".analysis" / "pyzed-5.5-cp311-cp311-win_amd64.whl"
TARGET = ROOT / ".analysis" / "pyzed55"

if not WHEEL.is_file():
    raise SystemExit(f"Missing official wheel: {WHEEL}")

TARGET.mkdir(parents=True, exist_ok=True)
with ZipFile(WHEEL) as archive:
    archive.extractall(TARGET)
print(TARGET)
