import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

project_root = Path(SPECPATH)
pyzed55 = project_root / ".analysis" / "pyzed55"
if not (pyzed55 / "pyzed-5.5.dist-info" / "METADATA").is_file():
    raise SystemExit("Missing .analysis/pyzed55. Run tools/prepare_zed55_build.py first.")
sys.path.insert(0, str(pyzed55))

opensim_data, opensim_binaries, opensim_hidden = collect_all("opensim")
pyzed_data, pyzed_binaries, pyzed_hidden = collect_all("pyzed")

a = Analysis(
    ["zed_studio.py"],
    pathex=[str(pyzed55)],
    binaries=opensim_binaries + pyzed_binaries,
    datas=opensim_data + pyzed_data + [
        ("opensimPipeline", "opensimPipeline"),
        ("voice_commands.ps1", "."),
    ],
    hiddenimports=opensim_hidden + pyzed_hidden + ["capture_to_trc", "run_pipeline"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Camera, AI, CUDA and TensorRT DLLs come from the user's installed ZED SDK.
# Keeping them out of the package prevents a 5.4/5.5 native-library mismatch.
zed_sdk_root = Path(os.environ.get("ZED_SDK_ROOT_DIR", r"C:\Program Files (x86)\ZED SDK"))
a.binaries = [
    item for item in a.binaries
    if not Path(item[1]).is_relative_to(zed_sdk_root)
    and Path(item[0]).name.lower() not in {"sl_zed64.dll", "sl_ai64.dll"}
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ZED_MINI_Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

safe_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ZED_MINI_Studio_SafeMode",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    safe_exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ZED_MINI_Studio",
)
