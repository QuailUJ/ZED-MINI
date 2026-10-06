from PyInstaller.utils.hooks import collect_all

opensim_data, opensim_binaries, opensim_hidden = collect_all("opensim")
pyzed_data, pyzed_binaries, pyzed_hidden = collect_all("pyzed")

a = Analysis(
    ["zed_studio.py"],
    pathex=[],
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
