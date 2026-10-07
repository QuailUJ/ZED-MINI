"""
Real-data version of the ZED -> .trc -> OpenSim Scale -> IK pipeline.

Usage:
    python run_pipeline.py static_trial.trc motion_trial.trc

static_trial.trc  - a short trial of the subject standing still (used to
                    scale the generic model to the subject's body size).
motion_trial.trc  - the actual motion you want joint angles for.

Both files are produced by capture_to_trc.py. Outputs are written to a new
conversion_* directory beside motion_trial.trc, without replacing prior results.
"""
import argparse
import os
import shutil
import json
import traceback
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

import opensim as osim

from motion_data import write_trc, read_trc, repair_frames, select_static_segment, trim_complete_edges, unique_directory, MARKER_ORDER
from recording_io import save_json

PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = PROJECT_DIR / "opensimPipeline"

MODEL_FILE = PIPELINE_DIR / "Models" / "LaiUhlrich2022.osim"
MARKER_SET_FILE = PIPELINE_DIR / "Models" / "LaiUhlrich2022_markers_openpose.xml"
SCALING_TEMPLATE = PIPELINE_DIR / "Scaling" / "Setup_scaling_LaiUhlrich2022_openpose.xml"
IK_TEMPLATE = PIPELINE_DIR / "IK" / "Setup_IK_openpose.xml"

GEOMETRY_DIR = PIPELINE_DIR / "Geometry"
osim.ModelVisualizer.addDirToGeometrySearchPaths(str(GEOMETRY_DIR))


def set_text(root, path, value):
    el = root.find(path)
    if el is None:
        raise RuntimeError(f"XML element not found: {path}")
    el.text = str(value)


def trc_time_range(trc_path):
    """Read the first/last frame times out of a .trc file's data rows."""
    with open(trc_path) as f:
        lines = f.readlines()
    times = [float(line.split("\t")[1]) for line in lines[5:] if line.strip()]
    return times[0], times[-1]


def compute_ground_offset(static_frames):
    """
    ZED reports marker positions relative to the camera's own position, not a
    ground-referenced world frame, so the subject can end up floating well
    below/beside the origin. Anchor everything to the static trial: average
    heel height across the trial -> floor (Y=0), average midHip X/Z -> world
    origin, so the model stands on the ground plane near (0, 0, 0).
    """
    heel_ys, hip_xs, hip_zs = [], [], []
    for _, positions in static_frames:
        for name in ("RHeel", "LHeel"):
            if name in positions:
                heel_ys.append(positions[name][1])
        if "midHip" in positions:
            hip_xs.append(positions["midHip"][0])
            hip_zs.append(positions["midHip"][2])
    return (
        sum(hip_xs) / len(hip_xs),
        sum(heel_ys) / len(heel_ys),
        sum(hip_zs) / len(hip_zs),
    )


def apply_offset(frames, offset):
    ox, oy, oz = offset
    return [
        (t, {m: (x - ox, y - oy, z - oz) for m, (x, y, z) in positions.items()})
        for t, positions in frames
    ]


def build_scaling_xml(work_dir, static_trc, scaled_model):
    def rel(p):
        return os.path.relpath(p, work_dir)

    t0, t1 = trc_time_range(static_trc)

    tree = ET.parse(SCALING_TEMPLATE)
    root = tree.getroot().find("ScaleTool")
    set_text(root, "GenericModelMaker/model_file", rel(MODEL_FILE))
    set_text(root, "GenericModelMaker/marker_set_file", rel(MARKER_SET_FILE))
    set_text(root, "ModelScaler/marker_file", rel(static_trc))
    set_text(root, "ModelScaler/time_range", f"{t0} {t1}")
    set_text(root, "ModelScaler/output_model_file", rel(scaled_model))
    set_text(root, "MarkerPlacer/marker_file", rel(static_trc))
    set_text(root, "MarkerPlacer/time_range", f"{t0} {t1}")
    set_text(root, "MarkerPlacer/output_motion_file", rel(work_dir / "static_markerplacer.mot"))
    set_text(root, "MarkerPlacer/output_model_file", rel(scaled_model))
    set_text(root, "MarkerPlacer/output_marker_file", rel(work_dir / "static_markers.xml"))

    xml_path = work_dir / "Setup_scaling.xml"
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    return xml_path


def build_ik_xml(work_dir, scaled_model, motion_trc, output_motion, pelvis_reference=None):
    def rel(p):
        return os.path.relpath(p, work_dir)

    t0, t1 = trc_time_range(motion_trc)

    tree = ET.parse(IK_TEMPLATE)
    root = tree.getroot().find("InverseKinematicsTool")
    set_text(root, "model_file", rel(scaled_model))
    set_text(root, "marker_file", rel(motion_trc))
    set_text(root, "time_range", f"{t0} {t1}")
    set_text(root, "output_motion_file", rel(output_motion))
    set_text(root, "results_directory", ".")
    if pelvis_reference is not None:
        set_text(root, "coordinate_file", rel(pelvis_reference))
        task = ET.SubElement(root.find("IKTaskSet/objects"), "IKCoordinateTask", name="pelvis_tilt")
        for name, value in (("apply", "true"), ("weight", "0.1"), ("value_type", "from_file")):
            ET.SubElement(task, name).text = value

    xml_path = work_dir / "Setup_IK.xml"
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    return xml_path


def pelvis_tilt_reference(model, frames, destination):
    """Estimate pelvis orientation from its three markers, independently of torso fit.

    The model uses body-fixed Z-X-Y rotations. This reference is measured per
    frame rather than pulling the lumbar coordinate toward a neutral pose.
    """
    names = ("RHip", "LHip", "midHip")
    markers = model.getMarkerSet()
    reference = np.array([[markers.get(n).get_location().get(i) for i in range(3)] for n in names])
    reference -= reference.mean(axis=0)
    # OpenSim splines need six coordinate samples, even for a shorter trial.
    # Extend only the reference endpoints; never add TRC or output MOT frames.
    reference_frames = list(frames)
    while len(reference_frames) < 6:
        reference_frames.insert(0, (reference_frames[0][0] - .01, frames[0][1]))
        reference_frames.append((reference_frames[-1][0] + .01, frames[-1][1]))
    rows = []
    for time, positions in reference_frames:
        observed = np.array([positions[n] for n in names])
        observed -= observed.mean(axis=0)
        if np.linalg.svd(observed, compute_uv=False)[1] < 1e-4:
            raise ValueError("骨盆標記接近共線，無法估計方向")
        u, _, vt = np.linalg.svd(reference.T @ observed)
        correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
        rotation = vt.T @ correction @ u.T
        tilt = np.degrees(np.arctan2(-rotation[0, 1], rotation[1, 1]))
        rows.append(f"{time:.8f}\t{tilt:.8f}")
    Path(destination).write_text("pelvis_reference\nnRows=" + str(len(rows)) +
                                "\nnColumns=2\ninDegrees=yes\nendheader\ntime\tpelvis_tilt\n" +
                                "\n".join(rows) + "\n", encoding="utf-8")
    return destination


def prepare_trc(source, destination, offset=None, max_gap=0.2, static=False):
    names, _, frames = read_trc(source)
    if set(names) != set(MARKER_ORDER):
        raise ValueError("TRC 必須包含目前模型使用的 20 個標記")
    frames, trimming = select_static_segment(frames, names) if static else trim_complete_edges(frames, names)
    frames, quality = repair_frames(frames, names, max_gap=max_gap)
    quality.update(trimming)
    if offset is not None:
        frames = apply_offset(frames, offset)
    write_trc(destination, frames, quality["effective_fps"], names)
    return frames, quality


def validate_motion(path, expected_times):
    # OpenSim parser checks the file; independently verify numeric data and time coverage.
    storage = osim.Storage(str(path))
    if storage.getSize() != len(expected_times):
        raise ValueError("MOT 幀數與輸入 TRC 不符")
    lines = Path(path).read_text().splitlines()
    header = lines.index("endheader")
    nonempty = [line for line in lines[header + 1:] if line.strip()]
    values = np.array([[float(v) for v in line.split()] for line in nonempty[1:]])
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("MOT 包含無效數值")
    if not np.allclose(values[:, 0], expected_times, atol=1e-5, rtol=0):
        raise ValueError("MOT 時間與 TRC 不符")
    return {"frames": len(values), "start": float(values[0, 0]), "end": float(values[-1, 0])}


def marker_error_summary(directory):
    files = list(Path(directory).glob("*_ik_marker_errors.sto"))
    if not files:
        raise RuntimeError("IK 沒有產生標記誤差報告")
    lines = files[0].read_text().splitlines()
    rows = [line for line in lines[lines.index("endheader") + 1:] if line.strip()]
    names = rows[0].split()
    data = np.array([[float(v) for v in row.split()] for row in rows[1:]])
    if not data.size or not np.isfinite(data).all():
        raise ValueError("IK 標記誤差報告無效")
    rms, maximum = names.index("marker_error_RMS"), names.index("marker_error_max")
    return {"mean_rms_m": float(data[:, rms].mean()), "max_error_m": float(data[:, maximum].max())}


def apply_front_facing_sagittal_correction(path):
    """Correct sagittal signs for a subject recorded facing the camera.

    ZED depth and the OpenSim model use opposite forward directions for the
    affected coordinates. Keep the solved knee motion, reverse the sagittal
    joint signs, and counter the pelvis tilt at the lumbar joint so the trunk
    remains upright in the front-facing workflow verified with real footage.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines.index("endheader")
    names = lines[header + 1].split()
    required = {"pelvis_tilt", "hip_flexion_r", "hip_flexion_l",
                "lumbar_extension", "arm_flex_r", "arm_flex_l"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"MOT 缺少方向修正欄位：{', '.join(sorted(missing))}")
    indexes = {name: names.index(name) for name in required}
    output = lines[:header + 2]
    for line in lines[header + 2:]:
        if not line.strip():
            output.append(line)
            continue
        values = line.split()
        for name in required - {"lumbar_extension"}:
            index = indexes[name]
            values[index] = f"{-float(values[index]):.8f}"
        values[indexes["lumbar_extension"]] = f"{-float(values[indexes['pelvis_tilt']]):.8f}"
        output.append("\t".join(values))
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return path


def foot_surfaces(model):
    surfaces = []
    for name in ("talus_r", "calcn_r", "toes_r", "talus_l", "calcn_l", "toes_l"):
        body = model.getBodySet().get(name)
        for index in range(body.getPropertyByName("attached_geometry").size()):
            mesh = osim.Mesh.safeDownCast(body.get_attached_geometry(index))
            path = GEOMETRY_DIR / Path(mesh.get_mesh_file()).name
            data = ET.parse(path).find(".//Points/DataArray")
            if data is None or data.get("format") != "ascii":
                raise ValueError(f"無法讀取足部表面：{path.name}")
            vertices = np.fromstring(data.text or "", sep=" ").reshape(-1, 3)
            scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
            surfaces.append((mesh.getFrame(), vertices * scale))
    if not surfaces:
        raise ValueError("模型缺少足部表面")
    return surfaces


def foot_surface_height(state, surfaces):
    heights = []
    for frame, vertices in surfaces:
        transform = frame.getTransformInGround(state)
        y_axis = np.array([transform.R().get(1, i) for i in range(3)])
        heights.append(float((vertices @ y_axis).min() + transform.p().get(1)))
    return min(heights)


def apply_fixed_ground_clearance(model, path, clearance=0.05):
    """Raise every MOT frame by one constant offset so no foot crosses the floor."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines.index("endheader")
    names = lines[header + 1].split()
    if "pelvis_ty" not in names:
        raise ValueError("MOT 缺少地面修正欄位：pelvis_ty")
    indexes = {name: index for index, name in enumerate(names)}
    rows = [line.split() for line in lines[header + 2:] if line.strip()]
    if not rows:
        raise ValueError("MOT 沒有動作影格")
    coordinates = model.getCoordinateSet()
    state = model.initSystem()
    surfaces = foot_surfaces(model)
    in_degrees = any(line.strip().lower() == "indegrees=yes" for line in lines[:header])
    minimum = float("inf")
    for values in rows:
        for name, index in indexes.items():
            if name == "time":
                continue
            coordinate = coordinates.get(name)
            value = float(values[index])
            if in_degrees and coordinate.getMotionType() == osim.Coordinate.Rotational:
                value = np.deg2rad(value)
            coordinate.setValue(state, value, False)
        model.realizePosition(state)
        minimum = min(minimum, foot_surface_height(state, surfaces))
    shift = max(0.0, clearance - minimum)
    pelvis_ty = indexes["pelvis_ty"]
    for values in rows:
        values[pelvis_ty] = f"{float(values[pelvis_ty]) + shift:.8f}"
    path.write_text("\n".join(lines[:header + 2] + ["\t".join(row) for row in rows]) + "\n", encoding="utf-8")
    return {"method": "one fixed vertical translation for the full motion",
            "vertical_shift_m": float(shift), "minimum_surface_y_m": float(minimum + shift),
            "ground_clearance_m": float(clearance)}


def shift_frames_y(frames, shift):
    return [(time, {name: (x, y + shift, z) for name, (x, y, z) in positions.items()})
            for time, positions in frames]


def scale_trial(static_trc, work_dir):
    frames, quality = prepare_trc(static_trc, work_dir / "processed.trc", static=True)
    offset = compute_ground_offset(frames)
    names, fps, _ = read_trc(work_dir / "processed.trc")
    aligned = work_dir / "aligned.trc"
    write_trc(aligned, apply_offset(frames, offset), fps, names)
    model = work_dir / "model.osim"
    setup = build_scaling_xml(work_dir, aligned, model)
    if not osim.ScaleTool(setup.name).run():
        raise RuntimeError("OpenSim Scale 執行失敗")
    loaded = osim.Model(str(model))
    loaded.initSystem()
    if loaded.getMarkerSet().getSize() != len(MARKER_ORDER):
        raise ValueError("縮放模型標記數不符")
    calibration = {"version": 1, "model": "model.osim", "offset": list(offset),
                   "static_trc": str(Path(static_trc).resolve()), "quality": quality}
    save_json(work_dir / "calibration.json", calibration)
    return {"model": str(model), "calibration": str(work_dir / "calibration.json"), "quality": quality}


def ik_trial(motion_trc, calibration_path, work_dir):
    calibration_path = Path(calibration_path).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    source_model = calibration_path.parent / calibration["model"]
    model = work_dir / "model.osim"
    shutil.copyfile(source_model, model)
    save_json(work_dir / "calibration.json", calibration)
    frames, quality = prepare_trc(motion_trc, work_dir / "processed.trc", max_gap=None)
    names, fps, _ = read_trc(work_dir / "processed.trc")
    aligned = work_dir / "aligned.trc"
    offset = calibration["offset"]
    if len(offset) != 3 or not np.isfinite(offset).all():
        raise ValueError("校正座標無效")
    aligned_frames = apply_offset(frames, offset)
    write_trc(aligned, aligned_frames, fps, names)
    motion = work_dir / "motion.mot"
    pelvis_reference = pelvis_tilt_reference(osim.Model(str(model)), frames, work_dir / "pelvis_reference.mot")
    setup = build_ik_xml(work_dir, model, aligned, motion, pelvis_reference)
    if not osim.InverseKinematicsTool(setup.name).run():
        raise RuntimeError("OpenSim IK 執行失敗")
    marker_errors = marker_error_summary(work_dir)
    ground_correction = apply_fixed_ground_clearance(osim.Model(str(model)), motion)
    write_trc(work_dir / "processed.trc", shift_frames_y(aligned_frames, ground_correction["vertical_shift_m"]), fps, names)
    validation = validate_motion(motion, [t for t, _ in frames])
    return {"model": str(model), "motion": str(motion), "quality": quality,
            "pelvis_orientation_reference": "RHip/LHip/midHip rigid fit; pelvis_tilt weight=0.1",
            "ground_correction": ground_correction,
            "validation": validation, "marker_errors": marker_errors}


def execute_job(mode, source, work_dir, calibration=None):
    """Run in a child process. Output directory must be a new, empty reservation."""
    work_dir = Path(work_dir).resolve()
    source = Path(source).resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError("轉換資料夾不是空的，拒絕覆蓋")
    work_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    result = {"status": "failed", "mode": mode, "source": str(source)}
    try:
        os.chdir(work_dir)
        osim.Logger.removeFileSink()
        osim.Logger.addFileSink(str(work_dir / "opensim.log"))
        if mode == "scale":
            result.update(scale_trial(source, work_dir))
        elif mode == "ik":
            if calibration is None:
                raise ValueError("需要靜態校正檔")
            result.update(ik_trial(source, calibration, work_dir))
        else:
            raise ValueError(f"未知轉換模式：{mode}")
        result["status"] = "complete"
    except Exception as exc:
        result["error"] = str(exc)
        (work_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        osim.Logger.removeFileSink()
        os.chdir(cwd)
        save_json(work_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Scale static TRC and run IK without overwriting prior results")
    parser.add_argument("static_trc", type=Path, nargs="?")
    parser.add_argument("motion_trc", type=Path, nargs="?")
    parser.add_argument("--mode", choices=["scale", "ik"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    def resolve(path):
        return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()
    if args.mode:
        if args.source is None or args.output_dir is None:
            parser.error("--mode requires --source and --output-dir")
        result = execute_job(args.mode, resolve(args.source), resolve(args.output_dir),
                             resolve(args.calibration) if args.calibration else None)
    else:
        if args.static_trc is None or args.motion_trc is None:
            parser.error("Provide static_trc and motion_trc")
        static, motion = resolve(args.static_trc), resolve(args.motion_trc)
        directory = unique_directory(motion.parent, "conversion")
        scale_dir = directory / "static"
        result = execute_job("scale", static, scale_dir)
        if result["status"] == "complete":
            result = execute_job("ik", motion, directory / "motion", scale_dir / "calibration.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    main()


