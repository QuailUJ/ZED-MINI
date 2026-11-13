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

from motion_data import write_trc, read_trc, repair_frames, remove_isolated_spikes, select_static_segment, trim_complete_edges, unique_directory, MARKER_ORDER
from recording_io import save_json

PROJECT_DIR = Path(__file__).absolute().parent
PIPELINE_DIR = PROJECT_DIR / "opensimPipeline"

MODEL_FILE = PIPELINE_DIR / "Models" / "LaiUhlrich2022.osim"
MARKER_SET_FILE = PIPELINE_DIR / "Models" / "LaiUhlrich2022_markers_openpose.xml"
SCALING_TEMPLATE = PIPELINE_DIR / "Scaling" / "Setup_scaling_LaiUhlrich2022_openpose.xml"
IK_TEMPLATE = PIPELINE_DIR / "IK" / "Setup_IK_openpose.xml"
FOOT_MARKERS = ("RHeel", "LHeel", "RSmallToe", "LSmallToe", "RBigToe", "LBigToe")
FOOT_REACH_TOLERANCE_M = 0.05

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
    """Center static scaling on the mean pelvis X/Z and mean heel height."""
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


def clamp_trc_to_ground(frames, clearance=0.005):
    """Use one fixed floor offset; a floor must not follow noisy foot points."""
    if not frames:
        raise ValueError("TRC 沒有可用影格")
    available = [frames[0][1][name][1] for name in FOOT_MARKERS if name in frames[0][1]]
    if not available:
        raise ValueError("TRC 缺少腳部標記，無法校正地面")
    correction = max(0.0, clearance - min(available))
    output = [(time, {
        name: (x, y + correction, z) for name, (x, y, z) in positions.items()
    }) for time, positions in frames]
    return output, {
        "method": "fixed offset from first recorded foot position",
        "adjusted_frames": len(frames) if correction > 0 else 0,
        "max_vertical_correction_m": float(correction),
        "ground_clearance_m": float(clearance),
    }


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
    if not np.isfinite(reference).all() or np.linalg.svd(reference, compute_uv=False)[1] < 1e-4:
        raise ValueError('模型骨盆標記接近共線，無法估計方向；請檢查靜態校正模型')
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


def prepare_trc(source, destination, max_gap=0.2, static=False):
    names, _, frames = read_trc(source)
    if set(names) != set(MARKER_ORDER):
        raise ValueError("TRC 必須包含目前模型使用的 20 個標記")
    frames, trimming = select_static_segment(frames, names) if static else trim_complete_edges(frames, names)
    spikes = {}
    if not static:
        frames, spikes = remove_isolated_spikes(frames, names)
    frames, quality = repair_frames(frames, names, max_gap=max_gap)
    quality.update(trimming)
    quality.update(spikes)
    quality['quality_warning'] = bool(quality['quality_warning'] or spikes.get('isolated_spike_samples', 0))
    write_trc(destination, frames, quality["effective_fps"], names)
    return names, frames, quality


def validate_motion(path, expected_times, model=None):
    # OpenSim parser checks the file; independently verify numeric data and time coverage.
    storage = osim.Storage(str(path))
    if storage.getSize() != len(expected_times):
        raise ValueError("MOT 幀數與輸入 TRC 不符")
    lines = Path(path).read_text().splitlines()
    header = lines.index("endheader")
    nonempty = [line for line in lines[header + 1:] if line.strip()]
    names = nonempty[0].split() if nonempty else []
    values = np.array([[float(v) for v in line.split()] for line in nonempty[1:]])
    if values.shape != (len(expected_times), len(names)) or len(names) < 2 or not np.isfinite(values).all():
        raise ValueError("MOT 包含無效數值")
    times = values[:, 0]
    intervals = np.diff(times)
    if np.any(intervals <= 0) or not np.allclose(times, expected_times, atol=1e-5, rtol=0):
        raise ValueError("MOT 時間與 TRC 不符")
    warnings = []
    if model is not None:
        degrees = any(line.strip().lower() == 'indegrees=yes' for line in lines[:header])
        coordinates = model.getCoordinateSet()
        state = model.initSystem()
        weights = intervals[:-1] / (times[2:] - times[:-2])
        for index, name in enumerate(names[1:], 1):
            coordinate = coordinates.get(name)
            rotational = coordinate.getMotionType() == osim.Coordinate.Rotational
            data = np.deg2rad(values[:, index]) if degrees and rotational else values[:, index]
            if coordinate.getClamped(state) and np.any((data < coordinate.getRangeMin() - 1e-5) |
                                                       (data > coordinate.getRangeMax() + 1e-5)):
                raise ValueError(f'MOT 座標超出模型範圍：{name}；請檢查 IK 與模型配對')
            if rotational and len(data) >= 3:
                predicted = data[:-2] + weights * (data[2:] - data[:-2])
                residual = np.abs(data[1:-1] - predicted)
                neighbor = np.abs(data[2:] - data[:-2])
                isolated = set(np.flatnonzero((residual > np.deg2rad(30)) & (neighbor < residual * .5)) + 1)
                jumps = np.flatnonzero((np.abs(np.diff(data)) > np.deg2rad(45)) & (intervals <= .1 + 1e-8)) + 1
                for i in sorted(isolated):
                    warnings.append({'coordinate': name, 'time': float(times[i]),
                                     'reason': '單幀角度偏離前後軌跡超過 30°'})
                for i in jumps:
                    if i not in isolated and i - 1 not in isolated:
                        warnings.append({'coordinate': name, 'time': float(times[i]),
                                         'reason': f'相鄰幀角度改變 {np.rad2deg(data[i] - data[i - 1]):+.1f}°'})
    return {"frames": len(values), "start": float(times[0]), "end": float(times[-1]),
            'continuity_warnings': warnings}


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


def apply_ground_clearance(model, path, clearance=0.005):
    """Place the initial pelvis over the origin and initial feet above Y=0.

    The same translation is retained for the full trial. A floor that moves up
    with noisy foot estimates makes the subject appear to fly.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines.index("endheader")
    names = lines[header + 1].split()
    if not {"pelvis_tx", "pelvis_ty", "pelvis_tz"}.issubset(names):
        raise ValueError("MOT 缺少地面修正欄位：pelvis_tx/pelvis_ty/pelvis_tz")
    indexes = {name: index for index, name in enumerate(names)}
    coordinates = model.getCoordinateSet()
    state = model.initSystem()
    surfaces = foot_surfaces(model)
    in_degrees = any(line.strip().lower() == "indegrees=yes" for line in lines[:header])
    output = lines[:header + 2]
    initial_offset = None
    minimum_after = float("inf")
    for line in lines[header + 2:]:
        if not line.strip():
            output.append(line)
            continue
        values = line.split()
        if len(values) != len(names) or not np.isfinite([float(v) for v in values]).all():
            raise ValueError("MOT 包含無效座標")
        for name, index in indexes.items():
            if name == "time":
                continue
            coordinate = coordinates.get(name)
            value = float(values[index])
            if in_degrees and coordinate.getMotionType() == osim.Coordinate.Rotational:
                value = np.deg2rad(value)
            coordinate.setValue(state, value, False)
        if initial_offset is None:
            model.realizePosition(state)
            lowest = foot_surface_height(state, surfaces)
            pelvis = model.getBodySet().get("pelvis").getPositionInGround(state)
            initial_offset = (-pelvis.get(0), clearance - lowest, -pelvis.get(2))
        for name, shift in zip(("pelvis_tx", "pelvis_ty", "pelvis_tz"), initial_offset):
            index = indexes[name]
            values[index] = f"{float(values[index]) + shift:.8f}"
        for name in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
            coordinates.get(name).setValue(state, float(values[indexes[name]]), False)
        model.realizePosition(state)
        minimum_after = min(minimum_after, foot_surface_height(state, surfaces))
        output.append("\t".join(values))
    if initial_offset is None:
        raise ValueError("MOT 沒有動作影格")
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "method": "fixed translation from initial scaled foot meshes",
        "initial_translation_m": [float(v) for v in initial_offset],
        "minimum_surface_y_m": float(minimum_after),
        "adjusted_frames": 0,
        "max_vertical_correction_m": 0.0,
        "ground_clearance_m": float(clearance),
    }


def foot_surfaces(model):
    """Load this project's ASCII VTP foot surfaces with model-specific scaling."""
    surfaces = []
    for name in ("talus_r", "calcn_r", "toes_r", "talus_l", "calcn_l", "toes_l"):
        body = model.getBodySet().get(name)
        count = body.getPropertyByName("attached_geometry").size()
        for index in range(count):
            mesh = osim.Mesh.safeDownCast(body.get_attached_geometry(index))
            if mesh is None:
                raise ValueError(f"足部幾何不是 Mesh：{name}")
            path = GEOMETRY_DIR / Path(mesh.get_mesh_file()).name
            data = ET.parse(path).find(".//Points/DataArray")
            if data is None or data.get("format") != "ascii":
                raise ValueError(f"無法讀取足部表面：{path.name}")
            vertices = np.fromstring(data.text or "", sep=" ").reshape(-1, 3)
            scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
            vertices = vertices * scale
            if not vertices.size or not np.isfinite(vertices).all():
                raise ValueError(f"足部表面數值無效：{path.name}")
            surfaces.append((mesh.getFrame(), vertices))
        if count == 0:
            raise ValueError(f"模型缺少足部表面：{name}")
    return surfaces


def foot_surface_height(state, surfaces):
    heights = []
    for frame, vertices in surfaces:
        transform = frame.getTransformInGround(state)
        y_axis = np.array([transform.R().get(1, i) for i in range(3)])
        heights.append(float((vertices @ y_axis).min() + transform.p().get(1)))
    return min(heights)


def validate_foot_marker_reach(model, frames):
    """Reject ankle/heel targets the scaled pin-joint chain cannot reach."""
    state = model.initSystem()
    model.realizePosition(state)
    markers = model.getMarkerSet()
    joints = model.getJointSet()
    worst = None

    def point(vec):
        return np.array([vec.get(i) for i in range(3)])

    for side, label in (("r", "右"), ("l", "左")):
        ankle_name, heel_name = (side.upper() + part for part in ("Ankle", "Heel"))
        ankle_marker, heel_marker = markers.get(ankle_name), markers.get(heel_name)
        ankle_index = joints.getIndex(f"ankle_{side}")
        subtalar_index = joints.getIndex(f"subtalar_{side}")
        if ankle_index < 0 or subtalar_index < 0:
            continue
        ankle = joints.get(ankle_index)
        subtalar = joints.get(subtalar_index)
        if any(joint.getConcreteClassName() != "PinJoint" for joint in (ankle, subtalar)):
            continue
        parent, child = ankle.getParentFrame(), ankle.getChildFrame()
        next_parent, next_child = subtalar.getParentFrame(), subtalar.getChildFrame()
        rigid_pairs = ((ankle_marker.getParentFrame(), parent), (child, next_parent),
                       (next_child, heel_marker.getParentFrame()))
        if any(a.findBaseFrame().getAbsolutePathString() != b.findBaseFrame().getAbsolutePathString()
               for a, b in rigid_pairs):
            continue

        # Pin joints share a pivot. These three rigid distances bound every pose
        # by the triangle inequality, independent of the joint angle signs.
        bound = (np.linalg.norm(point(ankle_marker.getLocationInGround(state)) - point(parent.getPositionInGround(state)))
                 + np.linalg.norm(point(child.getPositionInGround(state)) - point(next_parent.getPositionInGround(state)))
                 + np.linalg.norm(point(next_child.getPositionInGround(state)) - point(heel_marker.getLocationInGround(state))))
        distances = np.linalg.norm(
            np.array([p[ankle_name] for _, p in frames]) - np.array([p[heel_name] for _, p in frames]), axis=1)
        i = int(np.argmax(distances))
        excess = distances[i] - bound - FOOT_REACH_TOLERANCE_M
        if excess > 0 and (worst is None or excess > worst[0]):
            worst = (excess, label, frames[i][0], distances[i], bound)
    if worst:
        _, label, time, distance, bound = worst
        raise ValueError(
            f"{time:.5f} 秒的{label}踝與腳跟相距 {distance * 100:.1f} 公分，"
            f"超過縮放模型可達上限 {bound * 100:.1f} 公分加 "
            f"{FOOT_REACH_TOLERANCE_M * 100:g} 公分容差。"
            "請回看該時間的腳點、補點與模型配對，再修正資料或重錄。"
        )


def scale_trial(static_trc, work_dir):
    names, frames, quality = prepare_trc(static_trc, work_dir / "processed.trc", static=True)
    offset = compute_ground_offset(frames)
    aligned = work_dir / "aligned.trc"
    write_trc(aligned, apply_offset(frames, offset), quality["effective_fps"], names)
    model = work_dir / "model.osim"
    setup = build_scaling_xml(work_dir, aligned, model)
    if not osim.ScaleTool(setup.name).run():
        raise RuntimeError("OpenSim Scale 執行失敗")
    loaded = osim.Model(str(model))
    loaded.initSystem()
    if loaded.getMarkerSet().getSize() != len(MARKER_ORDER):
        raise ValueError("縮放模型標記數不符")
    calibration = {"version": 1, "model": "model.osim", "offset": list(offset),
                   "static_trc": str(Path(static_trc).absolute()), "quality": quality}
    save_json(work_dir / "calibration.json", calibration)
    return {"model": str(model), "calibration": str(work_dir / "calibration.json"), "quality": quality}


def ik_trial(motion_trc, calibration_path, work_dir):
    calibration_path = Path(calibration_path).absolute()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    source_model = calibration_path.parent / calibration["model"]
    model = work_dir / "model.osim"
    shutil.copyfile(source_model, model)
    calibration["model"] = model.name
    save_json(work_dir / "calibration.json", calibration)
    names, frames, quality = prepare_trc(motion_trc, work_dir / "processed.trc", max_gap=None)
    fps = quality["effective_fps"]
    aligned = work_dir / "aligned.trc"
    # Each motion has its own origin; static calibration supplies body proportions.
    first = frames[0][1]
    offset = (float(first["midHip"][0]),
              float(min(first[name][1] for name in FOOT_MARKERS)),
              float(first["midHip"][2]))
    aligned_frames, trc_ground_correction = clamp_trc_to_ground(apply_offset(frames, offset))
    write_trc(aligned, aligned_frames, fps, names)
    motion = work_dir / "motion.mot"
    scaled_model = osim.Model(str(model))
    validate_foot_marker_reach(scaled_model, aligned_frames)
    pelvis_reference = pelvis_tilt_reference(scaled_model, frames, work_dir / "pelvis_reference.mot")
    setup = build_ik_xml(work_dir, model, aligned, motion, pelvis_reference)
    if not osim.InverseKinematicsTool(setup.name).run():
        raise RuntimeError("OpenSim IK 執行失敗")
    marker_errors = marker_error_summary(work_dir)
    ground_correction = apply_ground_clearance(scaled_model, motion)
    shifts = ground_correction['initial_translation_m']
    published_frames = apply_offset(aligned_frames, [-v for v in shifts])
    write_trc(work_dir / 'processed.trc', published_frames, fps, names)
    validation = validate_motion(motion, [t for t, _ in frames], scaled_model)
    quality['foot_surface_warning'] = ground_correction['minimum_surface_y_m'] < -1e-5
    quality['quality_warning'] = bool(quality['quality_warning'] or validation['continuity_warnings'] or quality['foot_surface_warning'])
    return {"model": str(model), "motion": str(motion), "quality": quality,
            "ik_solver_accuracy": float(ET.parse(setup).findtext('.//accuracy')),
            "pelvis_orientation_reference": "RHip/LHip/midHip rigid fit; pelvis_tilt weight=0.1",
            "sagittal_correction": "none; original IK joint angles retained",
            "trc_ground_correction": trc_ground_correction,
            "motion_origin_m": list(offset),
            "ground_correction": ground_correction,
            "validation": validation, "marker_errors": marker_errors}


def execute_job(mode, source, work_dir, calibration=None):
    """Run in a child process. Output directory must be a new, empty reservation."""
    work_dir = Path(work_dir).absolute()
    source = Path(source).absolute()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError("轉換資料夾不是空的，拒絕覆蓋")
    work_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    result = {"status": "failed", "mode": mode, "source": str(source),
              'pipeline_revision': 'source-quality-20261106'}
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
        return path.absolute() if path.is_absolute() else (PROJECT_DIR / path).absolute()
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


