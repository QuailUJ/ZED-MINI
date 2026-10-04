"""Camera-independent marker data, validation and short-gap repair."""
from pathlib import Path
import numpy as np

# ZED BODY_38_PARTS name -> OpenSim marker name (LaiUhlrich2022_markers_openpose.xml).
# Everything not listed here (spine joints, face, clavicles, sparse finger points)
# has no counterpart in that marker set and is intentionally dropped.
KEYPOINT_TO_MARKER = {
    "PELVIS": "midHip",
    "NECK": "Neck",
    "LEFT_SHOULDER": "LShoulder",
    "RIGHT_SHOULDER": "RShoulder",
    "LEFT_ELBOW": "LElbow",
    "RIGHT_ELBOW": "RElbow",
    "LEFT_WRIST": "LWrist",
    "RIGHT_WRIST": "RWrist",
    "LEFT_HIP": "LHip",
    "RIGHT_HIP": "RHip",
    "LEFT_KNEE": "LKnee",
    "RIGHT_KNEE": "RKnee",
    "LEFT_ANKLE": "LAnkle",
    "RIGHT_ANKLE": "RAnkle",
    "LEFT_HEEL": "LHeel",
    "RIGHT_HEEL": "RHeel",
    "LEFT_BIG_TOE": "LBigToe",
    "RIGHT_BIG_TOE": "RBigToe",
    "LEFT_SMALL_TOE": "LSmallToe",
    "RIGHT_SMALL_TOE": "RSmallToe",
}

MARKER_ORDER = [
    "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "midHip",
    "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel", "LHeel",
    "RSmallToe", "LSmallToe", "RBigToe", "LBigToe",
    "RElbow", "LElbow", "RWrist", "LWrist",
]

def write_trc(path, frames, fps, marker_order, camera_rate=None):
    n_frames = len(frames)
    n_markers = len(marker_order)
    with open(path, "w", newline="") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{Path(path).name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{camera_rate if camera_rate is not None else fps}\t{n_frames}\t{n_markers}\tm\t{fps}\t1\t{n_frames}\n")

        marker_header = "".join(f"{m}\t\t\t" for m in marker_order)[:-1]  # drop 1 trailing tab -> 2 blank cells after the last marker, matching every data row's column count
        f.write("Frame#\tTime\t" + marker_header + "\n")
        f.write("\t\t" + "".join(f"X{i+1}\tY{i+1}\tZ{i+1}\t" for i in range(n_markers)).rstrip("\t") + "\n")

        for i, (t, positions) in enumerate(frames):
            row = [str(i + 1), f"{t:.5f}"]
            for m in marker_order:
                x, y, z = positions.get(m, (float("nan"), float("nan"), float("nan")))
                row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
            f.write("\t".join(row) + "\n")


def read_trc(trc_path):
    with open(trc_path) as f:
        lines = f.readlines()
    if len(lines) < 5:
        raise ValueError("TRC 標頭不完整")
    if lines[2].split("\t")[4].strip() != "m":
        raise ValueError("目前僅接受單位為 m 的 TRC")
    fps = float(lines[2].split("\t")[0])
    n_markers = int(lines[2].split("\t")[3])
    name_tokens = [t for t in lines[3].rstrip("\n").split("\t") if t][2:]
    marker_order = name_tokens[:n_markers]
    if len(marker_order) != n_markers or len(set(marker_order)) != n_markers:
        raise ValueError("TRC 標記名稱不完整或重複")

    frames = []
    for line in lines[5:]:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        t = float(parts[1])
        vals = parts[2:]
        if len(vals) != n_markers * 3:
            raise ValueError("TRC 座標欄位數不符")
        positions = {
            m: (float(vals[i * 3]), float(vals[i * 3 + 1]), float(vals[i * 3 + 2]))
            for i, m in enumerate(marker_order)
        }
        frames.append((t, positions))
    return marker_order, fps, frames


def trim_complete_edges(frames, markers=MARKER_ORDER):
    """Remove walk-in/walk-out edges without modifying or extrapolating raw data."""
    if len(frames) < 2:
        raise ValueError("有效錄製不足兩幀，請重新錄製")
    values = np.array([[p.get(m, (np.nan,) * 3) for m in markers] for _, p in frames], dtype=float)
    complete = np.isfinite(values).all(axis=(1, 2))
    anchors = np.flatnonzero(complete)
    if len(anchors) < 2:
        raise ValueError("找不到足夠的完整全身骨架，請重新錄製")
    first, last = int(anchors[0]), int(anchors[-1])
    trimmed = frames[first:last + 1]
    return trimmed, {
        "trimmed_leading_frames": first,
        "trimmed_trailing_frames": len(frames) - last - 1,
        "raw_frames": len(frames),
    }


def select_static_segment(frames, markers=MARKER_ORDER, min_duration=0.5):
    """Use the longest fully tracked static interval; keep raw input untouched."""
    if len(frames) < 2:
        raise ValueError("有效錄製不足兩幀，請重新錄製")
    values = np.array([[p.get(m, (np.nan,) * 3) for m in markers] for _, p in frames], dtype=float)
    complete = np.isfinite(values).all(axis=(1, 2))
    best = None
    start = None
    for index, valid in enumerate(np.append(complete, False)):
        if valid and start is None:
            start = index
        elif not valid and start is not None:
            candidate = (start, index - 1)
            if best is None or frames[candidate[1]][0] - frames[candidate[0]][0] > frames[best[1]][0] - frames[best[0]][0]:
                best = candidate
            start = None
    if best is None or frames[best[1]][0] - frames[best[0]][0] < min_duration:
        longest = 0.0 if best is None else frames[best[1]][0] - frames[best[0]][0]
        raise ValueError(f"找不到至少 {min_duration:.1f} 秒的完整全身站姿（最長 {longest:.2f} 秒），請重新錄製")
    first, last = best
    return frames[first:last + 1], {
        "selected_static_start": float(frames[first][0]),
        "selected_static_end": float(frames[last][0]),
        "selected_static_frames": last - first + 1,
        "discarded_static_frames": len(frames) - (last - first + 1),
        "raw_frames": len(frames),
    }


def repair_frames(frames, markers=MARKER_ORDER, max_gap=0.2):
    """Fill bounded gaps only. Do not extrapolate, change timestamps or raw input."""
    if len(frames) < 2:
        raise ValueError("有效錄製不足兩幀，請重新錄製")
    times = np.array([t for t, _ in frames], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("時間戳記必須有限且嚴格遞增")
    intervals = np.diff(times)
    if np.any(intervals > 0.2 + 1e-8):
        raise ValueError("相機擷取中斷超過 0.2 秒，請重新錄製")
    sample_period = float(np.median(intervals))
    values = np.array([[p.get(m, (np.nan,) * 3) for m in markers] for _, p in frames], dtype=float)
    filled = 0
    longest_gap = 0.0
    gap_details = []
    for j, marker in enumerate(markers):
        valid = np.isfinite(values[:, j]).all(axis=1)
        if not valid[0] or not valid[-1]:
            raise ValueError(f"{marker} 在錄製頭尾缺失，無法補點，請重新錄製")
        anchors = np.flatnonzero(valid)
        for left, right in zip(anchors[:-1], anchors[1:]):
            if right == left + 1:
                continue
            gap = max(0.0, float(times[right] - times[left] - sample_period))
            if max_gap is not None and gap > max_gap + 1e-8:
                raise ValueError(f"{marker} 在 {times[left]:.2f}–{times[right]:.2f} 秒間缺失 {gap:.2f} 秒，超過 {max_gap} 秒上限，請重新錄製")
            weights = (times[left + 1:right] - times[left]) / (times[right] - times[left])
            values[left + 1:right, j] = values[left, j] + weights[:, None] * (values[right, j] - values[left, j])
            filled += right - left - 1
            longest_gap = max(longest_gap, gap)
            gap_details.append({"marker": marker, "start": float(times[left]), "end": float(times[right]),
                                "seconds": gap, "samples": int(right - left - 1)})
    repaired = [(float(t), {m: tuple(values[i, j]) for j, m in enumerate(markers)}) for i, t in enumerate(times)]
    return repaired, {"filled_marker_samples": int(filled), "max_gap_seconds": max_gap,
                      "max_interpolated_gap_seconds": longest_gap,
                      "interpolated_gaps": gap_details,
                      "quality_warning": longest_gap > 0.2 + 1e-8,
                      "effective_fps": float((len(times) - 1) / (times[-1] - times[0]))}


def unique_directory(parent, prefix):
    """Atomically reserve a directory; never reuse an existing result."""
    from datetime import datetime
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for i in range(1000):
        path = parent / f"{prefix}_{stamp}_{i:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise FileExistsError("無法建立唯一輸出資料夾")


