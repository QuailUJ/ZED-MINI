"""Durable recording files and timestamp-based constant-rate video encoding."""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from motion_data import MARKER_ORDER, write_trc


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def draw_skeleton(image, points, bones):
    if points is None:
        return image
    height, width = image.shape[:2]
    def valid(p):
        return p is not None and len(p) == 2 and all(v is not None and math.isfinite(v) for v in p) and 0 <= p[0] < width and 0 <= p[1] < height
    for a, b in bones:
        if a < len(points) and b < len(points) and valid(points[a]) and valid(points[b]):
            cv2.line(image, tuple(map(int, points[a])), tuple(map(int, points[b])), (60, 200, 60), 2)
    for point in points:
        if valid(point):
            cv2.circle(image, tuple(map(int, point)), 3, (0, 165, 255), -1)
    return image


class RecordingWriter:
    def __init__(self, directory, kind, body_id, fps, bones, image_size):
        self.directory = Path(directory)
        self.fps = float(fps)
        self.frames = []
        self.start_ns = None
        self.last_image = None
        self.video_frames = 0
        self.closed = False
        self.meta = {"version": 1, "kind": kind, "body_id": int(body_id), "video_fps": self.fps,
                     "bones": bones, "image_size": list(image_size), "status": "recording",
                     "video": "video.mp4", "skeleton": "skeleton.jsonl", "trc": "raw.trc"}
        # Directory is reserved by the caller; refuse any pre-existing recording.
        if any(self.directory.iterdir()):
            raise FileExistsError("錄製資料夾不是空的")
        self.video = cv2.VideoWriter(str(self.directory / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, image_size)
        if not self.video.isOpened():
            self.video.release()
            raise RuntimeError("無法建立 MP4，請檢查編碼器及輸出位置")
        self.skeleton = (self.directory / "skeleton.jsonl").open("x", encoding="utf-8")
        try:
            save_json(self.directory / "recording.json", self.meta)
        except Exception:
            self.skeleton.close()
            self.video.release()
            raise

    def append(self, timestamp_ns, image, positions, points, body_id):
        if self.start_ns is None:
            self.start_ns = timestamp_ns
        t = (timestamp_ns - self.start_ns) / 1e9
        if self.frames and t <= self.frames[-1][0]:
            return
        # Frame n represents time n/fps. Hold the previous camera image between grabs.
        target = int(math.ceil(t * self.fps - 1e-7))
        while self.last_image is not None and self.video_frames < target:
            self.video.write(self.last_image)
            self.video_frames += 1
        self.last_image = image.copy()
        if body_id != self.meta["body_id"]:
            positions, points, body_id = {}, None, None
        self.frames.append((t, positions))
        clean_points = None if points is None else [[float(v) if np.isfinite(v) else None for v in p] for p in points]
        clean_positions = {m: [float(v) if np.isfinite(v) else None for v in positions.get(m, (np.nan,) * 3)] for m in MARKER_ORDER}
        self.skeleton.write(json.dumps({"time": t, "timestamp_ns": int(timestamp_ns), "body_id": body_id,
                                        "points": clean_points, "positions": clean_positions}, allow_nan=False) + "\n")
        self.skeleton.flush()

    def finish(self, error=None):
        if self.closed:
            return self.directory
        self.closed = True
        try:
            if self.last_image is not None:
                self.video.write(self.last_image)
                self.video_frames += 1
        finally:
            self.video.release()
            self.skeleton.close()
        fps = (len(self.frames) - 1) / (self.frames[-1][0] - self.frames[0][0]) if len(self.frames) > 1 else self.fps
        write_trc(self.directory / "raw.trc", self.frames, fps, MARKER_ORDER, camera_rate=self.fps)
        self.meta.update(status="interrupted" if error else "recorded", error=error,
                         frames=len(self.frames), video_frames=self.video_frames,
                         duration=self.frames[-1][0] if self.frames else 0, effective_fps=fps)
        save_json(self.directory / "recording.json", self.meta)
        return self.directory
