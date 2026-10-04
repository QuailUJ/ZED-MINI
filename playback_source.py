"""Background file loading and coalesced video decoding for the Tk player."""
import json
import math
from pathlib import Path
import queue
import threading

import cv2


class PlaybackSource(threading.Thread):
    def __init__(self, directory):
        super().__init__(daemon=True)
        self.directory = Path(directory)
        self.events = queue.Queue()
        self.stopped = threading.Event()
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.target = None
        self.latest = None

    def request(self, index):
        with self.lock:
            if index == self.target:
                return
            self.target = index
        self.wake.set()

    def take_frame(self):
        with self.lock:
            frame, self.latest = self.latest, None
        return frame

    def close(self):
        self.stopped.set()
        self.wake.set()

    def run(self):
        capture = None
        try:
            meta = json.loads((self.directory / "recording.json").read_text(encoding="utf-8"))
            if not isinstance(meta["body_id"], int):
                raise ValueError("錄製人物 ID 格式無效")
            # Only open local recording files, never a URL or live video stream.
            video_path = self.directory / meta["video"]
            if not video_path.is_file():
                raise ValueError("找不到錄製影片")
            samples = []
            previous = -1.0
            with (self.directory / meta["skeleton"]).open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if self.stopped.is_set():
                        return
                    sample = json.loads(line)
                    t = sample["time"]
                    if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 or t <= previous:
                        raise ValueError(f"骨架第 {number} 列時間無效或未遞增")
                    previous = t
                    points = sample["points"]
                    if points is not None and (not isinstance(points, list) or any(
                        not isinstance(p, list) or len(p) != 2 or any(
                            v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)) for v in p
                        ) for p in points
                    )):
                        raise ValueError(f"骨架第 {number} 列座標格式無效")
                    # 3D positions are unnecessary for replay; don't retain them in memory.
                    samples.append({"time": t, "body_id": sample["body_id"], "points": points})
            bones = meta["bones"]
            if not isinstance(bones, list) or any(not isinstance(b, list) or len(b) != 2 or
                any(not isinstance(i, int) or i < 0 for i in b) for b in bones):
                raise ValueError("骨架連線格式無效")
            if not samples:
                raise ValueError("沒有骨架時間資料")
            if self.stopped.is_set():
                return
            capture = cv2.VideoCapture(str(video_path))
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not capture.isOpened() or count < 1 or not math.isfinite(fps) or fps <= 0:
                raise ValueError("影片無法讀取或沒有有效影格")
            self.events.put(("ready", (meta, samples, fps, count)))
            decoded = -1
            while not self.stopped.is_set():
                self.wake.wait(.1)
                self.wake.clear()
                if self.stopped.is_set():
                    break
                with self.lock:
                    target = self.target
                if target is None or target == decoded:
                    continue
                target = max(0, min(count - 1, target))
                if target != decoded + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"第 {target + 1} 幀讀取失敗，影片可能不完整")
                decoded = target
                with self.lock:
                    if target == self.target:
                        self.latest = (target, frame)
        except Exception as exc:
            if not self.stopped.is_set():
                self.events.put(("error", str(exc)))
        finally:
            if capture is not None:
                capture.release()
