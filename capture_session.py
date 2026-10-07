"""Own the ZED camera on one worker thread. GUI never calls SDK methods."""
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from motion_data import MARKER_ORDER
from recording_io import RecordingWriter


_DLL_DIRECTORIES = []


def prepare_zed_sdk():
    """Make the installed ZED SDK DLLs available to the bundled Python API."""
    if os.name != "nt":
        return None
    sdk_root = Path(os.environ.get("ZED_SDK_ROOT_DIR", r"C:\Program Files (x86)\ZED SDK"))
    bin_dir = sdk_root / "bin"
    required = (bin_dir / "sl_zed64.dll", bin_dir / "sl_ai64.dll")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"找不到 ZED SDK 元件：{', '.join(missing)}。"
            "請安裝 ZED SDK 5.5.0，安裝完成後重新啟動程式。"
        )
    _DLL_DIRECTORIES.append(os.add_dll_directory(str(bin_dir)))
    return sdk_root


def select_body(bodies, locked_id=None):
    if locked_id is None:
        return bodies[0] if len(bodies) == 1 else None
    return next((body for body in bodies if body.id == locked_id), None)


class CaptureSession(threading.Thread):
    def __init__(self, events, ready_seconds=0.5, logger=None):
        super().__init__(daemon=False)
        self.events = events
        self.ready_seconds = ready_seconds
        self.commands = queue.Queue()
        self.stop_event = threading.Event()
        self.latest = None
        self.lock = threading.Lock()
        self.logger = logger

    def report_status(self, message):
        if self.logger is not None:
            self.logger(message)
        self.events.put(("camera_status", message))

    def snapshot(self):
        with self.lock:
            return self.latest

    def run(self):
        camera = None
        opened = False
        writer = None
        positional = tracking = False
        locked_id = None
        last_success = time.monotonic()
        try:
            self.report_status("正在載入 ZED SDK…")
            sdk_root = prepare_zed_sdk()
            if sdk_root is not None:
                self.report_status(f"使用電腦已安裝的 ZED SDK：{sdk_root}")
            import pyzed.sl as sl
            from capture_to_trc import MARKER_TO_KEYPOINT_INDEX, SKELETON_BONES, to_opensim_axes
            self.report_status(f"ZED SDK {sl.Camera.get_sdk_version()} 已載入，正在開啟相機…")
            camera = sl.Camera()
            init = sl.InitParameters()
            init.coordinate_units = sl.UNIT.METER
            init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            init.depth_mode = sl.DEPTH_MODE.NEURAL
            init.camera_resolution = sl.RESOLUTION.HD720
            init.camera_fps = 30
            status = camera.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"無法開啟 ZED 相機：{status}")
            opened = True
            self.report_status("相機已開啟，正在啟用位置追蹤…")
            status = camera.enable_positional_tracking(sl.PositionalTrackingParameters())
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"無法啟用位置追蹤：{status}")
            positional = True
            self.report_status("位置追蹤已啟用，正在載入人體追蹤 AI…")
            params = sl.BodyTrackingParameters()
            params.enable_tracking = True
            params.enable_body_fitting = True
            params.body_format = sl.BODY_FORMAT.BODY_38
            params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
            status = camera.enable_body_tracking(params)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"無法啟用人體追蹤：{status}")
            tracking = True
            self.report_status("人體追蹤 AI 已啟用，正在等待影像…")
            runtime = sl.RuntimeParameters()
            runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
            body_runtime = sl.BodyTrackingRuntimeParameters()
            bodies, image = sl.Bodies(), sl.Mat()
            fps = camera.get_camera_information().camera_configuration.fps or 30
            self.events.put(("camera_ready", None))
            last_success = time.monotonic()
            pending_start = None
            ready_since_ns = None
            ready_body_id = None
            while not self.stop_event.is_set():
                # Process stop even if grabs keep failing.
                while True:
                    try:
                        command, payload = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == "stop" and writer is not None:
                        directory = writer.finish()
                        writer = None
                        locked_id = None
                        self.events.put(("recorded", str(directory)))
                    elif command == "stop" and pending_start is not None:
                        pending_start = None
                        ready_since_ns = ready_body_id = None
                        self.events.put(("start_failed", "已取消等待錄製"))
                    elif command == "start" and writer is None:
                        pending_start = payload
                        ready_since_ns = ready_body_id = None
                status = camera.grab(runtime)
                if status != sl.ERROR_CODE.SUCCESS:
                    if pending_start is not None:
                        self.events.put(("start_failed", "相機沒有提供影像，請稍後重試"))
                    if time.monotonic() - last_success > 3:
                        raise RuntimeError(f"相機影像中斷：{status}")
                    self.stop_event.wait(0.01)
                    continue
                last_success = time.monotonic()
                if camera.retrieve_image(image, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError("無法取得相機影像")
                image_ns = camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
                status = camera.retrieve_bodies(bodies, body_runtime)
                fresh = status == sl.ERROR_CODE.SUCCESS and bodies.is_new
                # A stale detection is missing data, never relabel it with a new image time.
                if fresh:
                    fresh = abs(bodies.timestamp.get_nanoseconds() - image_ns) <= 1_000_000
                candidates = [b for b in bodies.body_list if b.tracking_state == sl.OBJECT_TRACKING_STATE.OK and b.id >= 0] if fresh else []
                body = select_body(candidates, locked_id)
                points = None if body is None else np.array(body.keypoint_2d, copy=True)
                positions = {}
                if body is not None:
                    for marker, index in MARKER_TO_KEYPOINT_INDEX.items():
                        point = body.keypoint[index]
                        if np.isfinite(point).all() and body.keypoint_confidence[index] > 0:
                            positions[marker] = to_opensim_axes(*map(float, point))
                if pending_start is not None:
                    complete = body is not None and all(marker in positions for marker in MARKER_ORDER)
                    if not complete:
                        ready_since_ns = ready_body_id = None
                    else:
                        body_id = int(body.id)
                        if ready_body_id != body_id:
                            ready_since_ns, ready_body_id = image_ns, body_id
                        if (image_ns - ready_since_ns) / 1e9 >= self.ready_seconds:
                            directory, kind = pending_start
                            locked_id = body_id
                            writer = RecordingWriter(directory, kind, locked_id, fps, SKELETON_BONES, (bgr.shape[1], bgr.shape[0]))
                            pending_start = None
                            self.events.put(("recording", str(directory)))
                if writer is not None:
                    writer.append(image_ns, bgr, positions, points, int(body.id) if body is not None else None)
                with self.lock:
                    self.latest = (bgr, points, SKELETON_BONES, len(candidates),
                                   int(body.id) if body is not None else None,
                                   writer.frames[-1][0] if writer and writer.frames else None)
        except Exception as exc:
            if writer is not None:
                try:
                    directory = writer.finish(str(exc))
                    self.events.put(("interrupted", str(directory)))
                except Exception as save_exc:
                    self.events.put(("error", f"保存未完成：{save_exc}"))
                writer = None
            self.events.put(("camera_error", str(exc)))
        finally:
            try:
                if writer is not None:
                    directory = writer.finish("關閉介面，中止錄製")
                    self.events.put(("interrupted", str(directory)))
            finally:
                if camera is not None and opened:
                    if tracking:
                        camera.disable_body_tracking()
                    if positional:
                        camera.disable_positional_tracking()
                    camera.close()
                self.events.put(("camera_closed", None))
