"""Own the ZED camera on one worker thread. GUI never calls SDK methods."""
import queue
import threading
import time

import cv2
import numpy as np

from motion_data import MARKER_ORDER
from recording_io import RecordingWriter


def select_body(bodies, locked_id=None):
    if locked_id is None:
        return bodies[0] if len(bodies) == 1 else None
    return next((body for body in bodies if body.id == locked_id), None)


class CaptureSession(threading.Thread):
    def __init__(self, events, ready_seconds=0.5):
        super().__init__(daemon=False)
        self.events = events
        self.ready_seconds = ready_seconds
        self.commands = queue.Queue()
        self.stop_event = threading.Event()
        self.latest = None
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            return self.latest

    def run(self):
        camera = None
        writer = None
        positional = tracking = False
        locked_id = None
        last_success = time.monotonic()
        try:
            self.events.put(("camera_status", "正在載入 ZED SDK…"))
            import pyzed.sl as sl
            from capture_to_trc import MARKER_TO_KEYPOINT_INDEX, SKELETON_BONES, to_opensim_axes
            self.events.put(("camera_status", f"ZED SDK {sl.Camera.get_sdk_version()} 已載入，正在開啟相機…"))
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
            self.events.put(("camera_status", "相機已開啟，正在啟用位置追蹤…"))
            status = camera.enable_positional_tracking(sl.PositionalTrackingParameters())
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"無法啟用位置追蹤：{status}")
            positional = True
            self.events.put(("camera_status", "位置追蹤已啟用，正在載入人體追蹤 AI…"))
            params = sl.BodyTrackingParameters()
            params.enable_tracking = True
            params.enable_body_fitting = True
            params.body_format = sl.BODY_FORMAT.BODY_38
            params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
            status = camera.enable_body_tracking(params)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"無法啟用人體追蹤：{status}")
            tracking = True
            self.events.put(("camera_status", "人體追蹤 AI 已啟用，正在等待影像…"))
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
                if camera is not None:
                    if tracking:
                        camera.disable_body_tracking()
                    if positional:
                        camera.disable_positional_tracking()
                    camera.close()
                self.events.put(("camera_closed", None))
