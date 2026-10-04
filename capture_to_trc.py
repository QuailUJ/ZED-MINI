"""
Record ZED body tracking (BODY_38) and export it as a .trc file whose marker
names match opencap-core's LaiUhlrich2022_markers_openpose.xml, so the output
can be scaled/IK'd directly against opensimPipeline/Models/LaiUhlrich2022.osim.

Shows a live preview window with the skeleton overlay so you can see whether
you're in frame and being tracked before/while recording. Press 'q' in the
preview window (or Ctrl+C in the terminal) to stop early.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

from motion_data import KEYPOINT_TO_MARKER, MARKER_ORDER, write_trc

KEYPOINT_INDEX = {p.name: p.value for p in sl.BODY_38_PARTS if p.name != "LAST"}
MARKER_TO_KEYPOINT_INDEX = {marker: KEYPOINT_INDEX[zed_name] for zed_name, marker in KEYPOINT_TO_MARKER.items()}
USED_KEYPOINT_INDICES = set(MARKER_TO_KEYPOINT_INDEX.values())
SKELETON_BONES = [(a.value, b.value) for a, b in sl.BODY_38_BONES]


def to_opensim_axes(x, y, z):
    """
    ZED's RIGHT_HANDED_Y_UP has X to the camera's right, Y up, Z toward the
    camera/viewer. OpenSim/biomechanics convention wants X forward (the
    direction the subject faces), Y up, Z to the subject's right. Verified
    empirically: a subject standing square to the camera should IK to near-0
    pelvis_tilt/list/rotation; this mapping was the one that did that
    (previously the axes were off by ~90 degrees, giving nonsense pelvis
    rotation).
    """
    return z, y, -x


def draw_overlay(image_bgr, body, status_lines):
    if body is not None:
        kps_2d = body.keypoint_2d
        for i0, i1 in SKELETON_BONES:
            p0, p1 = kps_2d[i0], kps_2d[i1]
            if not np.isfinite(p0).all() or not np.isfinite(p1).all():
                continue
            cv2.line(image_bgr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (60, 200, 60), 2)
        for idx, pt in enumerate(kps_2d):
            if not np.isfinite(pt).all():
                continue
            color = (0, 165, 255) if idx in USED_KEYPOINT_INDICES else (120, 120, 120)
            cv2.circle(image_bgr, (int(pt[0]), int(pt[1])), 4, color, -1)

    y = 30
    for line, color in status_lines:
        cv2.putText(image_bgr, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        y += 35
    return image_bgr


def main():
    parser = argparse.ArgumentParser(description="Record ZED body tracking to an OpenCap-compatible .trc file")
    parser.add_argument("output", help="Output .trc path")
    parser.add_argument("--svo", help="Play back a recorded .svo2 file instead of a live camera")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (live camera mode)")
    parser.add_argument("--countdown", type=float, default=0.0, help="Auto-start after N seconds instead of waiting for spacebar (live camera mode)")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live preview window (headless)")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; choose another path")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    if args.svo:
        init_params.set_from_svo_file(args.svo)

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        sys.exit(1)

    if zed.enable_positional_tracking(sl.PositionalTrackingParameters()) != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError("Failed to enable positional tracking")

    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.enable_body_fitting = True
    body_params.body_format = sl.BODY_FORMAT.BODY_38
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    err = zed.enable_body_tracking(body_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to enable body tracking: {err}")
        zed.close()
        sys.exit(1)

    body_runtime_params = sl.BodyTrackingRuntimeParameters()
    bodies = sl.Bodies()
    image_mat = sl.Mat()

    fps = zed.get_camera_information().camera_configuration.fps or 30
    frames = []
    recording_start = None
    loop_start = time.monotonic()
    last_success = loop_start
    locked_id = None
    source_start_ns = None
    from capture_session import select_body
    preview = not args.no_preview
    window_name = "ZED capture (SPACE to start, q to quit)"

    # Auto-start only if the caller passed --countdown, or if there's no window
    # to receive a keypress in the first place (nothing to wait on then).
    auto_start_at = None
    if args.svo:
        auto_start_at = loop_start
    elif args.countdown > 0:
        auto_start_at = loop_start + args.countdown
    elif not preview:
        auto_start_at = loop_start

    if args.svo:
        print("Playing back SVO...")
    elif auto_start_at is not None and not preview:
        print("No preview window (--no-preview) - recording starts immediately.")
    elif auto_start_at is not None:
        print(f"Get in frame - recording starts in {args.countdown:.0f}s. Press 'q' to quit early.")
    else:
        print("Get in frame, then press SPACE in the preview window to start recording ('q' to quit).")

    try:
        while True:
            grab_status = zed.grab()
            if grab_status != sl.ERROR_CODE.SUCCESS:
                if args.svo:
                    break
                if time.monotonic() - last_success > 3:
                    print("Camera interrupted; preserving captured frames.")
                    break
                time.sleep(0.01)
                continue

            last_success = time.monotonic()
            body_status = zed.retrieve_bodies(bodies, body_runtime_params)
            now = time.monotonic()
            timestamp_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            candidates = [b for b in bodies.body_list if b.tracking_state == sl.OBJECT_TRACKING_STATE.OK and b.id >= 0] if body_status == sl.ERROR_CODE.SUCCESS and bodies.is_new else []
            body = select_body(candidates, locked_id)

            key = -1
            if preview:
                zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                image_bgr = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
                key = cv2.waitKey(1) & 0xFF

            waiting = recording_start is None and (auto_start_at is None or now < auto_start_at)
            if waiting and key == ord(' '):
                auto_start_at = now
                waiting = False

            if recording_start is None and not waiting and body is not None:
                recording_start = now
                source_start_ns = timestamp_ns
                locked_id = body.id
                print("Recording...")

            if recording_start is not None:
                positions = {} if body is None else {
                    marker: to_opensim_axes(*(float(c) for c in body.keypoint[idx]))
                    for marker, idx in MARKER_TO_KEYPOINT_INDEX.items()
                    if body.keypoint_confidence[idx] > 0
                }
                t = (timestamp_ns - source_start_ns) / 1e9
                if not frames or t > frames[-1][0]:
                    frames.append((t, positions))

            if preview:
                if recording_start is None:
                    if auto_start_at is not None:
                        remaining = max(0.0, auto_start_at - now)
                        status = [(f"Starts in {remaining:.1f}s", (0, 220, 255))]
                    else:
                        status = [("Press SPACE to start", (0, 220, 255))]
                else:
                    elapsed = now - recording_start
                    status = [(f"REC {elapsed:.1f}s", (0, 0, 255))]
                    if not args.svo:
                        status.append((f"/ {args.duration:.0f}s", (0, 0, 255)))
                status.append(("Tracked" if body is not None else "No person detected", (0, 220, 0) if body is not None else (0, 0, 255)))
                draw_overlay(image_bgr, body, status)
                cv2.imshow(window_name, image_bgr)
                if key == ord('q'):
                    break

            if recording_start is not None and not args.svo and (now - recording_start) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if preview:
            cv2.destroyAllWindows()
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()

    if not frames:
        print("No body frames captured - was a person visible to the camera?")
        sys.exit(1)

    if len(frames) > 1:
        fps = (len(frames) - 1) / (frames[-1][0] - frames[0][0])
    write_trc(args.output, frames, fps, MARKER_ORDER)
    print(f"Wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
