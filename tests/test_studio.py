"""Offline regression tests. No ZED device or existing recordings are modified."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import queue

import cv2
import numpy as np

from capture_session import CaptureSession, select_body
from motion_data import MARKER_ORDER, read_trc, repair_frames, select_static_segment, trim_complete_edges, unique_directory, write_trc
from recording_io import RecordingWriter
from run_pipeline import apply_front_facing_sagittal_correction
from zed_studio import append_error_log, cleanup_session_data, named_session_directory, next_recording_paths, persist_calibration, publish_file, write_quality_note


class DataTests(unittest.TestCase):
    def frames(self, times):
        return [(t, {m: (t, 1., 2.) for m in MARKER_ORDER}) for t in times]

    def test_short_gap_and_raw_unchanged(self):
        frames = self.frames([0., .1, .2])
        frames[1][1]['Neck'] = (np.nan, 1., 2.)
        output, quality = repair_frames(frames)
        self.assertEqual(output[1][1]['Neck'], (.1, 1., 2.))
        self.assertTrue(np.isnan(frames[1][1]['Neck'][0]))
        self.assertEqual(quality['filled_marker_samples'], 1)
        self.assertIs(type(quality['filled_marker_samples']), int)

    def test_motion_quality_mode_warns_up_to_one_second(self):
        frames = self.frames([0., .1, .2, .3, .4, .5, .6])
        for i in range(1, 6):
            frames[i][1].pop('RWrist')
        output, quality = repair_frames(frames, max_gap=1.0)
        self.assertEqual(len(output), 7)
        self.assertTrue(quality['quality_warning'])
        self.assertAlmostEqual(quality['max_interpolated_gap_seconds'], .5)

    def test_motion_unlimited_mode_fills_long_bounded_gap(self):
        frames = self.frames([i / 10 for i in range(21)])
        for index in range(1, 20):
            frames[index][1].pop('LHeel')
        output, quality = repair_frames(frames, max_gap=None)
        self.assertEqual(output[2][1]['LHeel'], (.2, 1., 2.))
        self.assertTrue(quality['quality_warning'])
        self.assertAlmostEqual(quality['max_interpolated_gap_seconds'], 1.9)
        self.assertEqual(quality['interpolated_gaps'][0]['marker'], 'LHeel')

    def test_long_and_edge_gaps_rejected(self):
        frames = self.frames([0., .1, .2, .3, .4])
        for index in (1, 2, 3):
            frames[index][1].pop('Neck')
        with self.assertRaises(ValueError):
            repair_frames(frames)
        for times, missing in [([0., .1, .2], 0), ([0., .1, .2], 2)]:
            frames = self.frames(times)
            frames[missing][1].pop('Neck')
            with self.assertRaises(ValueError):
                repair_frames(frames)
        with self.assertRaises(ValueError):
            repair_frames(self.frames([0., .3]))

    def test_complete_edges_are_trimmed_before_repair(self):
        frames = self.frames([0., .1, .2, .3, .4])
        frames[0][1].pop('Neck')
        frames[-1][1].pop('Neck')
        trimmed, quality = trim_complete_edges(frames)
        self.assertEqual([t for t, _ in trimmed], [.1, .2, .3])
        self.assertEqual(quality['trimmed_leading_frames'], 1)
        self.assertEqual(quality['trimmed_trailing_frames'], 1)
        self.assertEqual(len(repair_frames(trimmed)[0]), 3)

    def test_static_uses_longest_complete_interval(self):
        frames = self.frames([i / 10 for i in range(31)])
        for index in list(range(0, 5)) + list(range(17, 31)):
            frames[index][1].pop("LHeel")
        selected, quality = select_static_segment(frames)
        self.assertEqual([t for t, _ in selected], [i / 10 for i in range(5, 17)])
        self.assertEqual(quality["selected_static_frames"], 12)
        self.assertEqual(quality["discarded_static_frames"], 19)

    def test_static_rejects_when_no_full_second_exists(self):
        frames = self.frames([i / 10 for i in range(5)])
        with self.assertRaisesRegex(ValueError, "至少 0.5 秒"):
            select_static_segment(frames)

    def test_empty_single_and_invalid_timestamps(self):
        for frames in [[], self.frames([0.]), self.frames([0., 0.]), self.frames([.1, 0.]), self.frames([0., np.nan])]:
            with self.assertRaises(ValueError):
                repair_frames(frames)

    def test_person_lock(self):
        a, b = SimpleNamespace(id=1), SimpleNamespace(id=2)
        self.assertIsNone(select_body([]))
        self.assertIsNone(select_body([a, b]))
        self.assertIs(select_body([a]), a)
        self.assertIs(select_body([b, a], 1), a)
        self.assertIsNone(select_body([b], 1))

    def test_unique_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = unique_directory(directory, 'trial'), unique_directory(directory, 'trial')
            self.assertNotEqual(a, b)

    def test_named_session_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = named_session_directory(directory, "  鵪鶉測試  ")
            self.assertEqual(path.name, "鵪鶉測試")
            with self.assertRaises(FileExistsError):
                named_session_directory(directory, "鵪鶉測試")
            for name in ("", "../outside", "bad:name", "trailing."):
                with self.assertRaises(ValueError):
                    named_session_directory(directory, name)

    def test_clean_public_session_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            session = named_session_directory(directory, "01")
            internal, public = next_recording_paths(session, "motion")
            source = internal / "video.mp4"
            source.write_bytes(b"video")
            publish_file(source, public / f"{public.name}.mp4")
            self.assertEqual((public / "action_01.mp4").read_bytes(), b"video")
            cleanup_session_data(session)
            self.assertFalse((session / ".studio").exists())
            self.assertEqual((public / "action_01.mp4").read_bytes(), b"video")

    def test_persisted_calibration_survives_temporary_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            session = named_session_directory(directory, "01")
            conversion = session / ".studio" / "conversion"
            conversion.mkdir()
            (conversion / "model.osim").write_bytes(b"model")
            (conversion / "calibration.json").write_text(json.dumps({"version": 1, "model": "model.osim", "offset": [1, 2, 3]}))
            path = persist_calibration(session, {"model": str(conversion / "model.osim"),
                                                 "calibration": str(conversion / "calibration.json")})
            cleanup_session_data(session)
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text())["model"], "01.osim")
            self.assertEqual((session / "01.osim").read_bytes(), b"model")

    def test_one_error_log_per_session(self):
        with tempfile.TemporaryDirectory() as directory:
            session = named_session_directory(directory, "01")
            source = session / ".studio" / "error.txt"
            source.write_text("first error", encoding="utf-8")
            log = append_error_log(session, "靜態轉換失敗", source, "fallback")
            append_error_log(session, "動作轉換失敗", source, "fallback")
            content = log.read_text(encoding="utf-8")
            self.assertIn("靜態轉換失敗", content)
            self.assertIn("動作轉換失敗", content)
            self.assertEqual(list(session.glob("*-errors.log")), [log])

    def test_motion_quality_note_discloses_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            motion = Path(directory) / "action_01"
            motion.mkdir()
            result = {"quality": {"filled_marker_samples": 4, "max_interpolated_gap_seconds": .4,
                                  "trimmed_leading_frames": 1, "trimmed_trailing_frames": 2,
                                  "interpolated_gaps": [{"marker": "LHeel", "start": 1., "end": 1.5,
                                                         "seconds": .4, "samples": 4}]},
                      "marker_errors": {"mean_rms_m": .01, "max_error_m": .03}}
            note = write_quality_note(motion, result)
            self.assertEqual(note.name, "action_01-notes.txt")
            content = note.read_text(encoding="utf-8-sig")
            self.assertIn("線性補點", content)
            self.assertIn("LHeel", content)
            self.assertIn("400", content)

    def test_trc_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.trc'
            frames = self.frames([0., .1, .2])
            write_trc(path, frames, 10., MARKER_ORDER)
            names, fps, actual = read_trc(path)
            self.assertEqual(names, MARKER_ORDER)
            self.assertEqual(fps, 10.)
            self.assertEqual(actual, frames)

    def test_conversion_import_without_pyzed(self):
        script = "import sys; sys.modules['pyzed']=None; import run_pipeline; assert 'capture_to_trc' not in sys.modules"
        subprocess.run([sys.executable, '-B', '-c', script], cwd=Path(__file__).parent.parent, check=True, capture_output=True)

    def test_front_facing_sagittal_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.mot"
            names = ["time", "pelvis_tilt", "hip_flexion_r", "hip_flexion_l",
                     "lumbar_extension", "arm_flex_r", "arm_flex_l", "knee_angle_r"]
            path.write_text("Coordinates\nendheader\n" + "\t".join(names) +
                            "\n0\t30\t-20\t-25\t4\t10\t-12\t60\n", encoding="utf-8")
            apply_front_facing_sagittal_correction(path)
            values = [float(value) for value in path.read_text().splitlines()[-1].split()]
            actual = dict(zip(names, values))
            self.assertEqual(actual["pelvis_tilt"], -30)
            self.assertEqual(actual["hip_flexion_r"], 20)
            self.assertEqual(actual["hip_flexion_l"], 25)
            self.assertEqual(actual["lumbar_extension"], 30)
            self.assertEqual(actual["arm_flex_r"], -10)
            self.assertEqual(actual["arm_flex_l"], 12)
            self.assertEqual(actual["knee_angle_r"], 60)


class RecordingTests(unittest.TestCase):
    def test_video_timing_sidecar_and_person_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RecordingWriter(directory, 'motion', 7, 30, [(0, 1)], (64, 48))
            frame = np.zeros((48, 64, 3), np.uint8)
            positions = {m: (0., 1., 2.) for m in MARKER_ORDER}
            for i in range(4):
                writer.append(1_000_000_000 + i * 100_000_000, frame, positions, [[10., 10.], [20., 20.]], 8 if i == 1 else 7)
            writer.finish()
            writer.finish()  # Safe to close twice.
            meta = json.loads((Path(directory) / 'recording.json').read_text())
            self.assertEqual(meta['video_frames'], 10)
            self.assertAlmostEqual(meta['duration'], .3)
            samples = [json.loads(s) for s in (Path(directory) / 'skeleton.jsonl').read_text().splitlines()]
            self.assertIsNone(samples[1]['body_id'])
            self.assertIsNone(samples[1]['points'])
            _, _, raw = read_trc(Path(directory) / 'raw.trc')
            self.assertTrue(np.isnan(raw[1][1]['Neck'][0]))
            cap = cv2.VideoCapture(str(Path(directory) / 'video.mp4'))
            self.assertTrue(cap.isOpened())
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 10)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 30)
            cap.release()
            with self.assertRaises(FileExistsError):
                RecordingWriter(directory, 'motion', 7, 30, [], (64, 48))

    def test_empty_and_interrupted_recordings_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RecordingWriter(directory, 'static', 1, 30, [], (64, 48))
            writer.finish('camera lost')
            meta = json.loads((Path(directory) / 'recording.json').read_text())
            self.assertEqual(meta['status'], 'interrupted')
            self.assertTrue((Path(directory) / 'raw.trc').is_file())
            with self.assertRaises(ValueError):
                repair_frames(read_trc(Path(directory) / 'raw.trc')[2])


class WorkerTests(unittest.TestCase):
    def test_manual_stop_repeated_recording_and_disconnect(self):
        import pyzed.sl as sl
        import capture_to_trc  # Load real marker indexes before replacing SDK objects.
        events = queue.Queue()
        worker = CaptureSession(events, ready_seconds=0)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'first', Path(directory) / 'second'
            first.mkdir()
            second.mkdir()
            worker.commands.put(('start', (first, 'static')))
            body = SimpleNamespace(id=3, tracking_state=sl.OBJECT_TRACKING_STATE.OK,
                                   keypoint=np.ones((38, 3)), keypoint_2d=np.ones((38, 2)),
                                   keypoint_confidence=np.ones(38) * 100)

            class FakeCamera:
                index = 0
                closed = False
                def open(self, init): return sl.ERROR_CODE.SUCCESS
                def enable_positional_tracking(self, params): return sl.ERROR_CODE.SUCCESS
                def enable_body_tracking(self, params): return sl.ERROR_CODE.SUCCESS
                def get_camera_information(self): return SimpleNamespace(camera_configuration=SimpleNamespace(fps=30))
                def get_timestamp(self, ref): return SimpleNamespace(get_nanoseconds=lambda: 1_000_000_000 + self.index * 33_333_333)
                def grab(self, params):
                    self.index += 1
                    if self.index == 3:
                        worker.commands.put(('stop', None))
                    if self.index == 4:
                        worker.commands.put(('start', (second, 'motion')))
                    if self.index == 7:
                        raise RuntimeError('simulated camera disconnect')
                    return sl.ERROR_CODE.SUCCESS
                def retrieve_image(self, image, view): return sl.ERROR_CODE.SUCCESS
                def retrieve_bodies(self, bodies, runtime):
                    bodies.is_new = True
                    bodies.timestamp = self.get_timestamp(None)
                    bodies.body_list = [body]
                    return sl.ERROR_CODE.SUCCESS
                def disable_body_tracking(self): pass
                def disable_positional_tracking(self): pass
                def close(self): self.closed = True

            camera = FakeCamera()
            with patch.object(sl, 'Camera', return_value=camera), patch.object(sl, 'Bodies', SimpleNamespace), patch.object(sl, 'Mat', return_value=SimpleNamespace(get_data=lambda: np.zeros((48, 64, 4), np.uint8))):
                worker.run()
            self.assertTrue(camera.closed)
            self.assertEqual(json.loads((first/'recording.json').read_text())['status'], 'recorded')
            self.assertEqual(json.loads((second/'recording.json').read_text())['status'], 'interrupted')
            kinds = [events.get_nowait()[0] for _ in range(events.qsize())]
            self.assertIn('recorded', kinds)
            self.assertIn('interrupted', kinds)
            self.assertIn('camera_closed', kinds)

    def test_camera_initialization_failure(self):
        import pyzed.sl as sl
        from unittest.mock import MagicMock
        camera = MagicMock()
        camera.open.return_value = sl.ERROR_CODE.CAMERA_NOT_DETECTED
        events = queue.Queue()
        with patch.object(sl, 'Camera', return_value=camera):
            CaptureSession(events).run()
        camera.close.assert_called_once()
        self.assertEqual(events.get_nowait()[0], 'camera_error')
        self.assertEqual(events.get_nowait()[0], 'camera_closed')


class PipelineFailureTests(unittest.TestCase):
    def test_bad_data_keeps_raw_and_writes_failure_report(self):
        from run_pipeline import execute_job
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw.trc'
            write_trc(source, [], 30, MARKER_ORDER)
            before = source.read_bytes()
            result = execute_job('scale', source, root / 'conversion')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue((root / 'conversion' / 'error.txt').exists())
            self.assertFalse((root / 'conversion' / 'model.osim').exists())
            with self.assertRaises(FileExistsError):
                execute_job('scale', source, root / 'conversion')


if __name__ == '__main__':
    unittest.main()
