"""No-camera GUI/decoder regression tests; all recordings/settings are temporary."""
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from motion_data import MARKER_ORDER
from playback_source import PlaybackSource
from recording_io import RecordingWriter
from zed_studio import Playback, Studio


def make_recording(path):
    path.mkdir()
    writer = RecordingWriter(path, "motion", 1, 30, [(0, 1)], (320, 240))
    frame = np.zeros((240, 320, 3), np.uint8)
    positions = {m: (0., 1., 2.) for m in MARKER_ORDER}
    for i in range(31):
        writer.append(1_000_000_000 + round(i * 1e9 / 30), frame, positions, [[20., 20.], [100., 100.]], 1)
    writer.finish()


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "recording"
        make_recording(self.path)
        self.source = None

    def tearDown(self):
        if self.source:
            self.source.close()
            self.source.join(3)
            self.assertFalse(self.source.is_alive())
        self.temp.cleanup()

    def wait_frame(self, index):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.source.take_frame()
            if result is not None and result[0] == index:
                return result
            time.sleep(.005)
        self.fail(f"decoder did not deliver frame {index}")

    def test_sequential_decode_does_not_seek_each_frame(self):
        capture = cv2.VideoCapture(str(self.path / "video.mp4"))
        from unittest.mock import MagicMock
        spy = MagicMock(wraps=capture)
        with patch("playback_source.cv2.VideoCapture", return_value=spy):
            self.source = PlaybackSource(self.path)
            self.source.start()
            event, _ = self.source.events.get(timeout=3)
            self.assertEqual(event, "ready")
            for index in [0, 1, 2]:
                self.source.request(index)
                self.wait_frame(index)
            self.assertEqual(spy.set.call_count, 0)
            self.source.request(20)
            self.wait_frame(20)
            self.assertEqual(spy.set.call_count, 1)
            self.source.close()
            self.source.join(3)
        spy.release.assert_called_once()

    def test_corrupt_skeleton_reports_error(self):
        (self.path / "skeleton.jsonl").write_text("{broken", encoding="utf-8")
        self.source = PlaybackSource(self.path)
        self.source.start()
        self.assertEqual(self.source.events.get(timeout=3)[0], "error")

    def test_rapid_seeks_only_keep_latest_request(self):
        self.source = PlaybackSource(self.path)
        self.source.start()
        self.assertEqual(self.source.events.get(timeout=3)[0], "ready")
        for i in range(1000):
            self.source.request(i % 31)
        self.source.request(30)
        self.wait_frame(30)
        self.assertLessEqual(self.source.events.qsize(), 1)


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.app = Studio(settings_path=self.path / "settings.json", show_guide=False, enable_voice=False, auto_connect=False)
        self.app.withdraw()
        self.app.update_idletasks()
        self.errors = []
        self.app.report_callback_exception = lambda *error: self.errors.append(error)
        self.players = []

    def pump(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.app.destroyed:
                self.app.update()
            if predicate():
                return
            time.sleep(.005)
        self.fail("GUI operation timed out")

    def tearDown(self):
        self.app.close() if not self.app.destroyed else None
        self.pump(lambda: self.app.destroyed)
        for player in self.players:
            player.source.join(3)
            self.assertFalse(player.source.is_alive())
        self.temp.cleanup()
        self.assertEqual(self.errors, [])

    def player(self):
        directory = self.path / "recording"
        make_recording(directory)
        player = Playback(self.app, directory)
        player.withdraw()
        self.players.append(player)
        return player

    def test_guide_navigation_preference_and_reopen(self):
        self.app.show_guide()
        guide = self.app.guide
        guide.withdraw()
        self.assertEqual(guide.index, 0)
        guide.move(1)
        guide.move(-1)
        self.assertEqual(guide.index, 0)
        for _ in guide.STEPS:
            guide.move(1)
        self.assertFalse(guide.winfo_exists())
        self.assertFalse(json.loads(self.app.settings_path.read_text())["show_guide"])
        self.app.show_guide()
        self.assertTrue(self.app.guide.winfo_exists())

    def test_camera_absent_can_retry_and_close(self):
        # Exercise the actual SDK's no-device result, not a fake camera.
        import pyzed.sl as sl
        if sl.Camera.get_device_list():
            self.skipTest("Requires no connected ZED")
        for _ in range(2):
            self.app.connect()
            self.pump(lambda: not self.app.camera.is_alive() and not self.app.camera_ready, timeout=10)
            self.pump(lambda: "相機" in self.app.status.get())
            self.assertEqual(str(self.app.motion_button["state"]), "disabled")
            self.app.update_controls()
            self.assertEqual(str(self.app.connect_button["state"]), "normal")

    def test_playback_seek_toggle_end_and_close(self):
        player = self.player()
        self.pump(lambda: player.last_index == 0)
        player.seek("0.5")
        self.pump(lambda: player.last_index == 15)
        player.show_bones.set(False)
        player.render()
        player.show_bones.set(True)
        player.render()
        player.toggle()
        started = player.started
        callback_count = len(player.slider._tclCommands)
        self.pump(lambda: not player.playing and player.last_index == 30)
        self.assertEqual(player.started, started)
        self.assertEqual(len(player.slider._tclCommands), callback_count)
        player.toggle()
        self.pump(lambda: player.last_index < 10)
        player.close()
        player.close()

    def test_slow_decoder_keeps_ui_responsive(self):
        directory = self.path / "recording"
        make_recording(directory)
        capture = cv2.VideoCapture(str(directory / "video.mp4"))
        from unittest.mock import MagicMock
        spy = MagicMock(wraps=capture)
        def slow_read():
            time.sleep(.4)
            return capture.read()
        spy.read.side_effect = slow_read
        beats = []
        running = True
        def heartbeat():
            if running and not self.app.destroyed:
                beats.append(time.monotonic())
                self.app.after(10, heartbeat)
        with patch("playback_source.cv2.VideoCapture", return_value=spy):
            heartbeat()
            player = Playback(self.app, directory)
            player.withdraw()
            self.players.append(player)
            self.pump(lambda: player.last_index == 0)
            running = False
            player.close()
            player.source.join(3)
        self.assertGreater(len(beats), 10)
        self.assertLess(max(np.diff(beats)), .3)

    def test_close_during_playback_load(self):
        player = self.player()
        player.close()
        self.pump(lambda: not player.source.is_alive())

    def test_cancel_conversion_preserves_recording(self):
        directory = self.path / "recording"
        make_recording(directory)
        self.app.add_recording(directory)
        raw = (directory / "raw.trc").read_bytes()
        output = directory / "conversion"
        output.mkdir()
        self.app.process_log = (directory / "process.log").open("w")
        self.app.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                            stdout=self.app.process_log, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.app.job = (directory, output, "ik", False)
        self.app.cancel_conversion()
        self.pump(lambda: self.app.process is None)
        self.assertEqual(json.loads((output / "result.json").read_text())["status"], "cancelled")
        self.assertEqual((directory / "raw.trc").read_bytes(), raw)
