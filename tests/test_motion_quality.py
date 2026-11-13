"""Quality checks without camera input or existing-file mutation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import opensim as osim

from motion_data import MARKER_ORDER, read_trc, remove_isolated_spikes, repair_frames, select_static_segment, write_trc
from recording_io import RecordingWriter
from run_pipeline import MODEL_FILE, PROJECT_DIR, validate_motion
from zed_studio import cleanup_session_data, write_quality_note


class MotionQualityTests(unittest.TestCase):
    def test_isolated_depth_spike_repaired_with_evidence_and_raw_untouched(self):
        frames = [(i / 30, {'LHeel': (i / 100, 0., 2. if i == 3 else 0.)}) for i in range(9)]
        masked, evidence = remove_isolated_spikes(frames, ['LHeel'])
        result, quality = repair_frames(masked, ['LHeel'], max_gap=None)
        self.assertEqual(frames[3][1]['LHeel'][2], 2.)
        self.assertAlmostEqual(result[3][1]['LHeel'][2], 0.)
        self.assertEqual(quality['filled_marker_samples'], 1)
        self.assertEqual(evidence['isolated_spikes'][0]['original'], [.03, 0., 2.])
        json.dumps(evidence, allow_nan=False)

    def test_fast_continuous_motion_and_missing_edges_are_not_smoothed(self):
        for z in ([i*.4 for i in range(9)], [0., 0., .5, .5, .5, 0., 0., 0., 0.]):
            frames = [(i / 30, {'p': (0., 0., p)}) for i, p in enumerate(z)]
            result, evidence = remove_isolated_spikes(frames, ['p'])
            self.assertEqual(result, frames)
            self.assertEqual(evidence['isolated_spike_samples'], 0)
        frames = [(i / 30, {'p': (0., 0., np.nan if i == 0 else 0.)}) for i in range(3)]
        result, _ = remove_isolated_spikes(frames, ['p'])
        with self.assertRaisesRegex(ValueError, '頭尾缺失'):
            repair_frames(result, ['p'])

    def test_model_limits_and_within_range_reversal(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR / '.analysis') as folder:
            path = Path(folder) / 'motion.mot'
            model = osim.Model(str(MODEL_FILE))
            for degrees in (True, False):
                def write(angles):
                    angles = angles if degrees else np.deg2rad(angles)
                    path.write_text(f'Coordinates\nnRows={len(angles)}\nnColumns=2\ninDegrees='+('yes' if degrees else 'no')+
                                    '\nendheader\ntime\tknee_angle_r\n'+
                                    '\n'.join(f'{i/30}\t{v}' for i,v in enumerate(angles))+'\n')
                write([20., 100., 20.])
                before = path.read_bytes()
                result = validate_motion(path, [0., 1/30, 2/30], model)
                self.assertEqual(len(result['continuity_warnings']), 1)
                self.assertEqual(path.read_bytes(), before)
                write([20., 20., 100., 100.])
                result = validate_motion(path, [i / 30 for i in range(4)], model)
                self.assertEqual(len(result['continuity_warnings']), 1)
                self.assertIn('相鄰幀', result['continuity_warnings'][0]['reason'])
                write([20., 160., 20.])
                with self.assertRaisesRegex(ValueError, '超出模型範圍'):
                    validate_motion(path, [0., 1/30, 2/30], model)

    def test_static_selection_splits_at_camera_interruption(self):
        frames = [(t, {'p': (0., 1., 0.)}) for t in (0., .1, .2, .8, .9, 1., 1.1, 1.2, 1.3, 1.4)]
        selected, _ = select_static_segment(frames, ['p'])
        self.assertEqual(selected, frames[3:])
        repair_frames(selected, ['p'])

    def test_trc_rejects_non_increasing_time_before_trimming(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR / '.analysis') as folder:
            path = Path(folder) / 'bad.trc'
            write_trc(path, [(0., {}), (0., {}), (.1, {})], 30, MARKER_ORDER)
            with self.assertRaisesRegex(ValueError, '嚴格遞增'):
                read_trc(path)

    def test_failed_recording_finalization_preserves_video_and_skeleton(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR / '.analysis') as folder:
            session = Path(folder)
            recording = session / '.studio/motion_01'
            recording.mkdir(parents=True)
            writer = RecordingWriter(recording, 'motion', 1, 30, [], (64, 48))
            writer.append(1_000_000_000, np.zeros((48, 64, 3), np.uint8), {}, None, 1)
            with patch('recording_io.write_trc', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    writer.finish()
            with self.assertRaisesRegex(RuntimeError, '未完成保存'):
                writer.finish('disk unavailable')
            self.assertEqual(json.loads((recording / 'recording.json').read_text())['status'], 'recording')
            cleanup_session_data(session)
            archives = list((session / 'diagnostics').iterdir())
            self.assertEqual(len(archives), 1)
            self.assertTrue((archives[0] / 'skeleton.jsonl').stat().st_size)
            self.assertTrue((archives[0] / 'video.mp4').stat().st_size)

    def test_raw_diagnostics_survive_cleanup(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR / '.analysis') as folder:
            session = Path(folder)
            source = session / '.studio/motion_01'
            source.mkdir(parents=True)
            (source / 'raw.trc').write_bytes(b'original raw data')
            (source / 'skeleton.jsonl').write_text('{"time": 0}\n')
            cleanup_session_data(session)
            self.assertFalse((session / '.studio').exists())
            saved = list((session / 'diagnostics').glob('*/raw.trc'))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].read_bytes(), b'original raw data')

    def test_note_discloses_corrected_spike_and_preserved_angles(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR / '.analysis') as folder:
            note = write_quality_note(folder, {'quality': {'isolated_spikes': [
                {'marker': 'LHeel', 'time': .1, 'residual_m': 2.}], 'filled_marker_samples': 1}})
            text = note.read_text(encoding='utf-8-sig')
            self.assertIn('孤立突波：1', text)
            self.assertIn('不事後反號', text)
            self.assertIn('diagnostics', text)

    def test_failed_diagnostic_copy_preserves_raw_input(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR / '.analysis') as folder:
            source = Path(folder) / '.studio/motion_01'
            source.mkdir(parents=True)
            (source / 'raw.trc').write_bytes(b'raw')
            with patch('zed_studio.shutil.copy2', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    cleanup_session_data(folder)
            self.assertEqual((source / 'raw.trc').read_bytes(), b'raw')


if __name__ == '__main__':
    unittest.main()
