"""Verify grounding on real model geometry, independent of camera hardware."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import opensim as osim

from motion_data import MARKER_ORDER, read_trc, write_trc
from run_pipeline import (MODEL_FILE, MARKER_SET_FILE, PROJECT_DIR, FOOT_MARKERS,
                          apply_ground_clearance, foot_surfaces, execute_job)


class GroundingTests(unittest.TestCase):
    def setUp(self):
        (PROJECT_DIR / ".analysis").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_DIR / ".analysis")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = osim.Model(str(MODEL_FILE))
        self.model.updateMarkerSet(osim.MarkerSet(str(MARKER_SET_FILE)))

    def test_surface_clearance_angles_units_and_airborne_motion(self):
        # Nonuniform mesh scaling exercises the scaled-model path.
        mesh = osim.Mesh.safeDownCast(self.model.getBodySet().get("calcn_r").upd_attached_geometry(0))
        mesh.set_scale_factors(osim.Vec3(1.1, 1.4, .9))
        for degrees in (True, False):
            with self.subTest(degrees=degrees):
                path = self.root / f"motion-{degrees}.mot"
                angles = np.array([0., 25., -10.])
                if not degrees:
                    angles = np.deg2rad(angles)
                original = np.array([[0., 2., 1., -3., angles[0]],
                                     [.1, 2.2, .5, -2.9, angles[1]],
                                     [.2, 2.4, 2.5, -2.8, angles[2]]])
                header = (f"Coordinates\ninDegrees={'yes' if degrees else 'no'}\nendheader\n"
                          "time\tpelvis_tx\tpelvis_ty\tpelvis_tz\thip_flexion_r\n")
                path.write_text(header + '\n'.join('\t'.join(map(str, row)) for row in original) + '\n')
                report = apply_ground_clearance(self.model, path)
                json.dumps(report, allow_nan=False)
                result = np.loadtxt(path, skiprows=4)
                np.testing.assert_allclose(result[:, [0, 4]], original[:, [0, 4]], atol=1e-8)
                np.testing.assert_allclose(np.diff(result[:, [1, 2, 3]], axis=0),
                                           np.diff(original[:, [1, 2, 3]], axis=0), atol=1e-8)
                # Use OpenSim's station transform rather than the production matrix calculation.
                state = self.model.initSystem()
                surfaces = foot_surfaces(self.model)
                minima = []
                for row in result:
                    for n, v in zip(('pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'hip_flexion_r'), row[1:]):
                        self.model.getCoordinateSet().get(n).setValue(
                            state, np.deg2rad(v) if degrees and n == 'hip_flexion_r' else v, False)
                    self.model.realizePosition(state)
                    minima.append(min(frame.findStationLocationInGround(state, osim.Vec3(*map(float, p))).get(1)
                                      for frame, vertices in surfaces for p in vertices))
                self.assertAlmostEqual(minima[0], .005, places=6)
                self.assertLess(minima[1], .005)  # A fixed floor does not chase a noisy low frame.
                self.assertGreater(minima[2], 1.0)  # Does not pin an airborne frame down.
                self.assertAlmostEqual(result[0, 1], 0., places=6)
                self.assertAlmostEqual(result[0, 3], 0., places=6)
                self.assertGreater(result[0, 2], .5)  # Pelvis is not placed at ground height.

    def test_complete_scale_ik_uses_motion_origin_and_serializes_results(self):
        state = self.model.initSystem()
        self.model.realizePosition(state)
        pose = {n: tuple(self.model.getMarkerSet().get(n).getLocationInGround(state).get(i)
                         for i in range(3)) for n in MARKER_ORDER}
        # The base template's three hip markers are collinear. Give both static
        # and motion the same small depth offset to exercise the orientation fit.
        x, y, z = pose['midHip']
        pose['midHip'] = (x + .02, y, z)
        static = self.root / 'static.trc'
        write_trc(static, [(i / 10, pose.copy()) for i in range(11)], 10, MARKER_ORDER)
        scaled = execute_job('scale', static, self.root / 'scale')
        self.assertEqual(scaled['status'], 'complete', scaled.get('error'))
        motion = self.root / 'motion.trc'
        frames = [(i / 10, {n: (x + 2 + i / 100, y - .4, z - 3) for n, (x, y, z) in pose.items()})
                  for i in range(6)]
        write_trc(motion, frames, 10, MARKER_ORDER)
        source = motion.read_bytes()
        output = self.root / 'ik'
        with patch('run_pipeline.apply_front_facing_sagittal_correction', create=True,
                   side_effect=AssertionError('Do not rewrite IK angles')):
            result = execute_job('ik', motion, output, self.root / 'scale' / 'calibration.json')
        self.assertEqual(result['status'], 'complete', result.get('error'))
        self.assertEqual(motion.read_bytes(), source)
        self.assertEqual(json.loads((output / 'result.json').read_text(encoding='utf8'))['status'], 'complete')
        snapshot = json.loads((output / 'calibration.json').read_text(encoding='utf8'))
        self.assertEqual((output / snapshot['model']).read_bytes(), Path(scaled['model']).read_bytes())
        _, _, aligned = read_trc(output / 'aligned.trc')
        self.assertAlmostEqual(aligned[0][1]['midHip'][0], 0, places=5)
        self.assertAlmostEqual(aligned[0][1]['midHip'][2], 0, places=5)
        self.assertAlmostEqual(aligned[-1][1]['midHip'][0], .05, places=5)
        self.assertAlmostEqual(min(aligned[0][1][n][1] for n in FOOT_MARKERS), .005, places=5)
        _, _, published = read_trc(output / 'processed.trc')
        shift = result['ground_correction']['initial_translation_m']
        for (_, before), (_, after) in zip(aligned, published):
            for name in MARKER_ORDER:
                np.testing.assert_allclose(after[name], np.array(before[name]) + shift, atol=1e-6)
        self.assertEqual(result['ground_correction']['method'],
                         'fixed translation from initial scaled foot meshes')
        self.assertLess(result['marker_errors']['mean_rms_m'], .02)

    def test_incompatible_ankle_and_heel_targets_stop_before_ik(self):
        state = self.model.initSystem()
        self.model.realizePosition(state)
        markers = self.model.getMarkerSet()
        pose = {name: tuple(markers.get(name).getLocationInGround(state).get(i) for i in range(3))
                for name in MARKER_ORDER}
        bad_pose = pose.copy()
        x, y, z = pose['LAnkle']
        bad_pose['LAnkle'] = (x, y + .8, z)
        # The outlier lasts multiple frames and bypasses isolated spike removal.
        frames = [(i / 30, bad_pose if 2 <= i <= 5 else pose) for i in range(8)]
        source = self.root / 'conflicting.trc'
        write_trc(source, frames, 30, MARKER_ORDER)
        self.model.printToXML(str(self.root / 'model.osim'))
        calibration = self.root / 'calibration.json'
        calibration.write_text(json.dumps({'model': 'model.osim'}), encoding='utf-8')
        output = self.root / 'blocked'
        with patch('run_pipeline.osim.InverseKinematicsTool') as solver:
            result = execute_job('ik', source, output, calibration)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('左踝與腳跟', result['error'])
        self.assertIn('縮放模型可達上限', result['error'])
        solver.assert_not_called()
        self.assertFalse((output / 'motion.mot').exists())
