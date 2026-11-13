"""Formula checks against known vectors and OpenSim forward kinematics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import opensim as osim

from capture_to_trc import to_opensim_axes
from run_pipeline import PROJECT_DIR, MODEL_FILE, MARKER_SET_FILE, pelvis_tilt_reference


class ConversionMathTests(unittest.TestCase):
    def model(self):
        model = osim.Model(str(MODEL_FILE))
        model.updateMarkerSet(osim.MarkerSet(str(MARKER_SET_FILE)))
        # Synthetic non-collinear marker triangle with known body-frame geometry.
        for name, point in [('RHip', (-.04,-.06,.08)), ('LHip',(-.04,-.06,-.08)),
                            ('midHip',(-.04,-.01,0.))]:
            model.getMarkerSet().get(name).set_location(osim.Vec3(*point))
        return model

    def test_axes_preserve_handedness_lengths_and_inverse(self):
        rotation = np.array([to_opensim_axes(*v) for v in np.eye(3)]).T
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        # Subject faces the camera: forward is +Zzed; subject-right is -Xzed.
        self.assertEqual(to_opensim_axes(0,0,1), (1,0,0))
        self.assertEqual(to_opensim_axes(-1,0,0), (0,0,1))
        rng=np.random.default_rng(14)
        for point in rng.normal(size=(100,3)):
            converted=np.array(to_opensim_axes(*point))
            np.testing.assert_allclose(rotation.T @ converted, point, atol=1e-14)
            self.assertAlmostEqual(np.linalg.norm(converted), np.linalg.norm(point))

    def test_pelvis_tilt_sign_and_mixed_rotations_against_model(self):
        model=self.model()
        state=model.initSystem()
        poses=[(0,0,0),(25,0,0),(-25,0,0),(15,20,35),(-20,-25,-40),(35,70,50)]
        frames=[]
        for i,(tilt,listing,yaw) in enumerate(poses):
            for name,value in zip(('pelvis_tilt','pelvis_list','pelvis_rotation'),(tilt,listing,yaw)):
                model.getCoordinateSet().get(name).setValue(state,np.deg2rad(value),False)
            for name,value in zip(('pelvis_tx','pelvis_ty','pelvis_tz'),(2.3,1.2,-.7)):
                model.getCoordinateSet().get(name).setValue(state,value,False)
            model.realizePosition(state)
            points={}
            for name in ('RHip','LHip','midHip'):
                point=model.getMarkerSet().get(name).getLocationInGround(state)
                points[name]=tuple(point.get(j) for j in range(3))
            frames.append((i/30,points))
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR/'.analysis') as folder:
            path=Path(folder)/'reference.mot'
            pelvis_tilt_reference(model,frames,path)
            data=np.loadtxt(path,skiprows=6)
            np.testing.assert_allclose(data[:,1],[p[0] for p in poses],atol=1e-6)

    def test_coupled_knee_coordinate_is_not_a_degree_column(self):
        model=self.model()
        state=model.initSystem()
        model.getCoordinateSet().get('knee_angle_r').setValue(state,np.deg2rad(60),True)
        primary=model.getCoordinateSet().get('knee_angle_r').getValue(state)
        dependent=model.getCoordinateSet().get('knee_angle_r_beta')
        self.assertEqual(dependent.getMotionType(),osim.Coordinate.Coupled)
        self.assertAlmostEqual(dependent.getValue(state),primary,places=6)
        # A textual beta value ~1.047 in an inDegrees MOT is not 1.047 degrees.
        self.assertAlmostEqual(dependent.getValue(state),np.pi/3,places=6)
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR/'.analysis') as folder:
            path=Path(folder)/'radians.mot'
            path.write_text('Coordinates\nnRows=1\nnColumns=3\ninDegrees=no\nendheader\n'
                            'time\tknee_angle_r\tknee_angle_r_beta\n'
                            f'0\t{np.pi/3}\t{np.pi/3}\n')
            storage=osim.Storage(str(path))
            model.getSimbodyEngine().convertRadiansToDegrees(storage)
            data=storage.getStateVector(0).getData()
            self.assertAlmostEqual(data.get(0),60.,places=6)
            self.assertAlmostEqual(data.get(1),np.pi/3,places=6)

    def test_model_reference_collinearity_is_rejected(self):
        model=self.model()
        model.getMarkerSet().get('midHip').set_location(osim.Vec3(-.04,-.06,0.))
        good={'RHip':(-.04,-.06,.08),'LHip':(-.04,-.06,-.08),'midHip':(-.04,-.01,0.)}
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR/'.analysis') as folder:
            with self.assertRaisesRegex(ValueError,'模型骨盆'):
                pelvis_tilt_reference(model,[(i/30,good) for i in range(6)],Path(folder)/'ref.mot')


if __name__=='__main__':
    unittest.main()
