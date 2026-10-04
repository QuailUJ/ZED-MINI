"""
End-to-end smoke test for the ZED -> .trc -> OpenSim Scale -> IK pipeline,
using synthetic marker data (a static standing pose) instead of a real
camera capture. This only proves the plumbing works; it says nothing about
real motion accuracy.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import opensim as osim

from motion_data import write_trc, MARKER_ORDER

PROJECT_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_DIR / "opensimPipeline"
TEST_DIR = PROJECT_DIR / "test_data"
TEST_DIR.mkdir(exist_ok=True)

GEOMETRY_DIR = PIPELINE_DIR / "Geometry"
osim.ModelVisualizer.addDirToGeometrySearchPaths(str(GEOMETRY_DIR))


def rel(path):
    # OpenSim's XML path resolution mishandles Windows drive-letter absolute
    # paths (it prepends the setup file's own directory to them), so every
    # path written into a setup XML must be relative to that XML's directory.
    return os.path.relpath(path, TEST_DIR)

MODEL_FILE = PIPELINE_DIR / "Models" / "LaiUhlrich2022.osim"
MARKER_SET_FILE = PIPELINE_DIR / "Models" / "LaiUhlrich2022_markers_openpose.xml"
SCALING_TEMPLATE = PIPELINE_DIR / "Scaling" / "Setup_scaling_LaiUhlrich2022_openpose.xml"
IK_TEMPLATE = PIPELINE_DIR / "IK" / "Setup_IK_openpose.xml"

TRC_FILE = TEST_DIR / "static_pose.trc"
SCALING_XML = TEST_DIR / "Setup_scaling_test.xml"
IK_XML = TEST_DIR / "Setup_IK_test.xml"
SCALED_MODEL = TEST_DIR / "scaled_model.osim"
SCALE_OUTPUT_MOTION = TEST_DIR / "static_pose_markerplacer.mot"
SCALE_OUTPUT_MARKERS = TEST_DIR / "static_pose_markers.xml"
IK_OUTPUT_MOTION = TEST_DIR / "ik_test.mot"

# Approximate standing (A-pose) marker positions in meters, right-handed Y-up,
# X forward, Z to the subject's right. Purely synthetic, just needs to be a
# geometrically plausible human pose so Scale/IK have something sane to chew on.
STATIC_POSE = {
    "Neck": (0.0, 1.50, 0.0),
    "RShoulder": (0.0, 1.43, 0.18),
    "LShoulder": (0.0, 1.43, -0.18),
    "RHip": (0.0, 0.92, 0.09),
    "LHip": (0.0, 0.92, -0.09),
    "midHip": (0.0, 0.92, 0.0),
    "RKnee": (0.0, 0.47, 0.09),
    "LKnee": (0.0, 0.47, -0.09),
    "RAnkle": (0.0, 0.08, 0.09),
    "LAnkle": (0.0, 0.08, -0.09),
    "RHeel": (-0.05, 0.02, 0.09),
    "LHeel": (-0.05, 0.02, -0.09),
    "RSmallToe": (0.16, 0.02, 0.11),
    "LSmallToe": (0.16, 0.02, -0.11),
    "RBigToe": (0.17, 0.02, 0.06),
    "LBigToe": (0.17, 0.02, -0.06),
    "RElbow": (0.05, 1.15, 0.30),
    "LElbow": (0.05, 1.15, -0.30),
    "RWrist": (0.05, 0.88, 0.38),
    "LWrist": (0.05, 0.88, -0.38),
}


def make_synthetic_trc():
    frames = [(0.0, STATIC_POSE), (0.1, STATIC_POSE)]
    write_trc(TRC_FILE, frames, fps=30, marker_order=MARKER_ORDER)
    print(f"Wrote synthetic static-pose trc to {TRC_FILE}")


def set_text(root, path, value):
    el = root.find(path)
    if el is None:
        raise RuntimeError(f"XML element not found: {path}")
    el.text = str(value)


def build_scaling_xml():
    tree = ET.parse(SCALING_TEMPLATE)
    root = tree.getroot().find("ScaleTool")

    set_text(root, "GenericModelMaker/model_file", rel(MODEL_FILE))
    set_text(root, "GenericModelMaker/marker_set_file", rel(MARKER_SET_FILE))
    set_text(root, "ModelScaler/marker_file", rel(TRC_FILE))
    set_text(root, "ModelScaler/time_range", "0.0 0.1")
    set_text(root, "ModelScaler/output_model_file", rel(SCALED_MODEL))
    set_text(root, "MarkerPlacer/marker_file", rel(TRC_FILE))
    set_text(root, "MarkerPlacer/time_range", "0.0 0.1")
    set_text(root, "MarkerPlacer/output_motion_file", rel(SCALE_OUTPUT_MOTION))
    set_text(root, "MarkerPlacer/output_model_file", rel(SCALED_MODEL))
    set_text(root, "MarkerPlacer/output_marker_file", rel(SCALE_OUTPUT_MARKERS))

    tree.write(SCALING_XML, encoding="UTF-8", xml_declaration=True)
    print(f"Wrote scaling setup to {SCALING_XML}")


def build_ik_xml():
    tree = ET.parse(IK_TEMPLATE)
    root = tree.getroot().find("InverseKinematicsTool")

    set_text(root, "model_file", rel(SCALED_MODEL))
    set_text(root, "marker_file", rel(TRC_FILE))
    set_text(root, "time_range", "0.0 0.1")
    set_text(root, "output_motion_file", rel(IK_OUTPUT_MOTION))
    set_text(root, "results_directory", ".")

    tree.write(IK_XML, encoding="UTF-8", xml_declaration=True)
    print(f"Wrote IK setup to {IK_XML}")


def run_scaling():
    print("\n--- Running ScaleTool ---")
    tool = osim.ScaleTool(SCALING_XML.name)
    tool.run()
    print(f"Scaled model written to {SCALED_MODEL}")


def run_ik():
    print("\n--- Running InverseKinematicsTool ---")
    tool = osim.InverseKinematicsTool(IK_XML.name)
    tool.run()
    print(f"IK motion written to {IK_OUTPUT_MOTION}")


if __name__ == "__main__":
    make_synthetic_trc()
    build_scaling_xml()
    build_ik_xml()
    os.chdir(TEST_DIR)
    run_scaling()
    run_ik()
    print("\nPipeline smoke test complete.")
