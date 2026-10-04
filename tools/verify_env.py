import numpy
import scipy
import pandas
import matplotlib
import cv2
import OpenGL
import opensim as osim
import pyzed.sl as sl

print("numpy    ", numpy.__version__)
print("scipy    ", scipy.__version__)
print("pandas   ", pandas.__version__)
print("matplotlib", matplotlib.__version__)
print("opencv   ", cv2.__version__)
print("pyopengl ", OpenGL.__version__)
print("opensim  ", osim.GetVersion())
print("zed sdk  ", sl.Camera().get_sdk_version())

devices = sl.Camera.get_device_list()
print("ZED cameras detected:", len(devices))
for d in devices:
    print(" -", d.camera_model, d.serial_number)
