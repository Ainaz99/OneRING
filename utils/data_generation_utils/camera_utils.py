import math
import numpy as np

def calc_camera_intrinsics(fov_y, frame_height, frame_width):
    # this functionality is now here to avoid a circularity or duplication issue
    focal_length = 0.5 * frame_height / math.tan(math.radians(fov_y / 2))
    f_x = f_y = focal_length

    c_x = frame_width / 2
    c_y = frame_height / 2
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    return K