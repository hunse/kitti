import os

import numpy as np


def read_calib_file(path):
    float_chars = set("0123456789.e+- ")

    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    values = map(float, value.split(' '))
                    data[key] = np.array(values)
                except ValueError:
                    pass  # casting error is fine, b/c data[key] == value

    return data


def make_disparity_image(calib_dir, data_dir, frame):

    # read calibration data
    calib = read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))

    # read velodyne points
    points_path = os.path.join(data_dir, "%010d.bin" % frame)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)


if __name__ == '__main__':
    calib_dir = '/home/ehunsber/workspace/kitti/data/2011_09_26'
    data_dir = '/home/ehunsber/workspace/kitti/data/2011_09_26/2011_09_26_drive_0011_sync/velodyne_points/data'

    make_disparity_image(calib_dir, data_dir, 0)
