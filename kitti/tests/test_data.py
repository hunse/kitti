import os

import numpy as np
import matplotlib.pyplot as plt

from kitti.data import get_drives, image_shape, data_dir, Calib
from kitti.raw import load_stereo_frame
from kitti.velodyne import load_velodyne_points


def test_get_drives():
    drives = get_drives()
    print drives


def test_disp2rect():

    drive = 11
    frame = 0
    color = True

    img0, img1 = load_stereo_frame(drive, frame, color=color)

    calib = Calib(color=color)  # get calibration

    # get points
    vpts = load_velodyne_points(drive, frame)

    # remove invalid points
    # m = (vpts[:, 0] >= 5)
    # m = (vpts[:, 0] >= 5) & (np.abs(vpts[:, 1]) < 5)
    m = (vpts[:, 0] >= 5) & (vpts[:, 2] >= -3)
    vpts = vpts[m, :]

    rpts = calib.velo2rect(vpts)

    # get disparities
    xyd = calib.rect2disp(rpts)
    xyd, valid_rpts = calib.filter_disps(xyd, return_mask=True)

    if 1:
        # plot disparities
        disp = np.zeros(image_shape, dtype=np.uint8)
        for x, y, d in np.round(xyd):
            disp[y, x] = d

        plt.figure(101)
        plt.clf()
        plt.subplot(211)
        plt.imshow(img0, cmap='gray')
        plt.subplot(212)
        plt.imshow(disp)
        plt.show()

        # assert False

    # convert back to rect
    rpts2 = calib.disp2rect(xyd)

    assert np.allclose(rpts[valid_rpts], rpts2)

    # plotting
    if 0:
        plt.figure(101)
        plt.clf()
        img0, img1 = load_stereo_frame(drive, frame, color=color)
        plt.imshow(img0, cmap='gray')

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(rpts[:, 0], rpts[:, 1], rpts[:, 2], '.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


if __name__ == '__main__':
    # print data_dir
    test_get_drives()
    # test_disp2rect()
