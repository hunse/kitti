import os
import numpy as np
import matplotlib.pyplot as plt

from kitti.data import get_calib_dir
from kitti.raw import load_stereo_frame
from kitti.velodyne import (
    get_velodyne_dir, get_disparity_points, image_shape,
    lin_interp, lstsq_interp)


def test_get_disparity_points(drive=11, frame=0):
    calib_dir = get_calib_dir()
    velodyne_dir = get_velodyne_dir(drive)

    # create sparse disparity image
    points, disps = get_disparity_points(
        calib_dir, velodyne_dir, frame, color=False)

    disparity = np.zeros(image_shape, dtype=np.uint8)
    for [i, j], d in zip(np.round(points), np.round(disps)):
        disparity[i, j] = d

    plt.figure(1)
    plt.clf()
    plt.imshow(disparity)
    plt.show()


def test_disparity_with_bm(drive=11, frame=0):
    import cv2

    calib_dir = get_calib_dir()
    velodyne_dir = get_velodyne_dir(drive)

    # create sparse disparity image
    points, disps = get_disparity_points(
        calib_dir, velodyne_dir, frame, color=False)

    disparity = np.zeros(image_shape, dtype=np.uint8)
    for [i, j], d in zip(np.round(points), np.round(disps)):
        disparity[i, j] = d

    # compare with block matching
    left, right = load_stereo_frame(drive, frame, color=False)

    ndisp = 128
    bm = cv2.createStereoBM(numDisparities=ndisp, blockSize=9)
    bm.setPreFilterSize(41)
    bm.setPreFilterCap(31)
    bm.setTextureThreshold(20)
    bm.setUniquenessRatio(10)
    bm.setSpeckleWindowSize(100)
    bm.setSpeckleRange(32)

    bm_disp = bm.compute(left, right) / 16

    # compute difference
    diff = np.zeros(image_shape)
    for [i, j], d in zip(np.round(points), np.round(disps)):
        if bm_disp[i, j] >= 0:
            diff[i, j] = bm_disp[i, j] - d

    # downsample for visualization purposes
    diff2 = np.zeros((image_shape[0] / 2, image_shape[1] / 2))
    for i in range(diff2.shape[0]):
        for j in range(diff2.shape[1]):
            r = diff[2*i:2*i+2, 2*j:2*j+2].flatten()
            ind = np.argmax(np.abs(r))
            diff2[i, j] = r[ind]

    # custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    g = 1.0
    cdict1 = {
        'red':   ((0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 1.0, 1.0)),
        'green': ((0.0, g, g),
                  (0.5, 0.0, 0.0),
                  (1.0, g, g)),
        'blue':  ((0.0, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),
    }
    cmap = LinearSegmentedColormap('BlueRed', cdict1)

    plt.figure(1)
    plt.clf()
    plt.subplot(311)
    plt.imshow(disparity)
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(bm_disp)
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(diff2, cmap=cmap, interpolation='nearest')
    plt.colorbar()

    plt.show()


def test_interp(drive=11, frame=0):
    calib_dir = get_calib_dir()
    velodyne_dir = get_velodyne_dir(drive)
    points, disps = get_disparity_points(
        calib_dir, velodyne_dir, frame, color=False)
    lin_disp = lin_interp(image_shape, points, disps)
    lstsq_disp = lstsq_interp(image_shape, points, disps)

    plt.figure()
    plt.clf()
    plt.subplot(211)
    plt.imshow(lin_disp)
    plt.subplot(212)
    plt.imshow(lstsq_disp)
    plt.show()


if __name__ == '__main__':
    # test_get_disparity_points()
    # test_disparity_with_bm()
    test_interp()
