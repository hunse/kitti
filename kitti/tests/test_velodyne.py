import os
import numpy as np
import matplotlib.pyplot as plt

from kitti.data import image_shape
from kitti.raw import load_stereo_frame
from kitti.velodyne import (
    load_disparity_points, lin_interp, lstsq_interp, bp_interp, bp_stereo_interp)


def test_load_disparity_points(drive=11, frame=0):
    xyd = load_disparity_points(drive, frame, color=False)
    disp = np.zeros(image_shape, dtype=np.uint8)
    for x, y, d in np.round(xyd):
        disp[y, x] = d

    plt.figure(1)
    plt.clf()
    plt.imshow(disp)
    plt.show()


def test_disparity_with_bm(drive=11, frame=0):
    import cv2

    xyd = load_disparity_points(drive, frame, color=False)
    disp = np.zeros(image_shape, dtype=np.uint8)
    for x, y, d in np.round(xyd):
        disp[y, x] = d

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
    for x, y, d in np.round(xyd):
        if bm_disp[y, x] >= 0:
            diff[y, x] = bm_disp[y, x] - d

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
    plt.imshow(disp)
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(bm_disp)
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(diff2, cmap=cmap, interpolation='nearest')
    plt.colorbar()

    plt.show()


def test_interp(drive=11, frame=0):
    xyd = load_disparity_points(drive, frame, color=False)

    lin_disp = lin_interp(image_shape, xyd)
    lstsq_disp = lstsq_interp(image_shape, xyd, lamb=0.5)
    # lstsq_disp = lstsq_interp(image_shape, points, disps, maxiter=100)

    plt.figure()
    plt.clf()
    plt.subplot(211)
    plt.imshow(lin_disp)
    plt.subplot(212)
    plt.imshow(lstsq_disp)
    plt.show()


def test_bp_interp(drive=11, frame=150):
    xyd = load_disparity_points(drive, frame, color=False)

    disp = np.zeros(image_shape)
    for j, i, d in np.round(xyd):
        disp[i, j] = d

    bp_disp = bp_interp(image_shape, xyd)

    pair = load_stereo_frame(drive, frame)
    bp_disp2 = bp_stereo_interp(pair[0], pair[1], xyd)

    plt.figure()
    plt.clf()
    plt.subplot(411)
    plt.imshow(pair[0], cmap='gray')
    plt.subplot(412)
    plt.imshow(disp)
    plt.subplot(413)
    plt.imshow(bp_disp)
    plt.subplot(414)
    plt.imshow(bp_disp2)
    plt.show()


if __name__ == '__main__':
    # test_load_disparity_points()
    # test_disparity_with_bm()
    # test_interp()
    test_bp_interp()
