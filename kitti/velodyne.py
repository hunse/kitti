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
                # try to cast to float array
                try:
                    data[key] = np.array(map(float, value.split(' ')))
                except ValueError:
                    pass  # casting error: data[key] already eq. value, so pass

    return data


def lin_interp(shape, ij_points, values):
    from scipy.interpolate import LinearNDInterpolator

    m, n = shape
    f = LinearNDInterpolator(ij_points, values, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


def lstsq_interp(shape, ij_points, values, lamb=1):
    import scipy.sparse
    import scipy.sparse.linalg

    n_pixels = np.prod(shape)
    n_points = ij_points.shape[0]
    assert ij_points.ndim == 2 and ij_points.shape[1] == 2

    Cmask = np.zeros(shape, dtype=bool)
    m = np.zeros(shape)
    for [i, j], v in zip(ij_points, values):
        Cmask[i, j] = 1
        m[i, j] = v

    def calcAA(x):
        x = x.reshape(shape)
        y = np.zeros_like(x)

        # --- smoothness constraints

        #  L = [[1 -1 0 ...], [0 1 -1 ...], ...] (horizontal first-order)
        Lx = -np.diff(x, axis=1)
        y[:,0] += Lx[:,0]
        y[:,-1] -= Lx[:,-1]
        y[:,1:-1] += np.diff(Lx, axis=1)

        #  T = [[1 0 0 ...], [-1 1 0 ...], [0 -1 0 ...], ...] (vert. 1st-order)
        Tx = -np.diff(x, axis=0)
        y[0,:] += Tx[0,:]
        y[-1,:] -= Tx[-1,:]
        y[1:-1,:] += np.diff(Tx, axis=0)

        y *= lamb

        # --- measurement constraints
        y[Cmask] += x[Cmask]

        return y.flatten()

    G = scipy.sparse.linalg.LinearOperator(
        (n_pixels, n_pixels), matvec=calcAA, dtype=np.float)

    x0 = lin_interp(shape, ij_points, values)
    # x0 = np.zeros(shape)

    # x, info = scipy.sparse.linalg.cg(G, m.flatten(), x0=x0.flatten(), maxiter=100)
    x, info = scipy.sparse.linalg.cg(G, m.flatten(), x0=x0.flatten())

    return x.reshape(shape)


def make_disparity_image(calib_dir, data_dir, frame):

    cam0 = 0
    cam1 = 1

    # read calibration data
    cam2cam = read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
    rigid = read_calib_file(os.path.join(calib_dir, "calib_velo_to_cam.txt"))

    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = rigid['R'].reshape(3, 3)
    Tr_velo2cam[:3, 3] = rigid['T']


    R_rect00 = np.eye(4)
    R_rect00[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    # R[:3, :3] = cam2cam['R_rect_%02d' % cam].reshape(3, 3)

    def get_cam_transform(cam):
        P = cam2cam['P_rect_%02d' % cam].reshape(3, 4)
        return np.dot(P, np.dot(R_rect00, Tr_velo2cam))

    P_velo2img0 = get_cam_transform(cam0)
    P_velo2img1 = get_cam_transform(cam1)

    # read velodyne points
    points_path = os.path.join(data_dir, "%010d.bin" % frame)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance

    # remove all points behind image plane (approximation)
    points = points[points[:, 0] >= 5, :]

    # convert points to each camera
    def project(points, T):
        dim_norm, dim_proj = T.shape

        # do transformation in homogeneous coordinates
        if points.shape[1] < dim_proj:
            points = np.hstack([points, np.ones((points.shape[0], 1))])

        new_points = np.dot(points, T.T)

        # normalize homogeneous coordinates
        new_points = new_points[:, :dim_norm-1] / new_points[:, [dim_norm-1]]
        return new_points

    points0 = project(points, P_velo2img0)
    points1 = project(points, P_velo2img1)
    diff = points0 - points1
    assert (np.abs(diff[:, 1]) < 1e-3).all()  # assert all y-coordinates close

    # take only points that fall in the first image
    w, h = 1242, 375
    mask = ((points0[:, 0] >= 0) & (points0[:, 0] <= w-1) &
            (points0[:, 1] >= 0) & (points0[:, 1] <= h-1) &
            (diff[:, 0] >= 0) & (diff[:, 0] <= 255))
    points0 = points0[mask]
    diff = diff[mask]

    import matplotlib.pyplot as plt
    plt.ion()

    if 0:
        # create sparse disparity image
        disparity = np.zeros((h, w), dtype=np.uint8)
        points0 = np.round(points0).astype('int')
        disps = np.round(diff[:, 0])
        for [j, i], d in zip(points0, disps):
            disparity[i, j] = d

        plt.figure(1)
        plt.clf()
        plt.imshow(disparity)

    elif 0:
        # linearly interpolate disparity image
        disparity = lin_interp((h, w), points0[:, ::-1], diff[:, 0])

        plt.figure(2)
        plt.clf()
        plt.imshow(disparity)

    else:
        # lstsq interpolation of disparity image
        disparity = lstsq_interp((h, w), np.round(points0[:, ::-1]), diff[:, 0], lamb=1)

        plt.figure(3)
        plt.clf()
        plt.imshow(disparity)


    return disparity


if __name__ == '__main__':
    calib_dir = '/home/ehunsber/workspace/kitti/data/2011_09_26'
    data_dir = '/home/ehunsber/workspace/kitti/data/2011_09_26/2011_09_26_drive_0011_sync/velodyne_points/data'

    make_disparity_image(calib_dir, data_dir, 150)
