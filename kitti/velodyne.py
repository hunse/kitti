import os

import numpy as np

from kitti.data import get_drive_dir, get_calib_dir, get_inds

image_shape = 375, 1242


def get_velodyne_dir(drive, **kwargs):
    drive_dir = get_drive_dir(drive, **kwargs)
    return os.path.join(drive_dir, 'velodyne_points', 'data')


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


def get_disparity_points(calib_dir, data_dir, frame, color=False):

    cam0, cam1 = (0, 1) if not color else (2, 3)

    # read calibration data
    cam2cam = read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
    rigid = read_calib_file(os.path.join(calib_dir, "calib_velo_to_cam.txt"))

    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = rigid['R'].reshape(3, 3)
    Tr_velo2cam[:3, 3] = rigid['T']

    R_rect00 = np.eye(4)
    R_rect00[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)

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
    assert (np.abs(diff[:, 1]) < 0.5).all()  # assert all y-coordinates close

    points = points0[:, ::-1]  # points in left image, flip to (i, j) format
    disps = diff[:, 0]

    # take only points that fall in the first image
    mask = ((points[:, 0] >= 0) & (points[:, 0] <= image_shape[0] - 1) &
            (points[:, 1] >= 0) & (points[:, 1] <= image_shape[1] - 1) &
            (disps >= 0) & (disps <= 255))

    points = points[mask]
    disps = disps[mask]
    return points, disps


def lin_interp(shape, points_ij, values):
    from scipy.interpolate import LinearNDInterpolator

    m, n = shape
    f = LinearNDInterpolator(points_ij, values, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


def lstsq_interp(shape, points_ij, values, lamb=1, maxiter=None, valid=True):
    import scipy.sparse
    import scipy.sparse.linalg

    if valid:
        # clip out the valid region, and call recursively
        I, J = points_ij.T
        i0, i1, j0, j1 = I.min(), I.max(), J.min(), J.max()
        subpoints = np.array([I - i0, J - j0]).T

        output = -np.ones(shape)
        suboutput = output[i0:i1+1, j0:j1+1]
        subshape = suboutput.shape

        suboutput[:] = lstsq_interp(subshape, subpoints, values,
                                    lamb=lamb, maxiter=maxiter, valid=False)
        return output

    n_pixels = np.prod(shape)
    n_points = points_ij.shape[0]
    assert points_ij.ndim == 2 and points_ij.shape[1] == 2

    Cmask = np.zeros(shape, dtype=bool)
    m = np.zeros(shape)
    for [i, j], v in zip(points_ij, values):
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

    # x0 = np.zeros(shape)
    x0 = lin_interp(shape, points_ij, values)

    x, info = scipy.sparse.linalg.cg(G, m.flatten(), x0=x0.flatten(),
                                     maxiter=maxiter)

    return x.reshape(shape)


def create_disparity_video(drive, color=False, **kwargs):
    import scipy.misc
    from kitti.raw import get_disp_dir

    disp_dir = get_disp_dir(drive, **kwargs)
    if os.path.exists(disp_dir):
        raise RuntimeError("Target directory already exists. "
                           "Please delete '%s' and re-run." % disp_dir)

    calib_dir = get_calib_dir(**kwargs)
    velodyne_dir = get_velodyne_dir(drive, **kwargs)
    inds = get_inds(velodyne_dir, ext='.bin')

    os.makedirs(disp_dir)
    for i in inds:
        points, disps = get_disparity_points(
            calib_dir, velodyne_dir, i, color=color)
        disp = lstsq_interp(image_shape, points, disps)

        disp[disp < 0] = 0
        disp = disp.astype('uint8')

        path = os.path.join(disp_dir, '%010d.png' % i)
        scipy.misc.imsave(path, disp)
        print "Created disp image %d" % i
