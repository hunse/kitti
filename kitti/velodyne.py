import os

import numpy as np

from kitti.data import get_drive_dir, Calib, get_inds, image_shape, get_calib_dir


def get_velodyne_dir(drive, **kwargs):
    drive_dir = get_drive_dir(drive, **kwargs)
    return os.path.join(drive_dir, 'velodyne_points', 'data')


def load_velodyne_points(drive, frame, **kwargs):
    velodyne_dir = get_velodyne_dir(drive, **kwargs)
    points_path = os.path.join(velodyne_dir, "%010d.bin" % frame)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    return points


def load_disparity_points(drive, frame, color=False, **kwargs):

    calib = Calib(color=color, **kwargs)

    # read velodyne points
    points = load_velodyne_points(drive, frame, **kwargs)

    # remove all points behind image plane (approximation)
    points = points[points[:, 0] >= 1, :]

    # convert points to each camera
    xyd = calib.velo2disp(points)

    # take only points that fall in the first image
    xyd = calib.filter_disps(xyd)

    return xyd


def lin_interp(shape, xyd):
    from scipy.interpolate import LinearNDInterpolator

    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


def lstsq_interp(shape, xyd, lamb=1, maxiter=None, valid=True):
    import scipy.sparse
    import scipy.sparse.linalg

    assert xyd.ndim == 2 and xyd.shape[1] == 3

    if valid:
        # clip out the valid region, and call recursively
        j, i, d = xyd.T
        i0, i1, j0, j1 = i.min(), i.max(), j.min(), j.max()
        subpoints = xyd - [[j0, i0, 0]]

        output = -np.ones(shape)
        suboutput = output[i0:i1+1, j0:j1+1]
        subshape = suboutput.shape

        suboutput[:] = lstsq_interp(subshape, subpoints,
                                    lamb=lamb, maxiter=maxiter, valid=False)
        return output

    Cmask = np.zeros(shape, dtype=bool)
    m = np.zeros(shape)
    for j, i, d in xyd:
        Cmask[i, j] = 1
        m[i, j] = d

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

    n_pixels = np.prod(shape)
    G = scipy.sparse.linalg.LinearOperator(
        (n_pixels, n_pixels), matvec=calcAA, dtype=np.float)

    # x0 = np.zeros(shape)
    x0 = lin_interp(shape, xyd)

    x, info = scipy.sparse.linalg.cg(G, m.flatten(), x0=x0.flatten(),
                                     maxiter=maxiter)

    return x.reshape(shape)


def bp_interp(image_shape, xyd):
    from kitti.bp import interp

    seed = np.zeros(image_shape, dtype='uint8')
    for x, y, d in np.round(xyd):
        seed[y, x] = d

    # import matplotlib.pyplot as plt
    # plt.figure(101)
    # plt.hist(seed.flatten(), bins=30)
    # # plt.imshow(seed)
    # plt.show()

    # disp = interp(seed, values=seed.max(), seed_weight=1000, seed_max=10000)
    disp = interp(seed, values=seed.max(), seed_weight=10, disc_max=5)

    return disp


def bp_stereo_interp(img0, img1, xyd):
    from kitti.bp import stereo

    assert img0.shape == img1.shape

    seed = np.zeros(img0.shape, dtype='uint8')
    for x, y, d in np.round(xyd):
        seed[y, x] = d

    params = dict(values=seed.max(), levels=6, min_level=1,
                  disc_max=30, seed_weight=1, data_weight=0.01, data_max=100)
    # disp = stereo(img0, img1, seed, values=128,
    disp = stereo(img0, img1, seed, **params)

    return disp


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
