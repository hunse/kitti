import os

import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(root_dir), 'data')
image_shape = 375, 1242


def get_drive_dir(drive, date='2011_09_26'):
    return os.path.join(data_dir, date, date + '_drive_%04d_sync' % drive)


def get_inds(path, ext='.png'):
    inds = [int(os.path.splitext(name)[0]) for name in os.listdir(path)
            if os.path.splitext(name)[1] == ext]
    inds.sort()
    return inds


def get_calib_dir(date='2011_09_26'):
    return os.path.join(data_dir, date)


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


def homogeneous_transform(points, transform):
    """
    Parameters
    ----------
    points : (n_points, M) array-like
        The points to transform. If `points` is shape (n_points, M-1), a unit
        homogeneous coordinate will be added to make it (n_points, M).
    transform : (M, N) array-like
        The transformation to apply.
    """
    points = np.asarray(points)
    transform = np.asarray(transform)
    n_points, D = points.shape
    M, N = transform.shape

    # do transformation in homogeneous coordinates
    if D == M - 1:
        points = np.hstack([points, np.ones((n_points, 1), dtype=points.dtype)])
    elif D != M:
        raise ValueError("Number of dimensions of points (%d) does not match"
                         "input dimensions of transform (%d)." % (D, M))

    new_points = np.dot(points, transform)

    # normalize homogeneous coordinates
    new_points = new_points[:, :-1] / new_points[:, [-1]]
    return new_points


class Calib(object):
    def __init__(self, date='2011_09_26', color=False):
        self.calib_dir = get_calib_dir(date=date)
        self.imu2velo = read_calib_file(
            os.path.join(self.calib_dir, "calib_imu_to_velo.txt"))
        self.velo2cam = read_calib_file(
            os.path.join(self.calib_dir, "calib_velo_to_cam.txt"))
        self.cam2cam = read_calib_file(
            os.path.join(self.calib_dir, "calib_cam_to_cam.txt"))
        self.color = color

    def get_imu2velo(self):
        RT_imu2velo = np.eye(4)
        RT_imu2velo[:3, :3] = self.imu2velo['R'].reshape(3, 3)
        RT_imu2velo[:3, 3] = self.imu2velo['T']
        return RT_imu2velo.T

    def get_velo2rect(self):
        RT_velo2cam = np.eye(4)
        RT_velo2cam[:3, :3] = self.velo2cam['R'].reshape(3, 3)
        RT_velo2cam[:3, 3] = self.velo2cam['T']

        R_rect00 = np.eye(4)
        R_rect00[:3, :3] = self.cam2cam['R_rect_00'].reshape(3, 3)

        RT_velo2rect = np.dot(R_rect00, RT_velo2cam)
        return RT_velo2rect.T

    def get_rect2disp(self):
        cam0, cam1 = (0, 1) if not self.color else (2, 3)
        P_rect0 = self.cam2cam['P_rect_%02d' % cam0].reshape(3, 4)
        P_rect1 = self.cam2cam['P_rect_%02d' % cam1].reshape(3, 4)

        P0, P1, P2 = P_rect0
        Q0, Q1, Q2 = P_rect1
        assert np.array_equal(P2, Q2) and np.array_equal(P1, Q1)

        # create disp transform
        T = np.array([P0, P1, P0 - Q0, P2])
        return T.T

    def get_imu2rect(self):
        return np.dot(self.get_imu2velo(), self.get_velo2rect())

    def get_imu2disp(self):
        return np.dot(self.get_imu2rect(), self.get_rect2disp())

    def get_disp2rect(self):
        return np.linalg.inv(self.get_rect2disp())

    def get_disp2imu(self):
        return np.linalg.inv(self.get_imu2disp())

    def rect2disp(self, points, valid=True):
        T = self.get_rect2disp()
        xyd = homogeneous_transform(points, T)

        if valid:
            # only take points that fall in the reference image
            x, y, d = xyd.T
            mask = ((x >= 0) & (x <= image_shape[1] - 1) &
                    (y >= 0) & (y <= image_shape[0] - 1) &
                    (d >= 0) & (d <= 255))
            xyd = xyd[mask]
            return xyd, mask
        else:
            return xyd

    def disp2rect(self, xyd):
        return homogeneous_transform(xyd, self.get_disp2rect())

    def velo2rect(self, points):
        return homogeneous_transform(points, self.get_velo2rect())

    def imu2rect(self, points):
        return homogeneous_transform(points, self.get_imu2rect())

    def rect2imu(self, points):
        return homogeneous_transform(points, self.get_rect2imu())



# TODO: functions to automatically download data
