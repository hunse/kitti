import os
import numpy as np

from kitti.data import get_drive_dir, get_inds


def get_video_dir(drive, color=False, right=False, date='2011_09_26'):
    drive_dir = get_drive_dir(drive, date)
    image_dir = 'image_%02d' % (0 + (1 if right else 0) + (2 if color else 0))
    return os.path.join(drive_dir, image_dir, 'data')


def get_disp_dir(drive, date='2011_09_26'):
    return os.path.join(
        data_dir, date, date + '_drive_%04d_sync' % drive, 'disp', 'data')


def get_video_images(path, indices, ext='.png'):
    import scipy.ndimage
    images = [
        scipy.ndimage.imread(os.path.join(path, '%010d%s' % (index, ext)))
        for index in indices]
    return images


def get_video_odometry(oxts_path, indices, ext='.txt'):
    data_path = os.path.join(oxts_path, 'data')

    odometry = []
    for index in indices:
        filename = os.path.join(data_path, '%010d%s' % (index, ext))
        with open(filename, 'r') as f:
            line = f.readline().strip('\n')
            odometry.append(map(float, line.split(' ')))

    # get timestamps
    import datetime
    with open(os.path.join(oxts_path, 'timestamps.txt'), 'r') as f:
        parse = lambda s: datetime.datetime.strptime(s[:-3], "%Y-%m-%d %H:%M:%S.%f")
        timestamps = [parse(line.strip('\n')) for line in f.readlines()]
        times = [(t - timestamps[0]).total_seconds() for t in timestamps]

    odometry = np.array(odometry)
    times = np.array(times).reshape(-1, 1)
    return np.concatenate([odometry, times], axis=1)


def odometry_to_positions(odometry):
    lat, lon, alt = odometry.T[:3]

    R = 6378137  # Earth's radius in metres

    if 0:
        # convert to Mercator (based on devkit MATLAB code, untested)
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        scale = np.cos(lat)
        mx = R * scale * lon
        my = R * scale * np.log(np.tan(0.5 * (lat + 0.5 * np.pi)))
    else:
        # convert to metres
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        mx = R * lon * np.cos(lat)
        my = R * lat

    times = odometry.T[-1]
    return np.vstack([mx, my, alt, times]).T


def load_video(drive, **kwargs):
    path = get_video_dir(drive, **kwargs)
    indices = get_inds(path)
    images = get_video_images(path, indices)
    return np.array(images)


def load_stereo_frame(drive, ind, **kwargs):
    left_path = get_video_dir(drive, right=False, **kwargs)
    right_path = get_video_dir(drive, right=True, **kwargs)
    left_images = get_video_images(left_path, [ind])
    right_images = get_video_images(right_path, [ind])
    return np.array(left_images[0]), np.array(right_images[0])


def load_stereo_video(drive, **kwargs):
    left_path = get_video_dir(drive, right=False, **kwargs)
    right_path = get_video_dir(drive, right=True, **kwargs)
    left_inds = get_inds(left_path)
    right_inds = get_inds(right_path)
    assert (np.unique(left_inds) == np.unique(right_inds)).all()

    left_images = get_video_images(left_path, left_inds)
    right_images = get_video_images(right_path, right_inds)
    return np.array(zip(left_images, right_images))


def load_disp_frame(drive, ind, **kwargs):
    path = get_disp_dir(drive, **kwargs)
    images = get_video_images(path, [ind])
    return np.asarray(images[0])


def load_disp_video(drive, **kwargs):
    path = get_disp_dir(drive, **kwargs)
    inds = get_inds(path)
    images = get_video_images(path, inds)
    return np.asarray(images)


def load_video_odometry(drive, raw=False, **kwargs):
    drive_dir = get_drive_dir(drive, **kwargs)
    oxts_dir = os.path.join(drive_dir, 'oxts')
    data_dir = os.path.join(drive_dir, 'oxts', 'data')
    inds = get_inds(data_dir, ext='.txt')
    odometry = get_video_odometry(oxts_dir, inds)
    return odometry if raw else odometry_to_positions(odometry)


def animate_video(video, fig=None, ax=None):
    """
    NOTE: the `ani` variable returned by this function must be referenced at
    the top level of your script, otherwise Python will garbage-collect it
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111)

    def update_image(index, image):
        image.set_data(video[index])
        return image

    image = ax.imshow(video[0], cmap='gray')

    ani = animation.FuncAnimation(
        fig, update_image, len(video), fargs=(image,), interval=100)

    return ani
