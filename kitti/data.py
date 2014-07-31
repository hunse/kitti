import os

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(root_dir), 'data')


def get_drive_dir(drive, date='2011_09_26'):
    return os.path.join(data_dir, date, date + '_drive_%04d_sync' % drive)


def get_inds(path, ext='.png'):
    inds = [int(os.path.splitext(name)[0]) for name in os.listdir(path)
            if os.path.splitext(name)[1] == ext]
    inds.sort()
    return inds


# TODO: functions to automatically download data
