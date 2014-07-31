import os

from kitti.data import data_dir


def load_image(index, test=False, right=False, future=False):
    import scipy.ndimage
    path = os.path.join(
        data_dir,
        'data_stereo_flow',
        'testing' if test else 'training',
        'image_1' if right else 'image_0',
        "%06d_%2d.png" % (index, 11 if future else 10))
    return scipy.ndimage.imread(path)


def load_pair(index, test=False, future=False):
    left = load_image(index, right=False, test=test, future=future)
    right = load_image(index, right=True, test=test, future=future)
    return left, right


def load_disp(index, test=False, occluded=False):
    assert not test, "disparity not available for test data"

    import scipy.ndimage
    path = os.path.join(
        data_dir,
        'data_stereo_flow',
        'testing' if test else 'training',
        'disp_occ' if occluded else 'disp_noc',
        "%06d_10.png" % (index))
    return scipy.ndimage.imread(path)
