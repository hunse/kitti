from kitti.stereo import load_pair, load_disp


def test_load():
    import matplotlib.pyplot as plt

    i = 90
    left, right = load_pair(i)
    disp = load_disp(i)

    plt.figure()
    plt.subplot(311)
    plt.imshow(left, cmap='gray')
    plt.subplot(312)
    plt.imshow(right, cmap='gray')
    plt.subplot(313)
    plt.imshow(disp)
    plt.show()


if __name__ == '__main__':
    test_load()
