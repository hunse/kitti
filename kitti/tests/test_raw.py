from kitti.raw import load_video, load_stereo_video, load_video_odometry, odometry_to_positions


def test_load_video(interactive=False):
    import matplotlib.pyplot as plt

    video = load_video(11, right=False, color=True)

    if interactive:
        plt.ion()
        plt.figure()
        visimage = plt.imshow(video[0])

        for frame in video:
            visimage.set_data(frame)
            plt.draw()
    else:
        plt.figure()
        inds = [0, 50, 100, 150]
        for i, frame in enumerate(inds):
            plt.subplot(len(inds), 1, i+1)
            plt.imshow(video[frame])
        plt.show()


def test_load_stereo_video(color=False):
    import matplotlib.pyplot as plt

    video = load_stereo_video(11, color=color)

    plt.figure()
    inds = [0, 50, 100, 150]
    args = dict() if color else dict(cmap='gray')
    for i, frame in enumerate(inds):
        plt.subplot(len(inds), 2, 2*i+1)
        plt.imshow(video[frame][0], **args)
        plt.subplot(len(inds), 2, 2*i+2)
        plt.imshow(video[frame][1], **args)
    plt.show()


def test_load_video_odometry():
    import numpy as np
    import matplotlib.pyplot as plt

    dt = 0.1035

    odometry = load_video_odometry(11, raw=True)
    t = odometry.T[-1]
    dts = np.diff(t)
    print (t[-1] - t[0]) / (len(t) - 1)
    print dts.mean()

    positions = odometry_to_positions(odometry)
    x, y = positions.T[:2]
    x = x - x[0]
    y = y - y[0]

    vn, ve  = odometry.T[6:8]  # velocity north and east
    x2 = np.cumsum(ve) * dt
    y2 = np.cumsum(vn) * dt

    yaw, vf, vl = odometry.T[[5, 8, 9]]
    # vn3 = np.zeros_like(vn)
    # ve3 = np.zeros_like(ve)
    trap = lambda x: 0.5 * (x[:-1] + x[1:])
    yaw2, vf2, vl2 = trap(yaw), trap(vf), trap(vl)
    nv = len(x) - 1
    u = np.zeros(nv)  # v east
    v = np.zeros(nv)  # v north
    for i in range(nv):
        yawi, vfi, vli = yaw2[i], vf2[i], vl2[i]
        cy, sy = np.cos(yawi), np.sin(yawi)
        # u[i] = cy * vfi
        # v[i] = sy * vfi
        u[i] = cy * vfi - sy * vli
        v[i] = sy * vfi + cy * vli
    x3 = np.cumsum(u) * dt
    y3 = np.cumsum(v) * dt

    ax, ay, az, af, al, au = odometry.T[11:17]
    # vf4 = np.cumsum(af) * dt + vf[0]
    vf4 = vf[0] * np.ones_like(vf)
    vf4[1:] += np.cumsum(trap(af) * dts)

    plt.figure()
    plt.plot(x, y, 'k.-')
    plt.plot(x2, y2, 'b.-')
    plt.plot(x3, y3, 'r.-')

    plt.figure()
    plt.plot(vf)
    plt.plot(vf4)

    plt.show()


if __name__ == '__main__':
    # test_load_video()
    # test_load_stereo_video()
    test_load_video_odometry()
