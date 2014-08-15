# distutils: language = c++
# distutils: sources = interp.cpp, stereo.cpp, bp.cpp

import cython
from cython cimport view

import numpy as np
cimport numpy as np

from libcpp cimport bool

# see http://makerwannabe.blogspot.ca/2013/09/calling-opencv-functions-via-cython.html
cdef extern from "opencv2/opencv.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        Mat(int, int, int) except +
        void create(int, int, int)
        void* data

cdef extern from "opencv2/opencv.hpp":
    cdef int CV_8U

cdef extern from "interp.h":
    cdef Mat interp_ms(
        Mat seed,
        int values, int iters, int levels, int min_level, float smooth,
        float seed_weight, float seed_max, float disc_max)

cdef extern from "stereo.h":
    cdef Mat stereo_ms(
        Mat a, Mat b, Mat seed,
        int values, int iters, int levels, int min_level, float smooth,
        float data_weight, float data_max, float seed_weight, float disc_max)


def interp(
        np.ndarray[uchar, ndim=2, mode="c"] seed,
        int values=64, int iters=5, int levels=5, int min_level=0, float smooth=0.7,
        float seed_weight=1, float seed_max=300, float disc_max=30):

    cdef int m = seed.shape[0], n = seed.shape[1]

    # copy data on
    cdef Mat x
    x.create(m, n, CV_8U)
    (<np.uint8_t[:m, :n]> x.data)[:, :] = seed

    # run belief propagation
    cdef Mat z = interp_ms(x, values, iters, levels, min_level, smooth,
                           seed_weight, seed_max, disc_max)

    # copy data off
    cdef np.ndarray[uchar, ndim=2, mode="c"] c = np.zeros_like(seed)
    c[:, :] = <np.uint8_t[:m, :n]> z.data

    return c


def stereo(
        np.ndarray[uchar, ndim=2, mode="c"] a,
        np.ndarray[uchar, ndim=2, mode="c"] b,
        np.ndarray[uchar, ndim=2, mode="c"] seed = np.array([[]], dtype='uint8'),
        int values=64, int iters=5, int levels=5, int min_level=0, float smooth=0.7,
        float data_weight=0.07, float data_max=15,
        float seed_weight=1, float disc_max=1.7):

    assert a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1]
    cdef int m = a.shape[0], n = a.shape[1]

    # copy data on
    cdef Mat x
    x.create(m, n, CV_8U)
    cdef Mat y
    y.create(m, n, CV_8U)
    (<np.uint8_t[:m, :n]> x.data)[:, :] = a
    (<np.uint8_t[:m, :n]> y.data)[:, :] = b

    cdef Mat u
    if seed.size > 0:
        u.create(m, n, CV_8U)
        (<np.uint8_t[:m, :n]> u.data)[:, :] = seed
    else:
        u.create(0, 0, CV_8U)

    # run belief propagation
    cdef Mat z = stereo_ms(x, y, u, values, iters, levels, min_level, smooth,
                           data_weight, data_max, seed_weight, disc_max)

    # copy data off
    cdef np.ndarray[uchar, ndim=2, mode="c"] c = np.zeros_like(a)
    c[:, :] = <np.uint8_t[:m, :n]> z.data

    return c
