
#ifndef BP_H_2014_08_15
#define BP_H_2014_08_15

#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <assert.h>

#include "opencv2/opencv.hpp"

#include "volume.h"

#define INF 1E20

template <class T>
inline T abs(const T &x) { return (x > 0 ? x : -x); };

void msg(float* s1, float* s2,
         float* s3, float* s4,
         float* dst, int values, float threshold);

void collect_messages(
    volume<float> &u, volume<float> &d,
    volume<float> &l, volume<float> &r,
    volume<float> &data);

cv::Mat max_value(volume<float>& data);

void bp_cb(volume<float> &u, volume<float> &d,
           volume<float> &l, volume<float> &r,
           volume<float> &data,
           int iters, float threshold);

#endif
