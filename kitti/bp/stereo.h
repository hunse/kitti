
#ifndef STEREO_H_2014_08_15
#define STEREO_H_2014_08_15

#include "bp.h"

cv::Mat stereo_ms(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    int values, int iters, int levels, int min_level, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max);

#endif
