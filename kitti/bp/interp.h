
#ifndef INTERP_H_2014_08_15
#define INTERP_H_2014_08_15

#include "bp.h"

cv::Mat interp_ms(
    cv::Mat seed, int values, int iters, int levels, int min_level, float smooth,
    float seed_weight, float seed_max, float disc_max);

#endif
