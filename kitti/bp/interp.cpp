/*
  Copyright (C) 2006 Pedro Felzenszwalb

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include "interp.h"

// computation of data costs
volume<float> *comp_data(
    cv::Mat seed, int values, float lambda, float threshold, float sigma)
{
    int width = seed.cols;
    int height = seed.rows;
    volume<float> *datap = new volume<float>(width, height, values);
    volume<float> &data = *datap;

    if (lambda == 0) {
        return datap;
    }

    // seed.convertTo(seed, CV_32F);

    // cv::Mat sm;
    // if (sigma >= 0.1) {
    //     cv::Size size(9, 9);
    //     cv::GaussianBlur(seed, sm, size, sigma);
    // } else {
    //     sm = seed;
    // }

    for (int y = 0; y < data.height(); y++) {
        const uchar* seedi = seed.ptr<uchar>(y);

        for (int x= 0; x < data.width(); x++) {
            float seed_value = seedi[x];

            if (seed_value >= 1) {  // 0 seed means no info
                for (int value = 0; value < values; value++) {
                    float val = abs(seed_value - value);
                    data(x, y, value) = lambda * std::min(val, threshold);
                }
            }
        }
    }

    return datap;
}

// multiscale belief propagation for image restoration
cv::Mat interp_ms(
    cv::Mat seed, int values, int iters, int levels, int min_level, float smooth,
    float seed_weight, float seed_max, float disc_max)
{
    volume<float> *u[levels];
    volume<float> *d[levels];
    volume<float> *l[levels];
    volume<float> *r[levels];
    volume<float> *data[levels];

    // data costs
    data[0] = comp_data(seed, values, seed_weight, seed_max, smooth);

    // data pyramid
    for (int i = 1; i < levels; i++) {
        int old_width = data[i-1]->width();
        int old_height = data[i-1]->height();
        int new_width = (int)ceil(old_width/2.0);
        int new_height = (int)ceil(old_height/2.0);

        assert(new_width >= 1);
        assert(new_height >= 1);

        data[i] = new volume<float>(new_width, new_height, values);
        for (int y = 0; y < old_height; y++) {
            for (int x = 0; x < old_width; x++) {
                for (int value = 0; value < values; value++) {
                    (*data[i])(x/2, y/2, value) += (*data[i-1])(x, y, value);
                }
            }
        }
    }

    // run bp from coarse to fine
    for (int i = levels-1; i >= 0; i--) {
        int width = data[i]->width();
        int height = data[i]->height();

        // allocate & init memory for messages
        if (i == levels-1) {
            // in the coarsest level messages are initialized to zero
            u[i] = new volume<float>(width, height, values);
            d[i] = new volume<float>(width, height, values);
            l[i] = new volume<float>(width, height, values);
            r[i] = new volume<float>(width, height, values);
        } else {
            // initialize messages from values of previous level
            u[i] = new volume<float>(width, height, values, false);
            d[i] = new volume<float>(width, height, values, false);
            l[i] = new volume<float>(width, height, values, false);
            r[i] = new volume<float>(width, height, values, false);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int value = 0; value < values; value++) {
                        (*u[i])(x, y, value) = (*u[i+1])(x/2, y/2, value);
                        (*d[i])(x, y, value) = (*d[i+1])(x/2, y/2, value);
                        (*l[i])(x, y, value) = (*l[i+1])(x/2, y/2, value);
                        (*r[i])(x, y, value) = (*r[i+1])(x/2, y/2, value);
                    }
                }
            }
            // delete old messages and data
            delete u[i+1];
            delete d[i+1];
            delete l[i+1];
            delete r[i+1];
            delete data[i+1];
        }

        if (i >= min_level) {
            // BP
            bp_cb(*u[i], *d[i], *l[i], *r[i], *data[i], iters, disc_max);
        }
    }

    collect_messages(*u[0], *d[0], *l[0], *r[0], *data[0]);

    delete u[0];
    delete d[0];
    delete l[0];
    delete r[0];

    cv::Mat out = max_value(*data[0]);

    delete data[0];
    return out;
}
