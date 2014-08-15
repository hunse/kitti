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

#include "stereo.h"

// computation of data costs
volume<float> *comp_data(
    cv::Mat img1, cv::Mat img2, int values,
    float lambda, float threshold, float sigma)
{
    int width = img1.cols;
    int height = img1.rows;
    volume<float> *data = new volume<float>(width, height, values);

    if (lambda == 0) {
        return data;
    }

    img1.convertTo(img1, CV_32F);
    img2.convertTo(img2, CV_32F);

    cv::Mat sm1, sm2;
    if (sigma >= 0.1) {
        cv::Size size(9, 9);
        cv::GaussianBlur(img1, sm1, size, sigma);
        cv::GaussianBlur(img2, sm2, size, sigma);
    } else {
        sm1 = img1;
        sm2 = img2;
    }

    volume<float> &datar = *data;

    // single pixel differencing
    for (int y = 0; y < height; y++) {
        const float* sm1i = sm1.ptr<float>(y);
        const float* sm2i = sm2.ptr<float>(y);

        for (int x = values-1; x < width; x++) {
            for (int value = 0; value < values; value++) {
                float val = abs(sm1i[x] - sm2i[x-value]);
                datar(x, y, value) = lambda * std::min(val, threshold);
                // float val = imRef(sm1, x, y) - imRef(sm2, x-value, y);
                // imRef(data, x, y)[value] = lambda * std::min(val * val, threshold);
            }
        }
    }

    return data;
}

void add_seed_cost(
    volume<float> &data, cv::Mat seed, float epsilon)
{
    if (seed.rows == 0 || seed.cols == 0)
        return;

    for (int y = 0; y < data.height(); y++) {
        const uchar* seedi = seed.ptr<uchar>(y);

        for (int x = 0; x < data.width(); x++) {
            float seed_value = seedi[x];

            if (seed_value >= 1) {  // 0 seed means no info
                for (int value = 0; value < data.depth(); value++) {
                    data(x, y, value) +=
                        epsilon * abs(seed_value - value);
                }
            }
        }
    }
}

// multiscale belief propagation for image restoration
cv::Mat stereo_ms(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    int values, int iters, int levels, int min_level, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max)
{
    volume<float> *u[levels];
    volume<float> *d[levels];
    volume<float> *l[levels];
    volume<float> *r[levels];
    volume<float> *data[levels];

    // data costs
    data[0] = comp_data(img1, img2, values, data_weight, data_max, smooth);
    add_seed_cost(*data[0], seed, seed_weight);

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
