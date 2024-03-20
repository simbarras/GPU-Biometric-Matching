#include "NumCpp.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "png.h"
#include "Eigen/Dense"
#include <iostream> 
#include <string>
#include <array>
#include <numeric>
#include <tuple>
#include <algorithm>

std::tuple<nc::NdArray<int>, nc::NdArray<int>> whereNonZeroToIndex(nc::NdArray<uint8_t> input, int width, int height) {
    nc::NdArray<nc::uint32> numNonZero = nc::count_nonzero(input);

    nc::NdArray<int> x_axis = nc::zeros<int>(numNonZero, 1);
    nc::NdArray<int> y_axis = nc::zeros<int>(numNonZero, 1);
    int idx = 0;

    for (int i = 0; i < height && idx < numNonZero; i++) {
        for (int j = 0; j < width && idx < numNonZero; j++) {
            if (input(i, j) == 1) {
                // solely done in this order since Simon defined rows to be Y
                x_axis(idx, 0) = j;
                y_axis(idx, 0) = i;
                idx++;
            }
        }
    }

    return {x_axis, y_axis};
}

std::tuple<nc::NdArray<uint8_t>, nc::NdArray<uint16_t>> translation_alignment(nc::NdArray<uint8_t> img,
                                                                             nc::NdArray<uint8_t> mask,
                                                                             int width,
                                                                             int height) {

    std::tuple<nc::NdArray<int>, nc::NdArray<int>> res = whereNonZeroToIndex(mask, width, height);
    nc::NdArray<int> x = std::get<0>(res);
    nc::NdArray<int> y = std::get<1>(res);

    // TODO: Do linear regression fitting with Eigen?

    // Make mask to a double type
    nc::NdArray<double> maskDouble = mask.astype<double>();

    nc::NdArray<uint16_t> maskUI16 = mask.astype<uint16_t>();


    return {img, maskUI16};
}
