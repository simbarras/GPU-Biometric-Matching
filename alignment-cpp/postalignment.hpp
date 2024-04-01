#include "NumCpp.hpp"

#ifndef POSTALIGNMENT_H
#define POSTALIGNMENT_H


std::tuple<int, int> unravel_index (int arg_max, int width, int height);

std::tuple<double, int, int> miura_score (nc::NdArray<bool> model,
                                                nc::NdArray<bool> probe,
                                                int width,
                                                int height,
                                                int ch = 30,
                                                int cw = 90);

nc::NdArray<bool> miura_matching (nc::NdArray<bool> image, const nc::NdArray<bool> model, int width, int height);

#endif