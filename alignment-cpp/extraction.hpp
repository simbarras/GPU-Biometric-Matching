#include "NumCpp.hpp"

#ifndef EXTRACTION_H
#define EXTRACTION_H

std::vector<nc::NdArray<double>> detect_valleys (nc::NdArray<double> image,
                                    nc::NdArray<double> mask,
                                    double sigma,
                                    int width,
                                    int height);


nc::NdArray<double> _prob_1d (nc::NdArray<double> a, int width);

std::vector<std::tuple<int, int>> diag_indices(int nth_diag, int width, int height);

nc::NdArray<double> eval_vein_probabilities (std::vector<nc::NdArray<double>> input_matrices,
                                             int width,
                                             int height);

nc::NdArray<double> _connect_1d (nc::NdArray<double> a, int width);

std::vector<nc::NdArray<double>> connect_centers (nc::NdArray<double> V, int width, int height);

nc::NdArray<bool> binarise (nc::NdArray<double> G);

nc::NdArray<bool> maximum_curvature (nc::NdArray<uint8_t> image,
                                     nc::NdArray<double> mask,
                                     int width,
                                     int height,
                                     double sigma = 3);

#endif
