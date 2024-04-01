#include <NumCpp.hpp>

#ifndef MASK_EXTRACTION_H
#define MASK_EXTRACTION_H

int max_thresh(nc::NdArray<double> arr, int start, bool dir, int threshold);

std::array<int, 3> edge_points(nc::NdArray<double> img, int x_1, int f_1 = 130, int threshold = 4);

nc::NdArray<uint8_t> edge_mask_extraction(const nc::NdArray<uint8_t> img, 
                                          int camera_persp, int width, 
                                          int height, 
                                          std::tuple<int, int> roi1 = {35, 355}, 
                                          std::tuple<int, int> roi2 = {55, 360});
                                          
#endif