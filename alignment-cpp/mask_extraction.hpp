#include <NumCpp.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <array>
#include <tuple>
#include <list>

#ifndef MASK_EXTRACTION_H
#define MASK_EXTRACTION_H

/**
 * This function goes over an array of values (starting from index start)
 * going in the direction of dir and finds the maximum value that is below
 * threshold.
 * 
 * @param[in] arr: A 1-dimensional NdArray of doubles that needs to be traversed.
 * @param[in] start: The integer index value from which to start traversing.
 * @param[in] dir: A boolean indicating the direction in which to traverse, 
 * 'True' means traversing upwards, 'False' means traversing downwards.
 * @param[in] threshold: This value indicates when to stop traversing, namely,
 * if an array value is found that exceeds threshold we stop traversing.
 * @returns The index pointing to the maximum element in the NdArray. This 
 * element is either the maximum element of all traversed elements if all 
 * values were below the threshold, or it refers to the first encountered 
 * value that is above the threshold.
*/
int max_thresh(nc::NdArray<double> arr, int start, bool dir, int threshold);

/**
 * This function tries to find the edge points of a 2-dimensional NdArray of 
 * doubles. It will take the x_1-ith column and find two points that will 
 * denote the edges for this column.
 * 
 * @param[in] img: A 2-dimensional NdArray containing double values.
 * @param[in] x_1: Denotes the index of the column from which we want to find 
 * the edge points.
 * @param[in] f_1: Denotes the index from which to start searching for the edge 
 * points. (default = 130)
 * @param[in] threshold: The value that is given to max_thresh for computation. 
 * (default = 4)
 * @returns An integer array of size 3 of the form 
 * (column_index, upper_edge_point, lower_edge_point) denoting the two edge 
 * points of the chosen column.
*/
std::array<int, 3> edge_points(nc::NdArray<double> img, int x_1, int f_1 = 130, int threshold = 4);

/**
 * This function takes as input an image of a finger stored in a NdArray and 
 * computes a mask for the finger.
 * 
 * @param[in] img: A grayscale image stored in a 2-dimensional 
 * NdArray<uint8_t> for which one wants to compute the mask.
 * @param[in] camera_persp: An integer denoting which camera the image was
 * provided by (either 1 or 2). This is used to decide the region-of-interest 
 * (roi).
 * @param[in] width: An integer denoting the width of the image.
 * @param[in] height: An integer denoting the height of the image.
 * @param[in] roi1: The region-of-interest for camera 1. (default = {35, 355})
 * @param[in] roi2: The region of interest for camera 2. (default = {55, 360})
 * @returns A 2-dimensional uint8_t NdArray containing the mask for the image.
*/
nc::NdArray<uint8_t> edge_mask_extraction(const nc::NdArray<uint8_t> img, 
                                          int camera_persp, int width, 
                                          int height, 
                                          std::tuple<int, int> roi1 = {35, 355}, 
                                          std::tuple<int, int> roi2 = {55, 360});
                                          
#endif