#include "NumCpp.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream> 
#include <tuple>
#include <cmath>

#ifndef EXTRACTION_H
#define EXTRACTION_H

/**
 * This function detects valleys in the image respecting the mask.
 * 
 * @param[in] image: A 2-dimensional NdArray of doubles containing 
 * the input image.
 * @param[in] mask: A 2-dimensional NdArray of doubles of the same 
 * size as input indicating where the finger can be found in the image.
 * @param[in] sigma: A double denoting the varianbce of the Gaussian 
 * filter.
 * @param[in] width: An integer denoting the width (#columns) of 
 * the image and mask.
 * @param[in] height: An integer denoting the height (#rows) of 
 * the image and mask.
 * @returns A vector of four 2-dimensional NdArrays of doubles. Each 
 * array denotes the cross-section valley detection for a specific 
 * direction. The considered direction are horizontal, vertical, 45°, 
 * and -45° (in this order).
*/
std::vector<nc::NdArray<double>> detect_valleys (nc::NdArray<double> image,
                                    nc::NdArray<double> mask,
                                    double sigma,
                                    int width,
                                    int height);

/**
 * This function finds vein probabilities in a 1-dimensional array.
 * 
 * @param[in] a: A 1-dimensional NdArray of doubles for which we 
 * want to find the vein probabilities.
 * @param[in] width: An integer denoting the size (number of columns) 
 * of a.
 * @returns A 1-dimensional NdArray of doubles denoting the vein 
 * center probabilities.
*/
nc::NdArray<double> _prob_1d (nc::NdArray<double> a, int width);

/**
 * This function finds vein probabilities in a 1-dimensional array.
 * This is an optimized version to the one before.
 * 
 * @param[in] a: A 1-dimensional NdArray of doubles for which we 
 * want to find the vein probabilities.
 * @param[in] width: An integer denoting the size (number of columns) 
 * of a.
 * @returns A 1-dimensional NdArray of doubles denoting the vein 
 * center probabilities.
*/
nc::NdArray<double> _prob_1d_opt (nc::NdArray<double> a, int width);

/**
 * This function finds all the indices of the n-th diagonal from a 
 * matrix of size (heigth x width).
 * 
 * @param[in] nth_diag: An integer denoting the n-th diagonal for 
 * which we want to find all indices.
 * @param[in] width: An integer denoting the width (#columns) of 
 * the matrix.
 * @param[in] height: An integer denoting the heigth (#rows) of 
 * the matrix.
 * @returns A vector containing integer tuples. Each tuple represents 
 * the position of an element on the nth diagonal. The first tuple 
 * elements denotes the x value (row) of the position, and the second 
 * denotes the y value (column).
*/
std::vector<std::tuple<int, int>> diag_indices(int nth_diag, int width, int height);

/**
 * This function finds all the elements of the n-th diagonal from 
 * a given matrix with size height x width.
 * 
 * @param[in] mat: The 2-dimensional NdArray of doubles from which 
 * we want to obtain the n-th diagonal.
 * @param[in] nth_diag: An integer denoting the n-th diagonal for 
 * which we want to find all elements.
 * @param[in] width: An integer denoting the width (#columns) of 
 * the matrix.
 * @param[in] height: An integer denoting the heigth (#rows) of 
 * the matrix.
 * @returns A 1-dimensional NdArray containing doubles which are 
 * the elements of the n-th diagonal.
*/
nc::NdArray<double> diag_elems(nc::NdArray<double> mat, int nth_diag, int width, int height);

/**
 * This function evaluates the joint vein center probabilities taking 
 * valley widths and depths into consideration. 
 * 
 * The following explanation is taken from the reference implementation:
 * Once the arrays of curvatures (concavities) are calculated, detection 
 * works as follows: The code scans the image in a precise direction 
 * (vertical, horizontal, diagonal, etc). It tries to find a concavity 
 * in each direction and measures its width. It then identifies the 
 * centers of the concavity and assigns a value to it, which depends on 
 * its width and maximum depth (where the peak of darkness occurs) in 
 * such a concavity. This value is accumulated on a variable, which is 
 * re-used for all directions. This variable represents the vein 
 * probabilites.
 * 
 * @param[in] input_matrices: A vector consisting of 4 2-dimensional 
 * NdArrays of doubles, each array denotes cross-section valley 
 * detections for each of the contemplated directions (horizontal, 
 * vertical, 45°, and -45°; in this order).
 * @param[in] width: An integer denoting the width (#columns) of each array.
 * @param[in] height: An integer denoting the height (#rows) of each array.
 * @returns A 2-dimensional NdArray of doubles denoting the 
 * un-accumulated vein center probabilities.
*/
nc::NdArray<double> eval_vein_probabilities (std::vector<nc::NdArray<double>> input_matrices,
                                             int width,
                                             int height);

/**
 * This function connects the centers in the given 1-dimensional array.
 * 
 * @param[in] a: A 1-dimensional NdArray of doubles for which we want 
 * to connect the centers.
 * @param[in] width: An integer denoting the size (number of columns) 
 * of a.
 * @returns A 1-dimensional NdArray of doubles containing the corrected 
 * pixel values after filtering. Note that the output array is 4 elements 
 * shorter than the input array due to the windowing operation.
*/
nc::NdArray<double> _connect_1d (nc::NdArray<double> a, int width);

/**
 * This function connects the centers in the given 1-dimensional array.
 * This is an optimized version of the function above.
 * 
 * @param[in] a: A 1-dimensional NdArray of doubles for which we want 
 * to connect the centers.
 * @param[in] width: An integer denoting the size (number of columns) 
 * of a.
 * @returns A 1-dimensional NdArray of doubles containing the corrected 
 * pixel values after filtering. Note that the output array is 4 elements 
 * shorter than the input array due to the windowing operation.
*/
nc::NdArray<double> _connect_1d_opt (nc::NdArray<double> a, int width);

/**
 * This function connects vein centers by filtering vein probabilities.
 * 
 * @param[in] V: A 2-dimensional NdArray of doubles which represent the 
 * accumulated vein center probabilities.
 * @param[in] width: An integer denoting the number of columns of V.
 * @param[in] height: An integer denoting the number of rows of V.
 * @returns A vectors consisting of four 2-dimensional NdArrays of 
 * doubles. These arrays contain the results of the filtering operation 
 * for each of the directions. Each array corresponds to the horizontal, 
 * vertical, +45° (/), and -45° (\) directions.
*/
std::vector<nc::NdArray<double>> connect_centers (nc::NdArray<double> V, int width, int height);

/**
 * This function binarises vein images by using a threshold. 
 * This works under the assumption that the distribution is disphasic.
 * 
 * @param[in] G: The vein image given as a 2-dimensional NdArray of doubles
 * which we wish to binarise.
 * @returns A 2-dimensional NdArray of bool (values are either 0 or 1) 
 * denoting where fingerveins are present in the image.
*/
nc::NdArray<bool> binarise (nc::NdArray<double> G);

/**
 * This function extracts the fingerveins of an image given as a
 * NdArray<uint8_t>.
 * 
 * @param[in] image: The image given as a 2-dimensional 
 * NdArray of uint8_t values from which we want to extract the 
 * fingerveins.
 * @param[in] mask: A mask given as a 2-dimensional NdArray of doubles 
 * denoting the region where the finger is in the input image.
 * @param[in] width: An integer denoting the width (#columns) of 
 * the image and mask.
 * @param[in] height: An integer denoting the height (#rows) of 
 * the image and mask.
 * @param[in] sigma: A parameter used for valley detection. 
 * (default = 3)
 * @returns A 2-dimensional NdArray of bools of the same size as the 
 * input arrays. This array indicates where fingerveins were found. 
 * The presence of a fingervein at a specific pixel is denoted by a 1, 
 * and 0 otherwise.
*/
nc::NdArray<bool> maximum_curvature (nc::NdArray<uint8_t> image,
                                     nc::NdArray<double> mask,
                                     int width,
                                     int height,
                                     double sigma = 3);

#endif
