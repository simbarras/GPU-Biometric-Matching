#include "NumCpp.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <tuple>

#include "helper.hpp"

#ifndef POSTALIGNMENT_H
#define POSTALIGNMENT_H

/**
 * This function is equivalent to unravel_index by NumPy. It takes as input 
 * an integer in the range of a flattened (height x width) array, and 
 * returns the indices of the unflattened array. 
 * 
 * @param[in] arg_max: An integer denoting an index of a flattened array. 
 * This integer should be in the range of (0, (height * width) - 1).
 * @param[in] width: An integer denoting the width (#columns) of the 
 * unflattened array.
 * @param[in] height: An integer denoting the height (#rows) of the 
 * unflattened array.
 * @returns A tuple denoting the indices ((x, y)-values) of an unflattened 
 * array.
*/
std::tuple<int, int> unravel_index (int arg_max, int width, int height);

/**
 * This function computes a score denoting how well model and probe can be
 * maximally aligned. Furthermore, two integers are given denoting by how much
 * the probe should be shifted such that this maximal score can be achieved.
 * 
 * @param[in] model: A 2-dimensional NdArray of bools containing the 
 * fingerveins of the model.
 * @param[in] probe: A 2-dimensional NdArray of bools containing the 
 * fingerveins of the probe.
 * @param[in] width: An integer denoting the width (#columns) of the 
 * input images.
 * @param[in] height: An integer denoting the height (#rows) of the 
 * input images.
 * @param[in] ch: An integer denoting by how much the model should be 
 * cropped in the first dimension (height) in order to use it as a 
 * kernel for convolution. (default = 30)
 * @param[in] cw: An integer denoting by how much the model should be 
 * cropped in the second dimension (width) in order to use it as a 
 * kernel for convolution. (default = 90)
 * @returns A tuple consisting of a double and two integers. The double denotes 
 * a score indicating how well model and probe can be maximally aligned. The 
 * integers represent by how much the probe should be shifted to be maximally 
 * aligned with the probe. The first integer is used for the first dimension 
 * (height), the second for the second dimension (width).
*/
std::tuple<int, int> miura_score (nc::NdArray<bool> model,
                                                nc::NdArray<bool> probe,
                                                int width,
                                                int height,
                                                int ch = 30,
                                                int cw = 90);

/**
 * This function shifts the input image with respect to a model image, 
 * such that both of them are maximally aligned.
 * 
 * @param[in] image: A 2-dimensional NdArray of bools containing the 
 * fingervein image of the probe.
 * @param[in] model: A 2-dimensional NdArray of bools containing the 
 * fingervein image of the model.
 * @param[in] width: An integer denoting the width (#columns) of the 
 * input images.
 * @param[in] height: An integer denoting the height (#rows) of the 
 * input images.
 * @returns A 2-dimensional NdArray of bools containing the shifted 
 * input image, so that the model and probe are maximally aligned.
*/
nc::NdArray<bool> miura_matching (nc::NdArray<bool> image, const nc::NdArray<bool> model, int width, int height);

#endif