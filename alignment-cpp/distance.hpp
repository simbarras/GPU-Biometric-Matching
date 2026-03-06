#include "NumCpp.hpp"
#include "opencv2/core.hpp"

#ifndef DISTANCE_H
#define DISTANCE_H

/**
 * This function computes the distance between two extracted fingervein images.
 *
 * @param[in] model: A 2-dimensional NdArray of bools which we
 * compare to another fingervein image.
 * @param[in] probe: A 2-dimensional NdArray of bools which we
 * compare to the model to compute a distance value.
 * @returns A double denoting the distance between the two input images.
 */
double compute_miura_distance(nc::NdArray<bool> model, nc::NdArray<bool> probe);

/**
 * This function computes the distance between two extracted fingervein images
 * in Fourier space. This is function is used instead of the one before to fit
 * the new pipeline computation.
 *
 * @param[in] model: A 2-dimensional, 2-channel OpenCV Mat containing the
 * Fourier transformed fingervein image of the model.
 * @param[in] probe: A 2-dimensional, 2-channel OpenCV Mat containing the
 * Fourier transformed fingervein image of the probe.
 * @param[in] numNonZerosModel: A uint32_t denoting the number of non-zero
 * values in the extracted fingervein image of the model, i.e. how many pixels
 * were part of a vein.
 * @param[in] numNonZerosProbe: A uint32_t denoting the number of non-zero
 * values in the extracted fingervein image of the probe, i.e. how many pixels
 * were part of a vein.
 * @returns A double denoting the distance between the two input images.
 */
double compute_miura_distance_opt(cv::Mat model, cv::Mat probe,
                                  uint32_t numNonZerosModel,
                                  uint32_t numNonZerosProbe);

#endif
