#include "NumCpp.hpp"
#include <iostream>
#include <tuple>

#include "mask_extraction.hpp"
#include "prealignment.hpp"
#include "extraction.hpp"
#include "postalignment.hpp"
#include "distance.hpp"

#ifndef PIPELINE_H
#define PIPELINE_H

/**
 * This function executes the entire pipeline. The steps are as follows:
 *      1. Masking:              Edge Mask
 *      2. Prealignment:         Translation Alignment
 *      3. Preprocessing:        omitted
 *      4. Extracting Image:     Maximum Curvature
 *      5. Postprocessing:       omitted
 *      6. Postalignment:        Miura Matching
 *      7. Distance Computation: Miura Distance
 *
 * @param[in] width: A constant integer denoting the width of the image.
 * @param[in] height: A constant integer denoting the height of the image.
 * @param[in] camera_persp: An integer denoting which camera the image was
 * provided by (either 1 or 2).
 * @param[in] image: A 2-dimensional NdArray of uint8_t containing the image 
 * that will be run through the pipeline. This needs to be defined.
 * @param[in] modelIn: A 2-dimensional NdArray of bool containing a model that 
 * can be used as reference for the input. (default = nullptr)
 * @returns An extracted and aligned feature vector.
*/
nc::NdArray<bool> run_pipeline(const int width, const int height, 
                               int camera_persp, 
                               nc::NdArray<uint8_t>* image,
                               const nc::NdArray<bool>* modelIn = nullptr);

#endif