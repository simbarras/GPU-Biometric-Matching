#include "NumCpp.hpp"
#include <iostream>
#include <tuple>

#include "distance.hpp"
#include "extraction.hpp"
#include "mask_extraction.hpp"
#include "postalignment.hpp"
#include "prealignment.hpp"

#ifndef PIPELINE_H
#define PIPELINE_H

/**
 * This function executes the pipeline. The steps are as follows:
 *      1. Masking:              Edge Mask
 *      2. Prealignment:         Translation Alignment
 *      3. Preprocessing:        omitted
 *      4. Extracting Image:     Maximum Curvature
 *      5. Postprocessing:       omitted
 *      6. Postalignment:        Miura Matching (done outside of this function)
 *      7. Distance Computation: Miura Distance (done outside of this function)
 *
 * @param[in] width: A constant integer denoting the width of the image.
 * @param[in] height: A constant integer denoting the height of the image.
 * @param[in] camera_persp: An integer denoting which camera the image was
 * provided by (either 1 or 2).
 * @param[in] image: A 2-dimensional NdArray of uint8_t containing the image
 * that will be run through the pipeline. This needs to be defined.
 * @returns An extracted and aligned feature vector.
 */
nc::NdArray<bool> run_pipeline(const int width, const int height,
                               int camera_persp, nc::NdArray<uint8_t> *image);

#endif