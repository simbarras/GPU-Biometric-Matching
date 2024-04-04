#include "NumCpp.hpp"
#include "png.h"
#include <iostream>
#include <array>
#include <tuple>

#include "mask_extraction.hpp"
#include "prealignment.hpp"
#include "extraction.hpp"
#include "postalignment.hpp"
#include "distance.hpp"

#ifndef PIPELINE_H
#define PIPELINE_H

/**
 * This function reads out a 8-bit grayscale png file and writes it into a NdArray.
 * 
 * @param[in] filename: The file path as a string that leads to the image that
 * will run through the pipeline. It needs to be constant.
 * @param[in] wid: The expected width of the image given as a constant integer.
 * @param[in] hei: The expected height of the image given as a constant integer.
 * @returns A NumCpp NdArray of type uint8_t that contains the 8-bit grayscale 
 * pixel values of the image. 
*/
nc::NdArray<uint8_t> readpng_file_to_array(const char* filename, const int wid, const int hei);

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
 * that will be run through the pipeline. Either image or image_path need to 
 * be defined. (default = nullptr) 
 * @param[in] image_path: The file path as a string that leads to the image that
 * will run through the pipeline. It needs to be constant. Either image or 
 * image_path need to be defined. (default = nullptr)
 * @param[in] modelIn: A 2-dimensional NdArray of bool containing a model that 
 * can be used as reference for the input. (default = nullptr)
 * @param[in] model_path: The file path as a string that leads to the model that
 * will be used as reference. (default = nullptr)
 * @param[in] caching: A boolean indicating whether the result of each pipeline
 * step should be saved. (default = false)
 * @param[in] cache_path: The file path as a string where the caching results
   should be stored. (default = nullptr)
 * @returns An extracted and aligned feature vector.
*/
nc::NdArray<bool> run_pipeline(const int width, const int height, 
                               int camera_persp, 
                               nc::NdArray<uint8_t>* image = nullptr, 
                               const char* image_path = nullptr, 
                               const nc::NdArray<bool>* modelIn = nullptr, 
                               const char* model_path = nullptr, 
                               bool caching = false, 
                               std::string cache_path = "");

#endif