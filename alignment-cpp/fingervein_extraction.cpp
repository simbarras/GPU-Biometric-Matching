#include "NumCpp.hpp"
#include <iostream>
#include <cstring>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "pipeline.hpp"

#include "fingervein_extraction.h"
#include "distance.hpp"

#define THRESHOLD 0.88

size_t register_fingervein (const int width, const int height, 
                           const int camera_perspective,
                           bool** modelOut,
                           uint8_t* imageIn) {

    // Creates uint8_t NdArray from uint8_t pointer
    nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);

    // Runs the pipeline with the given inputs and the generated image NdArray
    nc::NdArray<bool> res = run_pipeline(width, height, camera_perspective, &image, nullptr, false, nullptr);

    // Creates a new bool array whose address is saved in modelOut
    *modelOut = new bool[res.size()];
    std::memcpy(*modelOut, &res[0], res.size());

    return res.size();
}

bool compare_model_with_input (const int width, const int height, 
                               const int camera_perspective,
                               uint8_t* imageIn,
                               bool* modelIn,
                               size_t modelSize) {

    // Makes sure that for comparison a model of the correct size is provided
    assert(modelSize == (height * width));
    assert(modelIn != nullptr);

    // Creates NdArrays for the input image and the model
    nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);
    nc::NdArray<bool> model = nc::NdArray<bool>(modelIn, height, width, nc::PointerPolicy::COPY);

    // Runs the pipeline on the input image
    nc::NdArray<bool> probe = run_pipeline(width, height, camera_perspective, &image, &model, false, nullptr);

    // Computes the distance between the images
    double dist = compute_miura_distance(model, probe);
    
    return dist < THRESHOLD;
}

void free_model(bool* model) {
    delete model;
}
