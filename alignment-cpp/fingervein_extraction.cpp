#include "NumCpp.hpp"
#include <iostream>
#include <cstring>
#include "pipeline.hpp"

#include "fingervein_extraction.h"
#include "distance.hpp"

#define THRESHOLD 0.88

// TODO: define register function
size_t register_fingervein (const int width, const int height, 
                           const int camera_perspective,
                           bool** modelOut,
                           uint8_t* imageIn) {

    nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);
    nc::NdArray<bool> res = run_pipeline(width, height, camera_perspective, &image, nullptr, nullptr, nullptr, false, nullptr);
    *modelOut = new bool[res.size()];
    std::memcpy(*modelOut, &res[0], res.size());

    return res.size();
}

// TODO: define comparison function
bool compare_model_with_input (const int width, const int height, 
                               const int camera_perspective,
                               uint8_t* imageIn,
                               bool* modelIn,
                               size_t modelSize) {

    assert(modelSize == (height * width));
    assert(modelIn != nullptr);

    nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);
    nc::NdArray<bool> model = nc::NdArray<bool>(modelIn, height, width, nc::PointerPolicy::COPY);
    nc::NdArray<bool> probe = run_pipeline(width, height, camera_perspective, &image, nullptr, &model, nullptr, false, nullptr);
    double dist = compute_miura_distance(model, probe);
    
    return dist < THRESHOLD;
}

void free_model(bool* model) {
    delete model;
}
