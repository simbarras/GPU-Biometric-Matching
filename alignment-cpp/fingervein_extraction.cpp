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

size_t register_fingervein_single (const int width, const int height, 
                                   const int camera_perspective,
                                   bool** modelOut,
                                   uint8_t* imageIn) {

    // Creates uint8_t NdArray from uint8_t pointer
    nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);

    // Runs the pipeline with the given inputs and the generated image NdArray
    nc::NdArray<bool> res = run_pipeline(width, height, camera_perspective, &image, nullptr);

    // Creates a new bool array whose address is saved in modelOut
    *modelOut = new bool[res.size()];
    std::memcpy(*modelOut, &res[0], res.size());

    return res.size();
}

size_t register_fingerveins (const int width, const int height,
                             bool** modelOut,
                             uint8_t* imageIn1,
                             uint8_t* imageIn2) {

    // Creates uint8_t NdArray from uint8_t pointer
    nc::NdArray<uint8_t> image1 = nc::NdArray<uint8_t>(imageIn1, height, width, nc::PointerPolicy::COPY);
    nc::NdArray<uint8_t> image2 = nc::NdArray<uint8_t>(imageIn2, height, width, nc::PointerPolicy::COPY);

    // Runs the pipeline with the given inputs and the generated image NdArray
    nc::NdArray<bool> res1 = run_pipeline(width, height, 1, &image1, nullptr);
    nc::NdArray<bool> res2 = run_pipeline(width, height, 2, &image2, nullptr);

    // Creates a new bool array whose address is saved in modelOut
    *modelOut = new bool[res1.size() + res2.size()];
    std::memcpy(*modelOut, &res1[0], res1.size());
    std::memcpy((*modelOut + res1.size()), &res2[0], res2.size());

    return (res1.size() + res2.size());
}

struct probeCache {
    bool alreadyCached;
    nc::NdArray<bool> cachedProbe1;
    nc::NdArray<bool> cachedProbe2;
};

struct probeCache* new_probeCache() {
    struct probeCache* newCache = new probeCache();
    newCache->alreadyCached = false;
    newCache->cachedProbe1 = nc::NdArray<bool>();
    newCache->cachedProbe2 = nc::NdArray<bool>();

    return newCache;
} 

void freeProbeCache(struct probeCache* cache) {
    delete cache;
}

bool compare_model_with_input_single (const int width, const int height, 
                                      const int camera_perspective,
                                      uint8_t* imageIn,
                                      bool* modelIn,
                                      size_t modelSize,
                                      struct probeCache* probeC) {

    // Makes sure that for comparison a model of the correct size is provided
    assert(modelSize == (height * width));
    assert(modelIn != nullptr);
    assert(probeC != nullptr);

    // Creates NdArray for the model
    nc::NdArray<bool> model = nc::NdArray<bool>(modelIn, height, width, nc::PointerPolicy::COPY);

    if (!(probeC->alreadyCached)) {
        // Creates NdArray for the input image
        nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);
        // Runs the pipeline on the input image
        probeC->cachedProbe1 = run_pipeline(width, height, camera_perspective, &image, nullptr);
        probeC->alreadyCached = true;
    }

    // Matches the probe and the model using Miura matching
    nc::NdArray<bool> probe = miura_matching(probeC->cachedProbe1, model, width, height);

    // Computes the distance between the images
    double dist = compute_miura_distance(model, probe);
    
    return dist < THRESHOLD;
}

bool compare_model_with_input (const int width, const int height,
                               const double tau,
                               uint8_t* imageIn1,
                               uint8_t* imageIn2,
                               bool* modelIn,
                               size_t modelSize,
                               struct probeCache* probeC) {

    // Makes sure that for comparison a model of the correct size is provided
    assert(modelSize == (height * width * 2));
    assert(modelIn != nullptr);
    assert(probeC != nullptr);

    // Creates NdArrays for the models
    nc::NdArray<bool> model1 = nc::NdArray<bool>(modelIn, height, width, nc::PointerPolicy::COPY);
    nc::NdArray<bool> model2 = nc::NdArray<bool>((modelIn + (height * width)), height, width, nc::PointerPolicy::COPY);

    if (!(probeC->alreadyCached)) {
        // Creates NdArray for the input images
        nc::NdArray<uint8_t> image1 = nc::NdArray<uint8_t>(imageIn1, height, width, nc::PointerPolicy::COPY);
        nc::NdArray<uint8_t> image2 = nc::NdArray<uint8_t>(imageIn2, height, width, nc::PointerPolicy::COPY);
        // Runs the pipeline on the input images
        probeC->cachedProbe1 = run_pipeline(width, height, 1, &image1, nullptr);
        probeC->cachedProbe2 = run_pipeline(width, height, 2, &image2, nullptr);
        probeC->alreadyCached = true;
    }

    // Matches the probes and the models using Miura matching
    nc::NdArray<bool> probe1 = miura_matching(probeC->cachedProbe1, model2, width, height);
    nc::NdArray<bool> probe2 = miura_matching(probeC->cachedProbe2, model2, width, height);

    // Computes the distance between the images
    double dist1 = compute_miura_distance(model1, probe1);
    double dist2 = compute_miura_distance(model2, probe2);

    double combined_distance = (tau * dist1) + ((1 - tau) * dist2);
    
    return combined_distance < THRESHOLD;
}

void free_model(bool* model) {
    delete model;
}
