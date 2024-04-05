#include "pipeline.hpp"

nc::NdArray<bool> run_pipeline(const int width, const int height, 
                               int camera_persp, 
                               nc::NdArray<uint8_t>* image,
                               const nc::NdArray<bool>* modelIn,
                               bool saveIntermediateSteps, 
                               std::string* saveIntermediateSteps_path) {
    
    assert(image != nullptr);

    // Open and load the image to use
    nc::NdArray<uint8_t> img = *image;

    // Extract mask
    nc::NdArray<uint8_t> mask;
    
    mask = edge_mask_extraction(img, 1, width, height);

    // TODO: Add "caching" of images

    // Prealign image
    std::tuple<nc::NdArray<uint8_t>, nc::NdArray<double>> res;
    res = translation_alignment(img, mask, width, height);

    img = std::get<0>(res);

    nc::NdArray<double> maskD;

    maskD = std::get<1>(res);

    // TODO: Add "caching" of images

    // Extract features
    nc::NdArray<bool> veins = maximum_curvature(img, maskD, width, height);

    // TODO: Add "caching" of images

    // Postalignment with the help of a model
    if (modelIn != nullptr) {
        nc::NdArray<bool> model = *modelIn;
        veins = miura_matching(veins, model, width, height);
    } 

    // TODO: Add "caching" of images
    
    return veins;
}
