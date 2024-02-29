#include "NumCpp.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Eigen/Dense>
#include <iostream> 
#include <string>
#include <numeric>

int main() {
    // here should be the code to be added, this will most likely run all
    // experiments and measure time

    std::cout << "Check if this is actually working" << std::endl;
    return 0;
}

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
 * @param[in] image_path: The file path as a string that leads to the image that
 * will run through the pipeline.
 * @param[in] caching: A boolean indicating whether the result of each pipeline
 * step should be saved.
 * @param[in] cache_path: The file path as a string where the caching results should
 * be stored.
 * @returns An extracted and aligned feature vector.
*/
void run_pipeline(std::string& image_path, bool caching = false, std::string cache_path = "") {
    
    //in some way obtain the information which camera is used either by taking
    //it as input param or extracting from image path name

    nc::uint32 NUM_ROWS = 512;
    nc::uint32 NUM_COLS = 512;

    auto numHalfCols = NUM_COLS / 2; // integer division
    auto ncArray     = nc::NdArray<nc::uint8>(NUM_ROWS, NUM_COLS);

    int x = 6;

    nc::square(x);

    //somehow figure out how to make the pictures binary, but I think they already are???

    //int camera_persp = 1;


    // load the image to use
    nc::NdArray<int> img;
    img = nc::load<int>(image_path);

    // load the mask depending on camera perspective, but at this point I'm not
    // even sure if we need it
    /*
    NdArray<int> mask;
    if (camera_persp == 1) {
        mask = nc::load<int>("../masks/mask_cam1.png");
    } else {
        mask = nc::load<int>("../masks/mask_cam2.png");
    }
    */
    return;
}