#include "pipeline.hpp"

#ifdef SAVE_INTERMEDIATE_STEPS
#define SAVE_INTERMEDIATE_STEPS_PATH "./saveIntmResults/"
#include <ctime>
#include "opencv2/imgcodecs.hpp"

void saveToPNG (std::string saveIntermediateSteps_path, nc::NdArray<int> image, int width, int height)  {

    image *= 255;
    nc::NdArray<uint8_t> imageU8 = image.astype<uint8_t>();

    cv::Mat img(height, width, CV_8U, &(imageU8(0,0)));
    std::cout << " saving image to " << saveIntermediateSteps_path << std::endl;
    cv::imwrite(saveIntermediateSteps_path, img);
}
#endif

nc::NdArray<bool> run_pipeline(const int width, const int height, 
                               int camera_persp, 
                               nc::NdArray<uint8_t>* image) {
    
    assert(image != nullptr);

    // Open and load the image to use
    nc::NdArray<uint8_t> img = *image;

    // Extract mask
    nc::NdArray<uint8_t> mask = edge_mask_extraction_opt(img, camera_persp, width, height);

    // Saves masks if needed
    #ifdef SAVE_INTERMEDIATE_STEPS
    std::string maskSave("_mask.png");
    std::string imageSave("_image.png");
    std::timespec ts;
    std::timespec_get(&ts, TIME_UTC);
    char buf[100];
    std::strftime(buf, sizeof buf, "%FT%TZ", std::gmtime(&ts.tv_sec));
    std::string time(buf);
    std::string pathMain(SAVE_INTERMEDIATE_STEPS_PATH);
    std::string path1("mask_extraction/");
    saveToPNG(pathMain + path1 + time + maskSave, mask.astype<int>(), width, height);
    #endif

    // Prealign image
    nc::NdArray<double> maskD = translation_alignment(img, mask, width, height);

    // Saves masks if needed
    #ifdef SAVE_INTERMEDIATE_STEPS
    std::string path2("translation_alignment/");
    saveToPNG(pathMain + path2 + time + imageSave, img.astype<int>(), width, height);
    #endif

    // Extract features
    nc::NdArray<bool> veins = maximum_curvature(img, maskD, width, height);

    // Saves masks if needed
    #ifdef SAVE_INTERMEDIATE_STEPS
    std::string path3("maximum_curvature/");
    saveToPNG(pathMain + path3 + time + imageSave, veins.astype<int>(), width, height);
    #endif
    
    return veins;
}
