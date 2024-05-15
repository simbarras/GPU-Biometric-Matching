#include "NumCpp.hpp"
#include "opencv2/core.hpp"
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
                                   uint8_t** modelOut,
                                   uint8_t* imageIn) {

    // Creates uint8_t NdArray from uint8_t pointer
    nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);

    // Runs the pipeline with the given inputs and the generated image NdArray
    nc::NdArray<bool> res = run_pipeline(width, height, camera_perspective, &image);

    // Counts the number of non-zero values in vein image
    uint32_t numNonZero = nc::count_nonzero(res)(0,0);

    // Computes Fourier transform of vein image
    cv::Mat veinsOCV(height, width, CV_8U, &(res(0,0)));

    cv::Mat padded; //expand input image to optimal size
    int m = cv::getOptimalDFTSize( veinsOCV.rows );
    int n = cv::getOptimalDFTSize( veinsOCV.cols ); // on the border add zero values
    cv::copyMakeBorder(veinsOCV, padded, 0, m - veinsOCV.rows, 0, n - veinsOCV.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI); // Add to the expanded another plane with zeros

    cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT);

    // Creates a new uint8_t array containing the number of non-zero values and
    // the Fourier transform of the vein image, the address is saved in modelOut
    size_t mat_Size = -1;
    if (complexI.isContinuous()) {
        mat_Size = complexI.total() * complexI.elemSize();
        *modelOut = new uint8_t[mat_Size + 4];
        std::memcpy(*modelOut, &numNonZero, 4);
        std::memcpy((*modelOut + 4), complexI.data, mat_Size);
    }

    if (mat_Size == -1) {
        // TODO: Indicate that copying failed due to non-continuous memory
    }

    return (mat_Size + 4);
}

size_t register_fingerveins (const int width, const int height,
                             uint8_t** modelOut,
                             uint8_t* imageIn1,
                             uint8_t* imageIn2) {

    // Creates uint8_t NdArray from uint8_t pointer
    nc::NdArray<uint8_t> image1 = nc::NdArray<uint8_t>(imageIn1, height, width, nc::PointerPolicy::COPY);
    nc::NdArray<uint8_t> image2 = nc::NdArray<uint8_t>(imageIn2, height, width, nc::PointerPolicy::COPY);

    // Runs the pipeline with the given inputs and the generated image NdArray
    nc::NdArray<bool> res1 = run_pipeline(width, height, 1, &image1);
    nc::NdArray<bool> res2 = run_pipeline(width, height, 2, &image2);

    // Counts the number of non-zero values in the vein images
    uint32_t numNonZero1 = nc::count_nonzero(res1)(0,0);
    uint32_t numNonZero2 = nc::count_nonzero(res2)(0,0);

    // Computes Fourier transform of vein images
    cv::Mat veins1OCV(height, width, CV_8U, &(res1(0,0)));
    cv::Mat veins2OCV(height, width, CV_8U, &(res2(0,0)));


    cv::Mat padded1; //expand input image to optimal size
    cv::Mat padded2;
    int m = cv::getOptimalDFTSize( veins1OCV.rows );
    int n = cv::getOptimalDFTSize( veins1OCV.cols ); // on the border add zero values
    cv::copyMakeBorder(veins1OCV, padded1, 0, m - veins1OCV.rows, 0, n - veins1OCV.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::copyMakeBorder(veins2OCV, padded2, 0, m - veins2OCV.rows, 0, n - veins2OCV.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes1[] = {cv::Mat_<float>(padded1), cv::Mat::zeros(padded1.size(), CV_32F)};
    cv::Mat planes2[] = {cv::Mat_<float>(padded2), cv::Mat::zeros(padded2.size(), CV_32F)};
    cv::Mat complexI1;
    cv::Mat complexI2;
    cv::merge(planes1, 2, complexI1); // Add to the expanded another plane with zeros
    cv::merge(planes2, 2, complexI2);

    cv::dft(complexI1, complexI1, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(complexI2, complexI2, cv::DFT_COMPLEX_OUTPUT);

    // Creates a new uint8_t array containing the number of non-zero values and
    // the Fourier transform of the vein image, the address is saved in modelOut
    size_t mat_Size1 = -1;
    size_t mat_Size2 = -1;
    if (complexI1.isContinuous() && complexI2.isContinuous()) {
        mat_Size1 = complexI1.total() * complexI1.elemSize();
        mat_Size2 = complexI2.total() * complexI2.elemSize();
        *modelOut = new uint8_t[mat_Size1 + mat_Size2 + 8];
        std::memcpy(*modelOut, &numNonZero1, 4);
        std::memcpy((*modelOut + 4), &numNonZero2, 4);
        std::memcpy((*modelOut + 8), complexI1.data, mat_Size1);
        std::memcpy((*modelOut + mat_Size1 + 8), complexI2.data, mat_Size2);
    }

    return (mat_Size1 + mat_Size2 + 8);
}

struct probeCache {
    bool alreadyCached;
    cv::Mat cachedProbe1;
    cv::Mat cachedProbe2;
    uint32_t cachedProbe1NumNonZeros;
    uint32_t cachedProbe2NumNonZeros;
};

struct probeCache* new_probeCache() {
    struct probeCache* newCache = new probeCache();
    newCache->alreadyCached = false;
    newCache->cachedProbe1 = cv::Mat(0, 0, CV_32FC2);
    newCache->cachedProbe2 = cv::Mat(0, 0, CV_32FC2);
    newCache->cachedProbe1NumNonZeros = 0;
    newCache->cachedProbe2NumNonZeros = 0;

    return newCache;
} 

void freeProbeCache(struct probeCache* cache) {
    delete cache;
}

bool compare_model_with_input_single (const int width, const int height, 
                                      const int camera_perspective,
                                      uint8_t* imageIn,
                                      uint8_t* modelIn,
                                      size_t modelSize,
                                      struct probeCache* probeC) {

    // Makes sure that for comparison a model of the correct size is provided
    int m = cv::getOptimalDFTSize( height );
    int n = cv::getOptimalDFTSize( width );
    assert(modelSize == (m * n * 8) + 4);
    assert(modelIn != nullptr);
    assert(probeC != nullptr);

    // Creates OpenCV matrix for the model
    cv::Mat model(m, n, CV_32FC2, (modelIn + 4));

    if (!(probeC->alreadyCached)) {
        // Creates NdArray for the input image
        nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);
        // Runs the pipeline on the input image
        nc::NdArray<bool> res = run_pipeline(width, height, camera_perspective, &image);
        // Counts the number of non-zero values in vein image
        uint32_t numNonZero = nc::count_nonzero(res)(0,0);

        // Computes Fourier transform of vein image
        cv::Mat probeOCV(height, width, CV_8U, &(res(0,0)));

        cv::Mat padded; //expand input image to optimal size
        int m = cv::getOptimalDFTSize( probeOCV.rows );
        int n = cv::getOptimalDFTSize( probeOCV.cols ); // on the border add zero values
        cv::copyMakeBorder(probeOCV, padded, 0, m - probeOCV.rows, 0, n - probeOCV.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
        cv::Mat complexI;
        cv::merge(planes, 2, complexI); // Add to the expanded another plane with zeros

        cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT);

        probeC->cachedProbe1 = complexI;
        probeC->cachedProbe1NumNonZeros = numNonZero;
        probeC->alreadyCached = true;
    }

    // Extracts the memory region containing the number of non-zero values
    uint32_t modelNonZeros = *((uint32_t*)(modelIn));

    // Computes the distance between the images using their Fourier transform and the number of non-zeros values
    double dist = compute_miura_distance_opt(model, probeC->cachedProbe1, modelNonZeros, probeC->cachedProbe1NumNonZeros);
    
    return dist < THRESHOLD;
}

bool compare_model_with_input (const int width, const int height,
                               const double tau,
                               uint8_t* imageIn1,
                               uint8_t* imageIn2,
                               u_int8_t* modelIn,
                               size_t modelSize,
                               struct probeCache* probeC) {

    // Makes sure that for comparison a model of the correct size is provided
    int m = cv::getOptimalDFTSize( height );
    int n = cv::getOptimalDFTSize( width );
    assert(modelSize == (m * n * 8 * 2) + 8);
    assert(modelIn != nullptr);
    assert(probeC != nullptr);

    // Creates OpenCV matrices for the model
    cv::Mat model1(m, n, CV_32FC2, (modelIn + 8));
    size_t mat_Size1 = model1.total() * model1.elemSize();
    cv::Mat model2(m, n, CV_32FC2, (modelIn + mat_Size1 + 8));

    if (!(probeC->alreadyCached)) {
        // Creates NdArray for the input images
        nc::NdArray<uint8_t> image1 = nc::NdArray<uint8_t>(imageIn1, height, width, nc::PointerPolicy::COPY);
        nc::NdArray<uint8_t> image2 = nc::NdArray<uint8_t>(imageIn2, height, width, nc::PointerPolicy::COPY);
        // Runs the pipeline on the input images
        nc::NdArray<bool> res1 = run_pipeline(width, height, 1, &image1);
        nc::NdArray<bool> res2 = run_pipeline(width, height, 2, &image2);
        // Counts the number of non-zero values in vein images
        uint32_t numNonZero1 = nc::count_nonzero(res1)(0,0);
        uint32_t numNonZero2 = nc::count_nonzero(res2)(0,0);

        // Computes Fourier transform of vein images
        cv::Mat probe1OCV(height, width, CV_8U, &(res1(0,0)));
        cv::Mat probe2OCV(height, width, CV_8U, &(res2(0,0)));

        cv::Mat padded1; //expand input image to optimal size
        cv::Mat padded2;
        int m = cv::getOptimalDFTSize( probe1OCV.rows );
        int n = cv::getOptimalDFTSize( probe1OCV.cols ); // on the border add zero values
        cv::copyMakeBorder(probe1OCV, padded1, 0, m - probe1OCV.rows, 0, n - probe1OCV.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::copyMakeBorder(probe2OCV, padded2, 0, m - probe2OCV.rows, 0, n - probe2OCV.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat planes1[] = {cv::Mat_<float>(padded1), cv::Mat::zeros(padded1.size(), CV_32F)};
        cv::Mat planes2[] = {cv::Mat_<float>(padded2), cv::Mat::zeros(padded2.size(), CV_32F)};
        cv::Mat complexI1;
        cv::Mat complexI2;
        cv::merge(planes1, 2, complexI1); // Add to the expanded another plane with zeros
        cv::merge(planes2, 2, complexI2);

        cv::dft(complexI1, complexI1, cv::DFT_COMPLEX_OUTPUT);
        cv::dft(complexI2, complexI2, cv::DFT_COMPLEX_OUTPUT);

        probeC->cachedProbe1 = complexI1;
        probeC->cachedProbe2 = complexI2;
        probeC->cachedProbe1NumNonZeros = numNonZero1;
        probeC->cachedProbe2NumNonZeros = numNonZero2;
        probeC->alreadyCached = true;
    }

    // Extracts the memory region containing the number of non-zeros values for both images
    uint32_t modelNonZeros1 = *((uint32_t*)(modelIn));
    uint32_t modelNonZeros2 = *((uint32_t*)(modelIn + 4));

    // Computes the distance between the images using their Fourier transform and the number of non-zero values
    double dist1 = compute_miura_distance_opt(model1, probeC->cachedProbe1, modelNonZeros1, probeC->cachedProbe1NumNonZeros);
    double dist2 = compute_miura_distance_opt(model2, probeC->cachedProbe2, modelNonZeros2, probeC->cachedProbe2NumNonZeros);

    double combined_distance = (tau * dist1) + ((1 - tau) * dist2);

    std::cout << "The distance is " << combined_distance << std::endl;
    
    return combined_distance < THRESHOLD;
}

void free_model(uint8_t* model) {
    delete model;
}
