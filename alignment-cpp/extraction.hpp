#include "NumCpp.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Eigen/Dense"
#include <iostream> 
#include <string>
#include <array>
#include <numeric>
#include <tuple>
#include <algorithm>
#include <cmath>

std::vector<nc::NdArray<double>> detect_valleys (nc::NdArray<double> image,
                                    nc::NdArray<double> mask,
                                    double sigma,
                                    int width,
                                    int height) {
                                        
    // Constructs the 2D Gaussian filter "h"

    double sigPow2 = pow(sigma, 2);
    double sigPow4 = pow(sigma, 4);
    double winsize = ceil(4.0 * sigma);
    nc::NdArray<double> window = nc::arange(-winsize, winsize + 1);
    std::pair<nc::NdArray<double>, nc::NdArray<double>> res = nc::meshgrid(window, window);
    nc::NdArray<double> X = std::get<0>(res);
    nc::NdArray<double> Y = std::get<1>(res);
    double G = 1.0 / (2.0 * M_PI * sigPow2);

    int size_rows = X.shape().rows;
    int size_cols = X.shape().cols;

    nc::NdArray<double> XSqrd = nc::power(X, 2);
    nc::NdArray<double> YSqrd = nc::power(Y, 2);

    nc::NdArray<double> G_arr = -(XSqrd + YSqrd) / (2 * sigPow2);
    G_arr = nc::operator*(nc::exp(G_arr), G);

    // Calculate first and second derivatives of G with respect to X
    nc::NdArray<double> G1_0 = (-X / sigPow2) * G_arr;
    nc::NdArray<double> G2_0 = ((XSqrd - sigPow2)/ sigPow4) * G_arr;
    nc::NdArray<double> G1_90 = G1_0.transpose();
    nc::NdArray<double> G2_90 = G2_0.transpose();
    nc::NdArray<double> hxy = ((X * Y) / sigPow4) * G_arr;

    // Calculates derivatives w.r.t. all directions of interest
    cv::Mat imageOCV(height, width, CV_64F, &(image(0,0)));
    cv::Mat G1_0OCV(size_rows, size_cols, CV_64F, &(G1_0(0,0)));
    cv::Mat G2_0OCV(size_rows, size_cols, CV_64F, &(G2_0(0,0)));
    cv::Mat G1_90OCV(size_rows, size_cols, CV_64F, &(G1_90(0,0)));
    cv::Mat G2_90OCV(size_rows, size_cols, CV_64F, &(G2_90(0,0)));
    cv::Mat hxyOCV(size_rows, size_cols, CV_64F, &(hxy(0,0)));

    nc::NdArray<double> image_g1_0 = nc::zeros_like<double>(image);
    nc::NdArray<double> image_g2_0 = nc::zeros_like<double>(image);
    nc::NdArray<double> image_g1_90 = nc::zeros_like<double>(image);
    nc::NdArray<double> image_g2_90 = nc::zeros_like<double>(image);
    nc::NdArray<double> fxy = nc::zeros_like<double>(image);

    cv::Mat image_G1_0OCV(height, width, CV_64F, &(image_g1_0(0,0)));
    cv::Mat image_G2_0OCV(height, width, CV_64F, &(image_g2_0(0,0)));
    cv::Mat image_G1_90OCV(height, width, CV_64F, &(image_g1_90(0,0)));
    cv::Mat image_G2_90OCV(height, width, CV_64F, &(image_g2_90(0,0)));
    cv::Mat fxyOCV(height, width, CV_64F, &(fxy(0,0)));

    // BORDER_REPLICATE

    cv::filter2D(imageOCV, image_G1_0OCV, -1, G1_0OCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);
    cv::filter2D(imageOCV, image_G2_0OCV, -1, G2_0OCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);
    cv::filter2D(imageOCV, image_G1_90OCV, -1, G1_90OCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);
    cv::filter2D(imageOCV, image_G2_90OCV, -1, G2_90OCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);
    cv::filter2D(imageOCV, fxyOCV, -1, hxyOCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);

    // for some reason with kernel G1_0 the convolution output is inverted in
    // comparison to the python code, I checked 5 times and let it be
    // cross-checked by someone else, the inputs are the exact same, only the
    // convolution output for anything related to G1_0 (also transposed) is
    // inversed, everything else works correctly;
    image_g1_0 = -(image_g1_0);
    image_g1_90 = -(image_g1_90);

    nc::NdArray<double> image_g1_45 = 0.5 * sqrt(2) * (image_g1_0 + image_g1_90);
    nc::NdArray<double> image_g1_m45 = 0.5 * sqrt(2) * (image_g1_0 - image_g1_90);

    nc::NdArray<double> image_g2_45 = 0.5 * image_g2_0 + fxy + 0.5 * image_g2_90;
    nc::NdArray<double> image_g2_m45 = 0.5 * image_g2_0 - fxy + 0.5 * image_g2_90;

    nc::NdArray<double> r1 = (image_g2_0 / (nc::sqrt((nc::power((1.0 + nc::power(image_g1_0, 2)), 3))))) * mask;
    nc::NdArray<double> r2 = (image_g2_90 / (nc::sqrt((nc::power((1.0 + nc::power(image_g1_90, 2)), 3))))) * mask;
    nc::NdArray<double> r3 = (image_g2_45 / (nc::sqrt((nc::power((1.0 + nc::power(image_g1_45, 2)), 3))))) * mask;
    nc::NdArray<double> r4 = (image_g2_m45 / (nc::sqrt((nc::power((1.0 + nc::power(image_g1_m45, 2)), 3))))) * mask;

    std::vector<nc::NdArray<double>> returnValue;
    nc::Slice slicer = nc::Slice(0, width);
    nc::Slice img_slicer = r1.cSlice(0, 1);

    for (int i = 0; i < height; i++) {
        nc::NdArray<double> partialMat = nc::zeros<double>(width, 4);

        nc::NdArray<double> slice1 = r1(i, img_slicer);
        nc::NdArray<double> slice2 = r2(i, img_slicer);
        nc::NdArray<double> slice3 = r3(i, img_slicer);
        nc::NdArray<double> slice4 = r4(i, img_slicer);

        partialMat.put(slicer, 0, slice1);
        partialMat.put(slicer, 1, slice2);
        partialMat.put(slicer, 2, slice3);
        partialMat.put(slicer, 3, slice4);

        returnValue.push_back(partialMat);
    }

    //std::cout << "rows: " << size_rows << ", cols: " << size_cols << std::endl;
    //image_g2_m45.print();


    return returnValue;

}

nc::NdArray<double> _prob_1d (nc::NdArray<double> a, int width) {

    nc::NdArray<int> b = (a > 0.).astype<int>();
    nc::Slice b1 = nc::Slice(1, width);
    nc::Slice b2 = nc::Slice(0, width - 1);

    nc::NdArray<int> diff = (b(0, b1)) - (b(0, b2));

    nc::NdArray<int> starts = nc::argwhere(diff > 0).astype<int>() + 1;
    nc::NdArray<int> ends = nc::argwhere(diff < 0).astype<int>() + 1;

    if (b(0, 0)) {
        starts = nc::insert(starts, 0, 0);
    }

    if (b(0, width - 1)) {
        nc::NdArray<int> w = {width};
        ends = nc::append(ends, w);
    }

    nc::NdArray<double> z = nc::zeros_like<double>(a);

    if (starts.size() == 0 && ends.size() == 0) {
        return z;
    }

    for (int i = 0; i < starts.size() && i < ends.size(); i++) {
        int start = starts(0, i);
        int end = ends(0, i);

        nc::Slice aSlice = nc::Slice(start, end);
        int maximum = nc::argmax(a(0, aSlice)).astype<int>()(0, 0);
        
        z(0, start + maximum) = a(0, start + maximum) * static_cast<double>(end - start);
    }

    return z;
}

nc::NdArray<double> eval_vein_probabilities (std::vector<nc::NdArray<double>> input_matrices,
                                             int width,
                                             int height) {

    nc::NdArray<double> V = nc::zeros<double>(height, width);

    nc::Slice cSlicerV = V.cSlice(0, 1);
    nc::Slice rSlicerInput = nc::Slice(0, width);
    for (int i = 0; i < height; i++) {
        V.put(i, cSlicerV, (V(i, cSlicerV) + _prob_1d(nc::flatten(input_matrices.at(i)(rSlicerInput, 0)), width)));
    }

    nc::Slice rSlicerV = V.rSlice(0, 1);
    nc::NdArray<double> slicedInput = nc::zeros<double>(1, height);
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            slicedInput(0, i) = input_matrices.at(i)(j, 1); 
        }
        V.put(rSlicerV, j, (nc::flatten(V(rSlicerV, j)) + (_prob_1d(slicedInput, height))).reshape(height, 1));
    }

    /*nc::NdArray<double> test = {{1., 1., 3., 0., -1., 3., 1.}};

    _prob_1d(test, 7);*/

    return V;
}


std::tuple<nc::NdArray<double>, nc::NdArray<double>> maximum_curvature (nc::NdArray<uint8_t> image,
                                                              nc::NdArray<double> mask,
                                                              int width,
                                                              int height,
                                                              double sigma = 3) {
    // Makes image to a double type
    nc::NdArray<double> finger_image = image.astype<double>(); 

    // TODO: detect valleys
    std::vector<nc::NdArray<double>> kappa = detect_valleys(finger_image,mask, sigma, width, height);
    // TODO: evaluate vein probabilities
    nc::NdArray<double> V = eval_vein_probabilities(kappa, width, height);
    // TODO: connect centers
    // TODO: binarise   

    return {finger_image, finger_image};  
}