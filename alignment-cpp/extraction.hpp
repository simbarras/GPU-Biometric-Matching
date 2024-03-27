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

/**
 * This function detects valleys in the image respecting the mask.
 * 
 * @param[in] image: A 2-dimensional NdArray of doubles containing 
 * the input image.
 * @param[in] mask: A 2-dimensional NdArray of doubles of the same 
 * size as input indicating where the finger can be found in the image.
 * @param[in] sigma: A double denoting the varianbce of the Gaussian 
 * filter.
 * @param[in] width: An integer denoting the width (#columns) of 
 * the image and mask.
 * @param[in] height: An integer denoting the height (#rows) of 
 * the image and mask.
 * @returns A vector of four 2-dimensional NdArrays of doubles. Each 
 * array denotes the cross-section valley detection for a specific 
 * direction. The considered direction are horizontal, vertical, 45°, 
 * and -45° (in this order).
*/
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

    // Calculates first and second derivatives of G with respect to X
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

    cv::filter2D(imageOCV, image_G1_0OCV, -1, G1_0OCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);
    cv::filter2D(imageOCV, image_G2_0OCV, -1, G2_0OCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);
    cv::filter2D(imageOCV, image_G1_90OCV, -1, G1_90OCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);
    cv::filter2D(imageOCV, image_G2_90OCV, -1, G2_90OCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);
    cv::filter2D(imageOCV, fxyOCV, -1, hxyOCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);

    // For some reason with kernel G1_0 the convolution output is inverted in
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

    std::vector<nc::NdArray<double>> returnValue{r1, r2, r3, r4};

    return returnValue;

}

/**
 * This function finds vein probabilities in a 1-dimensional array.
 * 
 * @param[in] a: A 1-dimensional NdArray of doubles for which we 
 * want to find the vein probabilities.
 * @param[in] width: An integer denoting the size (number of columns) 
 * of a.
 * @returns A 1-dimensional NdArray of doubles denoting the vein 
 * center probabilities.
*/
nc::NdArray<double> _prob_1d (nc::NdArray<double> a, int width) {

    nc::NdArray<double> z = nc::zeros_like<double>(a);
    if ( width < 2) return z;
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

/**
 * This function finds all the indices of the n-th diagonal from a 
 * matrix of size (heigth x width).
 * 
 * @param[in] nth_diag: An integer denoting the n-th diagonal for 
 * which we want to find all indices.
 * @param[in] width: An integer denoting the width (#columns) of 
 * the matrix.
 * @param[in] height: An integer denoting the heigth (#rows) of 
 * the matrix.
*/
std::vector<std::tuple<int, int>> diag_indices(int nth_diag, int width, int height) {
    assert(nth_diag < width && nth_diag > -height);

    std::vector<std::tuple<int, int>> res;

    int j = nth_diag;
    for (int i = 0; i < height && j < width; i++, j++) {
        if (j < 0) continue;
        res.push_back({i, j});
    }

    return res;
}

/**
 * This function evaluates the joint vein center probabilities taking 
 * valley widths and depths into consideration. 
 * 
 * The following explanation is taken from the reference implementation:
 * Once the arrays of curvatures (concavities) are calculated, detection 
 * works as follows: The code scans the image in a precise direction 
 * (vertical, horizontal, diagonal, etc). It tries to find a concavity 
 * in each direction and measures its width. It then identifies the 
 * centers of the concavity and assigns a value to it, which depends on 
 * its width and maximum depth (where the peak of darkness occurs) in 
 * such a concavity. This value is accumulated on a variable, which is 
 * re-used for all directions. This variable represents the vein 
 * probabilites.
 * 
 * @param[in] input_matrices: A vector consisting of 4 2-dimensional 
 * NdArrays of doubles, each array denotes cross-section valley 
 * detections for each of the contemplated directions (horizontal, 
 * vertical, 45°, and -45°; in this order).
 * @param[in] width: An integer denoting the width (#columns) of each array.
 * @param[in] height: An integer denoting the height (#rows) of each array.
 * @returns A 2-dimensional NdArray of doubles denoting the 
 * un-accumulated vein center probabilities.
*/
nc::NdArray<double> eval_vein_probabilities (std::vector<nc::NdArray<double>> input_matrices,
                                             int width,
                                             int height) {

    // Initializes the output matrix
    nc::NdArray<double> V = nc::zeros<double>(height, width);

    // Computes the vein center probabilities along the horizontal direction
    nc::Slice cSlicerV = V.cSlice(0, 1);
    for (int i = 0; i < height; i++) {
        V.put(i, cSlicerV, (V(i, cSlicerV) + _prob_1d(input_matrices.at(0)(i, cSlicerV), width)));
    }

    // Computes the vein center probabilities along the vertical direction
    nc::Slice rSlicerV = V.rSlice(0, 1);
    nc::NdArray<double> slicedInput = nc::zeros<double>(1, height);
    for (int j = 0; j < width; j++) {
        V.put(rSlicerV, j, (nc::flatten(V(rSlicerV, j)) + (_prob_1d(nc::flatten(input_matrices.at(1)(rSlicerV, j)), height))).reshape(height, 1));
    }

    // Computes the vein center probabilities along the 45° direction (/)
    nc::NdArray<double> curv = input_matrices.at(2);

    for (int index = -height + 1; index < width; index++) {
        std::vector<std::tuple<int, int>> indicesDiag = diag_indices(index, width, height);

        nc::NdArray<double> Vadd = _prob_1d(nc::diag(curv, index), indicesDiag.size());

        assert(indicesDiag.size() == Vadd.size());

        for (int idx = 0; idx < indicesDiag.size(); idx++) {
            std::tuple<int, int> idc = indicesDiag.at(idx);
            int i = std::get<0>(idc);
            int j = std::get<1>(idc);

            V(i, j) += Vadd(0, idx);
        }
    }

    // Computes the vein center probabilities along the -45° direction (\)
    curv = nc::flipud(input_matrices.at(3));
    nc::NdArray<double> Vud = nc::zeros_like<double>(V);

    for (int index = -height + 1; index < width; index++) {
        std::vector<std::tuple<int, int>> indicesDiag = diag_indices(index, width, height);

        nc::NdArray<double> Vadd = _prob_1d(nc::diag(curv, index), indicesDiag.size());

        assert(indicesDiag.size() == Vadd.size());

        for (int idx = 0; idx < indicesDiag.size(); idx++) {
            std::tuple<int, int> idc = indicesDiag.at(idx);
            int i = std::get<0>(idc);
            int j = std::get<1>(idc);

            Vud(i, j) += Vadd(0, idx);
        }
    }

    V += nc::flipud(Vud);

    return V;
}

/**
 * This function connects the centers in the given 1-dimensional array.
 * 
 * @param[in] a: A 1-dimensional NdArray of doubles for which we want 
 * to connect the centers.
 * @param[in] width: An integer denoting the size (number of columns) 
 * of a.
 * @returns A 1-dimensional NdArray of doubles containing the corrected 
 * pixel values after filtering. Note that the output array is 4 elements 
 * shorter than the input array due to the windowing operation.
*/
nc::NdArray<double> _connect_1d (nc::NdArray<double> a, int width) {
    nc::NdArray<double> z = nc::zeros<double>(1, 0);
    if (width - 4 < 1) return z;

    nc::Slice s1 = nc::Slice(3, width - 1);
    nc::Slice s2 = nc::Slice(4, width);
    nc::Slice s3 = nc::Slice(1, width - 3);
    nc::Slice s4 = nc::Slice(0, width - 4);
    nc::NdArray<double> max1 = nc::amax((nc::stack({a(0, s1), a(0, s2)}, nc::Axis::ROW)), nc::Axis::ROW);
    nc::NdArray<double> max2 = nc::amax((nc::stack({a(0, s3), a(0, s4)}, nc::Axis::ROW)), nc::Axis::ROW);

    return nc::amin(nc::stack({max1, max2}, nc::Axis::ROW), nc::Axis::ROW);
}

/**
 * This function connects vein centers by filtering vein probabilities.
 * 
 * @param[in] V: A 2-dimensional NdArray of doubles which represent the 
 * accumulated vein center probabilities.
 * @param[in] width: An integer denoting the number of columns of V.
 * @param[in] height: An integer denoting the number of rows of V.
 * @returns A vectors consisting of four 2-dimensional NdArrays of 
 * doubles. These arrays contain the results of the filtering operation 
 * for each of the directions. Each array corresponds to the horizontal, 
 * vertical, +45° (/), and -45° (\) directions.
*/
std::vector<nc::NdArray<double>> connect_centers (nc::NdArray<double> V, int width, int height) {

    // Initializes the "3"-dimensional matrix Cd which we implement as a vector
    // of NdArrays
    std::vector<nc::NdArray<double>> Cd;
    nc::NdArray<double> a1 = nc::zeros_like<double>(V);
    nc::NdArray<double> a2 = nc::zeros_like<double>(V);
    nc::NdArray<double> a3 = nc::zeros_like<double>(V);
    nc::NdArray<double> a4 = nc::zeros_like<double>(V);

    // Filters along the horizontal direction
    nc::Slice cSlicera1 = nc::Slice(2, width - 2);
    nc::Slice cSlicerV = V.cSlice(0, 1);
    for (int i = 0; i < height; i++) {
        a1.put(i, cSlicera1, (_connect_1d(V(i, cSlicerV), width)));
    }

    // Filters along the vertical direction
    nc::Slice rSlicera2 = nc::Slice(2, height - 2);
    nc::Slice rSlicerV = V.rSlice(0, 1);
    for (int i = 0; i < width; i++) {
        a2.put(rSlicera2, i, (_connect_1d(nc::flatten(V(rSlicerV, i)), height)));
    }

    // Filters along the 45° direction (/)
    nc::NdArray<double> border = nc::zeros<double>(1, 2);

    for (int index = -height + 5; index < width - 4; index++) {
        std::vector<std::tuple<int, int>> indicesDiag = diag_indices(index, width, height);

        nc::NdArray<double> in = nc::hstack({border, _connect_1d(nc::diag(V, index), indicesDiag.size()), border});

        assert(in.size() == indicesDiag.size());

        for (int idx = 0; idx < indicesDiag.size(); idx++) {
            std::tuple<int, int> idc = indicesDiag.at(idx);
            int i = std::get<0>(idc);
            int j = std::get<1>(idc);

            a3(i, j) = in(0, idx);
        }
    }

    // Filters along the -45° direction (\)
    nc::NdArray<double> Vud = nc::flipud(V);

    for (int index = -height + 5; index < width - 4; index++) {
        std::vector<std::tuple<int, int>> indicesDiag = diag_indices(index, width, height);

        nc::NdArray<double> in = nc::hstack({border, _connect_1d(nc::diag(Vud, index), indicesDiag.size()), border});

        assert(in.size() == indicesDiag.size());

        for (int idx = 0; idx < indicesDiag.size(); idx++) {
            std::tuple<int, int> idc = indicesDiag.at(idx);
            int i = std::get<0>(idc);
            int j = std::get<1>(idc);

            a4(i, j) = in(0, idx);
        }
    }

    a4 = nc::flipud(a4);

    // Instantiates the "3"-dimensional array
    Cd.push_back(a1);
    Cd.push_back(a2);
    Cd.push_back(a3);
    Cd.push_back(a4);

    return Cd;
}

/**
 * This function binarises vein images by using a threshold. 
 * This works under the assumption that the distribution is disphasic.
 * 
 * @param[in] G: The vein image given as a 2-dimensional NdArray of doubles
 * which we wish to binarise.
 * @returns A 2-dimensional NdArray of doubles (values are either 0 or 1) 
 * denoting where fingerveins are present in the image.
*/
nc::NdArray<double> binarise (nc::NdArray<double> G) {

    // Produces a flattened NdArray of doubles containing all values from G that
    // were > 0
    nc::NdArray<bool> mask = G > 0.;
    std::pair<nc::NdArray<uint>, nc::NdArray<uint>> ind = nc::nonzero(mask);

    nc::NdArray<int> x = std::get<0>(ind).astype<int>();
    nc::NdArray<int> y = std::get<1>(ind).astype<int>();
    nc::NdArray<double> Gnew = nc::zeros<double>(1, x.size());

    for (int i = 0; i < x.size(); i++) {
        Gnew(0, i) = G(x(0, i), y(0, i));
    }

    // Computes the median over all the found values
    double median = Gnew.median()(0, 0);

    // Creates a NdArray where all pixels that contain a fingervein are 1, and 0
    // otherwise
    nc::NdArray<double> Gbool = (G > median).astype<double>();

    return Gbool;
}

/**
 * This function extracts the fingerveins of an image given as a
 * NdArray<uint8_t>.
 * 
 * @param[in] image: The image given as a 2-dimensional 
 * NdArray of uint8_t values from which we want to extract the 
 * fingerveins.
 * @param[in] mask: A mask given as a 2-dimensional NdArray of doubles 
 * denoting the region where the finger is in the input image.
 * @param[in] width: An integer denoting the width (#columns) of 
 * the image and mask.
 * @param[in] height: An integer denoting the height (#rows) of 
 * the image and mask.
 * @param[in] sigma: A parameter used for valley detection. 
 * (default = 3)
 * @returns A tuple consisting of two 2-dimensional NdArrays of 
 * doubles of the same size as the input arrays. The first array 
 * indicates where fingerveins were found. The presence of a 
 * fingervein at a specific pixel is denoted by a 1, and 0 otherwise. 
 * The second array contains the mask which is equivalent to the 
 * input mask. 
*/
std::tuple<nc::NdArray<double>, nc::NdArray<double>> maximum_curvature (nc::NdArray<uint8_t> image,
                                                              nc::NdArray<double> mask,
                                                              int width,
                                                              int height,
                                                              double sigma = 3) {
    // Makes image to a double type
    nc::NdArray<double> finger_image = image.astype<double>(); 

    // Calls detect_valleys
    std::vector<nc::NdArray<double>> kappa = detect_valleys(finger_image,mask, sigma, width, height);
    
    // Calls eval_vein_probabilities
    nc::NdArray<double> V = eval_vein_probabilities(kappa, width, height);
    
    // Calls connect_centers
    std::vector<nc::NdArray<double>> Cd = connect_centers(V, width, height);
    
    // Finds the maximum pixel value considering each direction and calls
    // binarise on the result
    nc::NdArray<double> Cdin = nc::zeros<double>(height, width);
    nc::NdArray<double> a1 = Cd.at(0);
    nc::NdArray<double> a2 = Cd.at(1); 
    nc::NdArray<double> a3 = Cd.at(2); 
    nc::NdArray<double> a4 = Cd.at(3); 
    nc::Slice cSlicer = a1.cSlice(0, 1);  

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Cdin(i, j) = std::max({a1(i, j), a2(i, j), a3(i, j), a4(i, j)});
        }
    } 

    nc::NdArray<double> retval = binarise(Cdin);


    return {retval, mask};  
}