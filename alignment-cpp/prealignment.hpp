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
 * This function finds all indices in a NdArray of doubles where the value 
 * equals 1. The indices are split into two arrays, where the first one 
 * contains the x-axis indices (columns), and the second contains the y-axis 
 * indices (rows).
 * 
 * @param[in] input: A 2-dimensional NdArray of doubles for which the elements are 
 * checked.
 * @param[in] width: An integer denoting the width (#columns) of the input array.
 * @param[in] height: An integer denoting the height (#rows) of the input array.
 * @returns A tuple of size 3, where the first value is a 1-dimensional NdArray of 
 * doubles containing the x-axis indices (columns), the second value is a 
 * 1-dimensional NdArray of doubles containing the y-axis indices (rows), and the 
 * last value denotes the size of these two arrays.
*/
std::tuple<nc::NdArray<double>, nc::NdArray<double>, int> whereEqualOneToIndex(nc::NdArray<double> input, int width, int height) {
    
    // Finds the size that the indices arrays need to be (since we know that the
    // input array should only contain 0's and 1's, otherwise this would not
    // work)
    uint32_t numNonZero = static_cast<uint32_t>(nc::count_nonzero(input)(0,0));

    nc::NdArray<double> x_axis = nc::zeros<double>(numNonZero, 1);
    nc::NdArray<double> y_axis = nc::zeros<double>(numNonZero, 1);

    // Traverses the matrix up until the end or until we found all values that
    // equal 1
    int idx = 0;
    for (int i = 0; i < height && idx < numNonZero; i++) {
        for (int j = 0; j < width && idx < numNonZero; j++) {
            if (input(i, j) == 1) {
                // solely done in this order since Simon defined rows to be Y
                x_axis(idx, 0) = j;
                y_axis(idx, 0) = i;
                idx++;
            }
        }
    }

    return {x_axis, y_axis, numNonZero};
}

/**
 * A custom class used to compute a simple Linear Regression.
 * Taken from: https://www.geeksforgeeks.org/regression-analysis-and-the-best-fitting-line-using-c/
*/
class linearRegression {
    // An array which is going to contain all elements of x
    std::vector<double> x;
    // An array which is going to contain all elements of y
    std::vector<double> y;
 
    // Store the coefficient/slope in
    // the best fitting line
    double coeff;
 
    // Store the constant term in
    // the best fitting line
    double constTerm;
 
    // Contains sum of product of
    // all x and y elements
    double sum_xy;
 
    // Contains sum of all x elements
    double sum_x;
 
    // Contains sum of all y elements
    double sum_y;
 
    // Contains sum of square of
    // all x elements
    double sum_x_square;
 
    // Contains sum of square of
    // all y elements
    double sum_y_square;
 
public:
    // The constructor for the linearRegression class.
    linearRegression() {
        coeff = 0;
        constTerm = 0;
        sum_y = 0;
        sum_y_square = 0;
        sum_x_square = 0;
        sum_x = 0;
        sum_xy = 0;
    }
 
    // Function that calculate the coefficient/
    // slope of the best fitting line
    void calculateCoefficient() {
        double N = x.size();
        double numerator
            = (N * sum_xy - sum_x * sum_y);
        double denominator
            = (N * sum_x_square - sum_x * sum_x);
        coeff = numerator / denominator;
    }

    // Function instantiating x, y, and the other values needed to compute the
    // coefficient.
    void takeInput(int n, nc::NdArray<double> xin, nc::NdArray<double> yin) {
        for (int i = 0; i < n; i++) {
            double xi = xin(i, 0);
            double yi = yin(i, 0);
            sum_xy += xi * yi;
            sum_x += xi;
            sum_y += yi;
            sum_x_square += xi * xi;
            sum_y_square += yi * yi;
            x.push_back(xi);
            y.push_back(yi);
        }
    }

    // Function that returns the coefficient
    // of the best fitting line
    double coefficient()
    {
        if (coeff == 0)
            calculateCoefficient();
        return coeff;
    }
};

/**
 * This function takes a matrix and rotates it according to the given angle and
 * point.
 *
 * @param[in] rotMat: The matrix given as a NdArray of either type uint8_t or 
 * double which will be rotated.
 * @param[in] angle: A double denoting the angle by how much the matrix should 
 * be rotated.
 * @param[in] x: A double denoting the second dimension of the point used as 
 * the center for the rotation.
 * @param[in] y: A double denoting the sfirst dimension of the point used as 
 * the center for the rotation.
 * @param[in] width: An integer denoting the width (#columns) of the input 
 * matrix.
 * @param[in] height: An integer denoting the height (#rows) of the input matrix.
 * @returns A NdArray of the same type and size as the input matrix that contains 
 * the rotated input matrix.
*/
template<typename T>
nc::NdArray<T> rotateMat(nc::NdArray<T> rotMat, double angle, double x, double y, int width, int height) {

    // Depending on the input matrix type, the OpenCV matrices need to be
    // instantiated differently.
    int type;
    if constexpr ( std::is_same_v<uint8_t, T> ) {
        type = CV_8U;
    } else if constexpr ( std::is_same_v<double, T> ) {
        type = CV_64F;
    }

    // Creates the rotation matrix
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(y, x), angle, 1.0);

    // Applies the rotation matrix on the input matrix to obtain the
    // respectively rotated matrix.
    cv::Mat rotSrc(height, width, type, &(rotMat(0,0)));
    nc::NdArray<T> rotatedImage = nc::zeros_like<T>(rotMat);
    cv::Mat rotDest(height, width, type, &(rotatedImage(0,0)));
    cv::warpAffine(rotSrc, rotDest, rotationMatrix, cv::Size(width, height));

    return rotatedImage;
}

/**
 * This function shifts a matrix and pads the remaining values accordingly with
 * zeros, such that we obtain a matrix of the same size.
 * 
 * @param[in] img: The matrix given as a NdArray that needs to be shifted.
 * @param[in] t: An integer denoting by how much the matrix will be shifted in 
 * the first dimension (rows).
 * @param[in] s: An integer denoting by how much the matrix will be shifted in 
 * the second dimension (columns).
 * @param[in] width: An integer denoting the width (#columns) of the input 
 * matrix.
 * @param[in] height: An integer denoting the height (#rows) of the input matrix.
 * @returns A NdArray of the same type and size as the input matrix that 
 * contains the shifted input matrix padded with zeros.
*/
template<typename T>
nc::NdArray<T> shiftMat(nc::NdArray<T> img, int t, int s, int width, int height) {


    // Depending on whether the shift values are positive or negative, we need
    // to slice from the start of the matrix or the end.
    nc::Slice sliceT;
    nc::Slice shiftT;
    nc::Slice sliceS;
    nc::Slice shiftS;
    if (t >= 0) {
        sliceT = nc::Slice(t, height);
        shiftT = nc::Slice(0, (height - t));
    } else {
        sliceT = nc::Slice(0, (height + t));
        shiftT = nc::Slice(-t, height);
    }

    if (s >= 0) {
        sliceS = nc::Slice(s, width);
        shiftS = nc::Slice(0, (width - s));
    } else {
        sliceS = nc::Slice(0, (width + s));
        shiftS = nc::Slice(-s, width);
    }

    // Obtains the submatrix that we want to shift
    nc::NdArray<T> img_sliced = img(sliceT, sliceS);

    // Creates a matrix with the shifted submatrix padded with zeros.
    nc::NdArray<T> returnArray = nc::zeros_like<T>(img);
    returnArray.put(shiftT, shiftS, img_sliced);

    return returnArray;
}

/**
 * This function aligns the input images by fitting a line through the mask and
 * rotating and shifting the images to the center.
 * 
 * @param[in] img: A 2-dimensional NdArray of uint8_t values containing the image 
 * that we want to translate.
 * @param[in] mask: A 2-dimensional NdArray of uint8_t values containing the mask 
 * on which we compute the fitting and that we want to translate.
 * @param[in] width: An integer denoting the width (#columns) of the input 
 * matrices.
 * @param[in] height: An integer denoting the height (#rows) of the input 
 * matrices.
 * @returns A tuple of two 2-dimensional NdArrays containing the translated matrices. 
 * The first element contains the translated image given as a NdArray of uint8_t values, 
 * and the second element contains the translated mask given as a Ndarray of doubles.
*/
std::tuple<nc::NdArray<uint8_t>, nc::NdArray<double>> translation_alignment(nc::NdArray<uint8_t> img,
                                                                             nc::NdArray<uint8_t> mask,
                                                                             int width,
                                                                             int height) {

    // Makes mask to a double type
    nc::NdArray<double> maskDouble = mask.astype<double>();

    // Finds all indices for which the value is 1
    std::tuple<nc::NdArray<double>, nc::NdArray<double>, int> res = whereEqualOneToIndex(maskDouble, width, height);
    nc::NdArray<double> x = std::get<0>(res);
    nc::NdArray<double> y = std::get<1>(res);  
    int numNonZero = std::get<2>(res);

    // Computes the linear regression for the indices to find a coefficient that
    // best fits the line through the mask/finger.
    linearRegression lr;
    lr.takeInput(numNonZero, x, y);

    double coeff = lr.coefficient();

    int centerX = width / 2;
    int centerY = height / 2;

    double line_centerX = nc::average(x)(0,0);
    double line_centerY = nc::average(y)(0,0);

    int x_s = centerX - static_cast<int>(line_centerX);
    int y_s = centerY - static_cast<int>(line_centerY);

    // Rotates and shifts the input image according the the best fitting line.
    double angle = 360.0 * atan(coeff) / (2 * M_PI);
    img = rotateMat(img, angle, line_centerX, line_centerY, width, height);
    img = shiftMat(img, -y_s, -x_s, width, height);

    // Rotates and shifts the input mask according the the best fitting line.
    maskDouble = rotateMat(maskDouble, angle, line_centerX, line_centerY, width, height);
    maskDouble = shiftMat(maskDouble, -y_s, -x_s, width, height);

    // For now keep it as a double, since I have no idea why Simon would have
    // needed a uint16_t array
    nc::NdArray<uint16_t> maskUI16 = maskDouble.astype<uint16_t>();
    maskDouble = maskUI16.astype<double>();
    return {img, maskDouble};
}
