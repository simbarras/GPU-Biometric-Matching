#include "NumCpp.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <tuple>
#include <cmath>
 
#include "helper.hpp"
#include "prealignment.hpp"

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
