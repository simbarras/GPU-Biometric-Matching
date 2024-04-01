#include "NumCpp.hpp"
#include <tuple>
 
#include "helper.hpp"

#ifndef PREALIGNMENT_H
#define PREALIGNMENT_H

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
std::tuple<nc::NdArray<double>, nc::NdArray<double>, int> whereEqualOneToIndex(nc::NdArray<double> input, int width, int height);

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
    linearRegression();
 
    // Function that calculate the coefficient/
    // slope of the best fitting line
    void calculateCoefficient();

    // Function instantiating x, y, and the other values needed to compute the
    // coefficient.
    void takeInput(int n, nc::NdArray<double> xin, nc::NdArray<double> yin);

    // Function that returns the coefficient
    // of the best fitting line
    double coefficient();
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
nc::NdArray<T> rotateMat(nc::NdArray<T> rotMat, double angle, double x, double y, int width, int height);

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
                                                                             int height);

#endif