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

#include "helper.hpp"

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
