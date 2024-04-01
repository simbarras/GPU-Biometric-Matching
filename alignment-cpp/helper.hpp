#include "NumCpp.hpp"

#ifndef HELPER_H
#define HELPER_H

template<typename T>
nc::NdArray<T> shiftMat(nc::NdArray<T> img, int t, int s, int width, int height);

#endif
