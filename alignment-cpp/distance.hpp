#include "NumCpp.hpp"

#ifndef DISTANCE_H
#define DISTANCE_H

/**
 * This function computes the distance between two extracted fingervein images.
 * 
 * @param[in] model: A 2-dimensional NdArray of bools which we 
 * compare to another fingervein image.
 * @param[in] probe: A 2-dimensional NdArray of bools which we 
 * compare to the model to compute a distance value.
 * @returns A double denoting the distance between the two input images.
*/
double compute_miura_distance (nc::NdArray<bool> model, nc::NdArray<bool> probe);

#endif
