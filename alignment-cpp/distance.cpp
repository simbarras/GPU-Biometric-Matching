#include "NumCpp.hpp"
#include "distance.hpp"

double compute_miura_distance (nc::NdArray<bool> model, nc::NdArray<bool> probe) {

    nc::NdArray<bool> mAndp = model & probe;
    double countMP = (static_cast<double>(nc::count_nonzero(mAndp)(0,0)));
    double countM = (static_cast<double>(nc::count_nonzero(model)(0,0)));
    double countP = (static_cast<double>(nc::count_nonzero(probe)(0,0)));
    double distance = 1. - countMP / (countM + countP);
    return distance;
}