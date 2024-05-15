#include "distance.hpp"
#include "opencv2/imgcodecs.hpp"

double compute_miura_distance (nc::NdArray<bool> model, nc::NdArray<bool> probe) {

    // Bitwise Ands the model and probe
    nc::NdArray<bool> mAndp = model & probe;

    // Counts the number of 1's in the respective images
    double countMP = (static_cast<double>(nc::count_nonzero(mAndp)(0,0)));
    double countM = (static_cast<double>(nc::count_nonzero(model)(0,0)));
    double countP = (static_cast<double>(nc::count_nonzero(probe)(0,0)));

    // Computes the distance
    double distance = 1. - countMP / (countM + countP);
    return distance;
}

double compute_miura_distance_opt (cv::Mat model, cv::Mat probe, uint32_t numNonZerosModel, uint32_t numNonZerosProbe) {

    // Multiply Fourier transformed matrices to obtain correlation of images
    cv::Mat mulMats;
    cv::mulSpectrums(model, probe, mulMats, 0, true);
    // Compute inverse Fourier transform to be able to find best matching value
    cv::Mat iMulRes;
    cv::idft(mulMats, iMulRes, cv::DFT_REAL_OUTPUT|cv::DFT_SCALE);

    nc::NdArray<float> mulRes = nc::NdArray<float>((float*)iMulRes.data, iMulRes.rows, iMulRes.cols, nc::PointerPolicy::COPY);

    // Finds the maximum in entire matrix denoting the best matching value
    nc::NdArray<int32_t> maxMat = mulRes.astype<int32_t>();

    double maxMatch = static_cast<double>(maxMat.max()(0, 0));

    double countM = (static_cast<double>(numNonZerosModel));
    double countP = (static_cast<double>(numNonZerosProbe));

    // Computes the distance
    double distance = 1. - maxMatch / (countM + countP);

    return distance;
}