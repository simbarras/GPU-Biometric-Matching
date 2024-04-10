#include "postalignment.hpp"

std::tuple<int, int> unravel_index (int arg_max, int width, int height) {
    int y = arg_max % width;
    int x = arg_max / width;

    return {x, y};
}

std::tuple<double, int, int> miura_score (nc::NdArray<bool> model,
                                          nc::NdArray<bool> probe,
                                          int width,
                                          int height,
                                          int ch,
                                          int cw) {

    // Slices the model and uses it as a kernel for the convolution
    nc::Slice rSlice = nc::Slice(ch, height - ch);
    nc::Slice cSlice = nc::Slice(cw, width - cw);

    nc::NdArray<bool> crop_model = model(rSlice, cSlice);
    nc::Shape cSh = crop_model.shape();
    int hei = cSh.rows;
    int wid = cSh.cols;

    // Now, in the reference implementation they use fftconvolve on a rotated
    //matrix, however, using the rotated matrix directly for convolution will
    //lead to absolutely terrific results. I therefore skipped the rotation
    //which yields equivalent results to the reference. 

    //crop_model = nc::rot90(crop_model, 2);

    // Makes both input matrices to double for convolution
    nc::NdArray<double> in1 = probe.astype<double>();
    nc::NdArray<double> in2 = crop_model.astype<double>();

    cv::Mat probeOCV(height, width, CV_64F, &(in1(0,0)));
    cv::Mat kernelOCV(hei, wid, CV_64F, &(in2(0,0)));

    // Generates an output matrix for the convolution
    nc::NdArray<double> N_c = nc::zeros<double>(height, width);
    cv::Mat NcOCV(height, width, CV_64F, &(N_c(0,0)));

    // Convolves the probe and the cropped model
    cv::filter2D(probeOCV, NcOCV, -1, kernelOCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);

    // Now this must look like Slicing magic, and it kind of is. The reference
    // implementation makes use of fftconvolve in 'valid' mode which crops the
    // output image to the values that do not rely on the zero padding. Since 
    // there is no equivalent operation for OpenCV, I had to compute the 
    // 'center' by the definitions of SciPy and from the output that we obatin.
    N_c = N_c(nc::Slice(90, 151), nc::Slice(98, 279));
    nc::Shape NcSh = N_c.shape();
    int heightNc = NcSh.rows;
    int widthNc = NcSh.cols;

    // Finds the maximum value of the output array (the index where the images
    // were maximally aligned)
    std::tuple<int, int> unrav = unravel_index(N_c.argmax()(0, 0), widthNc, heightNc);
    
    int t0 = std::get<0>(unrav);
    int s0 = std::get<1>(unrav);
    
    // Computes a score denoting how well the images could be aligned
    nc::Slice rSlice2 = nc::Slice(t0, t0 + height - 2 * ch);
    nc::Slice cSlice2 = nc::Slice(s0, s0 + width - 2 * cw);
    double R_c = N_c(t0, s0) / ((nc::count_nonzero(crop_model)(0, 0)) + (nc::count_nonzero(probe(rSlice2, cSlice2))(0, 0)));

    return {R_c, t0, s0};
}

nc::NdArray<bool> miura_matching (nc::NdArray<bool> image, const nc::NdArray<bool> model, int width, int height) {

    // The default values where the convolution would be maximal, in case, that
    // probe and model are equivalent
    int ch = 30;
    int cw = 90;

    // Computes the miura score, and the indices by which to shift
    std::tuple<double, int, int> res = miura_score(model, image, width, height);
    double score = std::get<0>(res);
    int t0 = std::get<1>(res);
    int s0 = std::get<2>(res);

    // Shifts the probe matrix such that probe and model are maximally aligned
    image = shiftMat(image, t0 - ch, s0 - cw, width, height);

    return image;
}