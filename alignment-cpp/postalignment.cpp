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
    nc::Shape Rsh = model.shape();
    int h = Rsh.rows;
    int w = Rsh.cols;

    nc::Slice rSlice = nc::Slice(ch, h - ch);
    nc::Slice cSlice = nc::Slice(cw, w - cw);

    nc::NdArray<bool> crop_model = model(rSlice, cSlice);
    nc::Shape cSh = crop_model.shape();
    int hei = cSh.rows;
    int wid = cSh.cols;

    nc::Slice slice = nc::Slice(0, wid);

    // Now, in the reference implementation they use fftconvolve on a rotated
    //matrix, however, using the rotated matrix directly for convolution will
    //lead to absolutely terrific results. I therefore skipped the rotation
    //which yields equivalent results to the reference. 

    //crop_model = nc::rot90(crop_model, 2);
    nc::NdArray<double> in1 = probe.astype<double>();
    nc::NdArray<double> in2 = crop_model.astype<double>();

    cv::Mat probeOCV(height, width, CV_64F, &(in1(0,0)));
    cv::Mat kernelOCV(hei, wid, CV_64F, &(in2(0,0)));

    nc::NdArray<double> N_c = nc::zeros<double>(height, width);
    cv::Mat NcOCV(height, width, CV_64F, &(N_c(0,0)));

    cv::filter2D(probeOCV, NcOCV, -1, kernelOCV, cv::Point(-1, -1), (0.0), cv::BORDER_REPLICATE);

    // TODO: Find out where exactly to take the slice from since there is no 1:1
    // matching with fftconvolve
    N_c = N_c(nc::Slice(90, 151), nc::Slice(98, 279));
    nc::Shape NcSh = N_c.shape();
    int heightNc = NcSh.rows;
    int widthNc = NcSh.cols;

    std::tuple<int, int> unrav = unravel_index(N_c.argmax()(0, 0), widthNc, heightNc);
    
    int t0 = std::get<0>(unrav);
    int s0 = std::get<1>(unrav);
    
    nc::Slice rSlice2 = nc::Slice(t0, t0 + h - 2 * ch);
    nc::Slice cSlice2 = nc::Slice(s0, s0 + w - 2 * cw);
    double R_c = N_c(t0, s0) / ((nc::count_nonzero(crop_model)(0, 0)) + (nc::count_nonzero(probe(rSlice2, cSlice2))(0, 0)));

    return {R_c, t0, s0};
}

nc::NdArray<bool> miura_matching (nc::NdArray<bool> image, const nc::NdArray<bool> model, int width, int height) {

    int ch = 30;
    int cw = 90;

    std::tuple<double, int, int> res = miura_score(model, image, width, height);
    double score = std::get<0>(res);
    int t0 = std::get<1>(res);
    int s0 = std::get<2>(res);

    image = shiftMat(image, t0 - ch, s0 - cw, width, height);

    return image;
}