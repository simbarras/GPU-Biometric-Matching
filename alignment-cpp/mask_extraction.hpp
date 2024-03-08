#include <NumCpp.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <array>
#include <numeric>
#include <tuple>
#include <list>

int max_thresh(nc::NdArray<double> arr, int start, bool dir, int threshold) {

    double val = 0;
    int idx = start;
    double prev_val = 0;
    double max_val = 0;
    int max_idx = start;

    // dir = false means "down", dir = true means "up"
    if (dir) {
        while (val < threshold && idx > 30) {
            val = arr[idx];
            if (val > max_val) {
                max_val = val;
                max_idx = idx;
            }
            idx--;
            if (prev_val >= threshold && val >prev_val) {
                idx++;
                break;
            }
        }
    } else {
        while (val < threshold && idx < 220) {
            val = arr[idx];
            if (val > max_val) {
                max_val = val;
                max_idx = idx;
            }
            idx++;
            if (prev_val >= threshold && val < prev_val) {
                idx--;
                break;
            }
        }
    }

    if (idx == 220 or idx == 30) return max_idx;

    return idx;
}

std::array<std::array<int, 2>, 2> edge_points(nc::NdArray<double> img, int x_1, int f_1, int width, int height, int threshold = 4) {

    //img[:, x_1:x_1 +1] uses the x_1-ith column of the image => have a closer look at whether it should be Rslice or Cslice
    nc::Slice img_slicer = img.rSlice(0, 1);
    nc::NdArray<double> img_sliced = img(img_slicer, x_1);

    // Check if correct slice was taken ;)
    //img_sliced.nc::NdArray<double>::print();

    nc::NdArray<double> avg_1 = img_sliced.nc::NdArray<double>::sum(nc::Axis::COL);
    nc::NdArray<double> avg_comp = nc::average(avg_1);
    if (avg_comp.nc::NdArray<double>::size() != 1) abort();
    double avg = avg_comp[0];
    avg_1 = nc::operator/(avg_1, avg);
    //we'll see whether that works

    int a = max_thresh(avg_1, f_1, true, threshold);
    int b = max_thresh(avg_1, f_1, false, threshold);
    std::array<std::array<int, 2>, 2>  res = {{{x_1, a}, {x_1, b}}};

    return res;
}

std::tuple<nc::NdArray<uint8_t>, nc::NdArray<uint8_t>> edge_mask_extraction(const nc::NdArray<uint8_t> img, 
                                                                            int camera_persp,
                                                                            int width,
                                                                            int height, 
                                                                            std::tuple<int, int> roi1 = {35, 355}, 
                                                                            std::tuple<int, int> roi2 = {55, 360}) {

    std::tuple<int, int> roi;
    if (camera_persp == 1) roi = roi1;
    else if (camera_persp == 2) roi = roi2;
    else abort();

    nc::NdArray<double> gradient_x = nc::gradient(img);
    nc::NdArray<double> gradient_y = nc::gradient(img, nc::Axis::COL);

    nc::NdArray<double> gradient = nc::hypot(gradient_x, gradient_y);


    int mid_y = 130;

    std::list<int> points_up_x;
    std::list<int> points_up_y;
    std::list<int> points_down_x;
    std::list<int> points_down_y;

    int start = std::get<0>(roi);
    int end = std::get<1>(roi);

    for (int i = start; i < end; i++) {
        std::array<std::array<int, 2>, 2> ps = edge_points(gradient, i, mid_y, width, height);
        points_up_x.push_back(ps[0][0]);
        points_up_y.push_back(ps[0][1]);
        points_down_x.push_back(ps[1][0]);
        points_down_y.push_back(ps[1][1]);
    }

    nc::NdArray<uint8_t> mask = nc::zeros_like<uint8_t>(img);

    for (; !points_up_x.empty();) {
        int ux = points_up_x.front();
        points_up_x.pop_front();
        int uy = points_up_y.front();
        points_up_y.pop_front();
        int dy = points_down_y.front();
        points_down_y.pop_front();

        for (int i = uy; i < dy; i++) {
            if (img[i*width + ux] <= 240) {
                mask[i*width + ux] = 1;
                std::cout << "A one was set" << i << ", " << ux << std::endl;
            }
        }
    }


    return {img, img};
}