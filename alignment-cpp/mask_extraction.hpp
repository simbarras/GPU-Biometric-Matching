#include <NumCpp.hpp>
#include <opencv2/core/mat.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <array>
#include <numeric>
#include <tuple>
#include <list>

/**
 * This function goes over an array of values (starting from index start)
 * going in the direction of dir and finds the maximum value that is below
 * threshold.
 * 
 * @param[in] arr: A 1-dimensional NdArray of doubles that needs to be traversed.
 * @param[in] start: The integer index value from which to start traversing.
 * @param[in] dir: A boolean indicating the direction in which to traverse, 
 * 'True' means traversing upwards, 'False' means traversing downwards.
 * @param[in] threshold: This value indicates when to stop traversing, namely,
 * if an array value is found that exceeds threshold we stop traversing.
 * @returns The index pointing to the maximum element in the NdArray. This 
 * element is either the maximum element of all traversed elements if all 
 * values were below the threshold, or it refers to the first encountered 
 * value that is above the threshold.
*/
int max_thresh(nc::NdArray<double> arr, int start, bool dir, int threshold) {

    double val = 0;
    int idx = start;
    double prev_val = 0;
    double max_val = 0;
    int max_idx = start;

    // dir = False means "down", dir = True means "up"
    if (dir) {
        while (val < threshold && idx > 30) {
            val = arr[idx];
            if (val > max_val) {
                max_val = val;
                max_idx = idx;
            }
            idx--;
            // since prev_val is never set, this is redundant
            if (prev_val >= threshold && val > prev_val) {
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
            // since prev_val is never set, this is redundant
            if (prev_val >= threshold && val < prev_val) {
                idx--;
                break;
            }
        }
    }

    // if no value exceeded the threshold, return the index of the maximum value
    if (idx == 220 or idx == 30) return max_idx;
    // otherwise return the index of the first value that exceed the threshold
    return idx;
}

/**
 * This function tries to find the edge points of a 2-dimensional NdArray of 
 * doubles. It will take the x_1-ith column and find two points that will 
 * denote the edges for this column.
 * 
 * @param[in] img: A 2-dimensional NdArray containing double values.
 * @param[in] x_1: Denotes the index of the column from which we want to find 
 * the edge points.
 * @param[in] f_1: Denotes the index from which to start searching for the edge 
 * points. (default = 130)
 * @param[in] threshold: The value that is given to max_thresh for computation. 
 * (default = 4)
 * @returns An integer array of size 3 of the form 
 * (column_index, upper_edge_point, lower_edge_point) denoting the two edge 
 * points of the chosen column.
*/
std::array<int, 3> edge_points(nc::NdArray<double> img, int x_1, int f_1 = 130, int threshold = 4) {

    // Obtain the x_1-ith column of img
    nc::Slice img_slicer = img.rSlice(0, 1);
    nc::NdArray<double> img_sliced = img(img_slicer, x_1);

    // Divide all elements of the column with the average computed over all
    // these elements
    nc::NdArray<double> avg_1 = img_sliced.nc::NdArray<double>::sum(nc::Axis::COL);
    nc::NdArray<double> avg_comp = nc::average(avg_1);
    if (avg_comp.nc::NdArray<double>::size() != 1) abort();
    double avg = avg_comp[0];
    avg_1 = nc::operator/(avg_1, avg);

    // Call max_thresh and obtain the edge points
    int a = max_thresh(avg_1, f_1, true, threshold);
    int b = max_thresh(avg_1, f_1, false, threshold);
    std::array<int, 3>  res = {{x_1, a, b}};

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

    std::list<int> points_up_down_x;
    std::list<int> points_up_y;
    std::list<int> points_down_y;

    int start = std::get<0>(roi);
    int end = std::get<1>(roi);

    for (int i = start; i < end; i++) {
        std::array<int, 3> ps = edge_points(gradient, i);
        points_up_down_x.push_back(ps[0]);
        points_up_y.push_back(ps[1]);
        points_down_y.push_back(ps[2]);
    }

    nc::NdArray<uint8_t> mask = nc::zeros_like<uint8_t>(img);

    for (; !points_up_down_x.empty();) {
        int ux = points_up_down_x.front();
        points_up_down_x.pop_front();
        int uy = points_up_y.front();
        points_up_y.pop_front();
        int dy = points_down_y.front();
        points_down_y.pop_front();

        for (int i = uy; i < dy; i++) {
            if (img(i, ux) <= 240) {
                mask(i, ux) = 1;
            }
        }
    }

    int morph_size = 1;

    
    cv::Mat maskOCV(height, width, CV_8U, &(mask(0,0)));
    cv::Mat dest = cv::Mat::zeros(height, width, CV_8U);
    /** structure used for closing in python (use MORPH_CROSS)
     * [[False  True False]
       [ True  True  True]
       [False  True False]]

    */
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3), cv::Point(1, 1));
    cv::morphologyEx(maskOCV, dest, cv::MORPH_CLOSE, kernel);

    cv::Mat kernel2 = cv::Mat::zeros(3, 5, CV_8U);
    for (int i = 0; i < 5; i++) {
        kernel2.at<uint8_t>(cv::Point(i, 1)) = 1;
    }
    
    cv::morphologyEx(dest, maskOCV, cv::MORPH_OPEN, kernel2, cv::Point(-1, -1), 10); 

    // if I understood correctly then in order to compute the convex hull, we
    // first need to find the contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours( maskOCV, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    if (contours.size() != 1) abort();
    //compute convex hull
    std::vector<std::vector<cv::Point>> hull(contours.size());
    for( size_t i = 0; i < contours.size(); i++) {
        cv::convexHull( contours[i], hull[i] );
    }

    cv::fillConvexPoly(maskOCV, hull[0], cv::Scalar_<uint8_t>(1));

    return {img, mask};
}