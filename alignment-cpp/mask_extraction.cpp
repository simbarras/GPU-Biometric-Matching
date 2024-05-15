#include "mask_extraction.hpp"

std::tuple<int, int> max_thresh_opt (nc::NdArray<double> arr, int start, int threshold) {
    double a_val = arr(0, start);
    double b_val = arr(0, start);
    int a_idx = start;
    int b_idx = start;

    // Go through both directions and return the index of the maximum value or
    // the first value that was greater than threshold.
    for (int i = start - 1; i > 30; i--) {
        if (arr(0, i) > a_val) {
            a_val = arr(0, i);
            a_idx = i;
        }

        if (arr(0, i) >= threshold){
            a_idx = i;
            break;
        }
    }
    
    for (int i = start + 1; i < 220; i++) {
        if (arr(0, i) > b_val) {
            b_val = arr(0, i);
            b_idx = i;
        }

        if (arr(0, i) >= threshold){
            b_idx = i;
            break;
        }
    }

    return {a_idx, b_idx};
}

int max_thresh(nc::NdArray<double> arr, int start, bool dir, int threshold) {

    double val = 0;
    int idx = start;
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
        }
    } else {
        while (val < threshold && idx < 220) {
            val = arr[idx];
            if (val > max_val) {
                max_val = val;
                max_idx = idx;
            }
            idx++;
        }
    }

    // if no value exceeded the threshold, return the index of the maximum value
    if (idx == 220 or idx == 30) return max_idx;
    // otherwise return the index of the first value that exceed the threshold
    return idx;
}

std::array<int, 3> edge_points(nc::NdArray<double> img, int x_1, int f_1, int threshold) {

    // Obtains the x_1-ith column of img
    nc::Slice img_slicer = img.rSlice(0, 1);
    nc::NdArray<double> img_sliced = img(img_slicer, x_1);

    // Divides all elements of the column with the average computed over all
    // these elements
    nc::NdArray<double> avg_1 = img_sliced.nc::NdArray<double>::sum(nc::Axis::COL);
    nc::NdArray<double> avg_comp = nc::average(avg_1);
    if (avg_comp.nc::NdArray<double>::size() != 1) abort();
    double avg = avg_comp[0];
    avg_1 = avg_1 / avg;

    // Calls max_thresh_opt and obtains the edge points
    std::tuple<int, int> edges = max_thresh_opt(avg_1, f_1, threshold);
    int a = std::get<0>(edges);
    int b = std::get<1>(edges);
    std::array<int, 3>  res = {{x_1, a, b}};

    return res;
}

nc::NdArray<uint8_t> edge_mask_extraction_opt (const nc::NdArray<uint8_t> img, 
                                               int camera_persp, int width, 
                                               int height, 
                                               std::tuple<int, int> roi1, 
                                               std::tuple<int, int> roi2) {

    // Chooses the region-of-interest according to camera perspective
    std::tuple<int, int> roi;
    if (camera_persp == 1) roi = roi1;
    else if (camera_persp == 2) roi = roi2;
    else abort();

    // Computes the gradients of the image
    nc::NdArray<double> gradient_x = nc::gradient(img);
    nc::NdArray<double> gradient_y = nc::gradient(img, nc::Axis::COL);

    nc::NdArray<double> gradient = nc::hypot(gradient_x, gradient_y);

    int start = std::get<0>(roi);
    int end = std::get<1>(roi);

    nc::NdArray<uint8_t> mask = nc::zeros_like<uint8_t>(img);

    // TODO: Figure out where the difference is here
    /*std::vector<double> averages;
    averages.reserve(end - start);
    for (int j = 0; j < height; j++) {
        for (int i = start; i < end; i++) {
            averages[i - start] = averages[i - start] + gradient(j, i);
            if (j == (height - 1)) averages[i - start] = averages[i - start] / static_cast<double>(height);
        }
    }*/

    // Computes the averages over all rows of gradient
    nc::NdArray<double> averages = nc::average(gradient, nc::Axis::ROW);

    // Fills the mask respective to the edges that were found
    for (int i = start; i < end; i++) {
        nc::Slice img_slicer = gradient.rSlice(0, 1);
        nc::NdArray<double> img_sliced = nc::flatten(gradient(img_slicer, i));
        img_sliced = img_sliced / averages(0, i);

        std::tuple<int, int> upper_lower_edge = max_thresh_opt(img_sliced);

        int u_edge = std::get<0>(upper_lower_edge);
        int l_edge = std::get<1>(upper_lower_edge);
        // inclusively lower_edge bound, in contrast to original implementation
        for (int j = u_edge; j <= l_edge; j++) {
            if (img(j, i) <= 240) {
                mask(j, i) = 1;
            }
        }
    }

    // Executes some morphological operations to get rid of imperfections
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

    // Computes the contours of the image, needed to find convex hull
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours( maskOCV, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    if (contours.size() < 1) abort();

    // Computes convex hull
    std::vector<std::vector<cv::Point>> hull(contours.size());
    for( size_t i = 0; i < contours.size(); i++) {
        cv::convexHull( contours[i], hull[i] );
    }

    // Writes back the filled convex hull into our mask
    cv::fillConvexPoly(maskOCV, hull[0], cv::Scalar_<uint8_t>(1));

    return mask;
}

nc::NdArray<uint8_t> edge_mask_extraction(const nc::NdArray<uint8_t> img, 
                                          int camera_persp, int width, 
                                          int height, 
                                          std::tuple<int, int> roi1, 
                                          std::tuple<int, int> roi2) {

    // Chooses the region-of-interest according to camera perspective
    std::tuple<int, int> roi;
    if (camera_persp == 1) roi = roi1;
    else if (camera_persp == 2) roi = roi2;
    else abort();

    // Computes the gradients of the image
    nc::NdArray<double> gradient_x = nc::gradient(img);
    nc::NdArray<double> gradient_y = nc::gradient(img, nc::Axis::COL);

    nc::NdArray<double> gradient = nc::hypot(gradient_x, gradient_y);

    std::list<int> points_up_down_x;
    std::list<int> points_up_y;
    std::list<int> points_down_y;

    int start = std::get<0>(roi);
    int end = std::get<1>(roi);

    // Finds the edges of the image
    for (int i = start; i < end; i++) {
        std::array<int, 3> ps = edge_points(gradient, i);
        points_up_down_x.push_back(ps[0]);
        points_up_y.push_back(ps[1]);
        points_down_y.push_back(ps[2]);
    }

    nc::NdArray<uint8_t> mask = nc::zeros_like<uint8_t>(img);

    // Creates a mask according to the found edges
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
    
    
    // Executes some morphological operations to get rid of imperfections
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

    // Computes the contours of the image, needed to find convex hull
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours( maskOCV, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    if (contours.size() < 1) abort();

    // Computes convex hull
    std::vector<std::vector<cv::Point>> hull(contours.size());
    for( size_t i = 0; i < contours.size(); i++) {
        cv::convexHull( contours[i], hull[i] );
    }

    // Writes back the filled convex hull into our mask
    cv::fillConvexPoly(maskOCV, hull[0], cv::Scalar_<uint8_t>(1));

    return mask;
}