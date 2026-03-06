#include "../extraction.hpp"
#include "../mask_extraction.hpp"
#include "NumCpp.hpp"
#include <iostream>

/**
 * Tests the correctness of the optimized _prob_1d function
 */
bool _prob_1d_testing() {

    nc::NdArray<double> test1 = {{3, 2, 5, 0, 5, 0, 0, 1, 1, 3, 0}};
    nc::NdArray<double> test2 = {{1, 0, 0, 3, 5, 0, 1, 4}};
    nc::NdArray<double> test3 = {{0, 3, 4, 1, 0, 3, 5, 0, 1, 4, 9, 4, 7}};

    nc::NdArray<double> res1 = _prob_1d(test1, 11);
    nc::NdArray<double> res2 = _prob_1d_opt(test1, 11);

    bool correctness = ((res1 == res2)(0, 0));

    /*std::cout << "Here begin the tests" << std::endl;

    res1.print();
    res2.print();*/

    res1 = _prob_1d(test2, 8);
    res2 = _prob_1d_opt(test2, 8);

    correctness = correctness && ((res1 == res2)(0, 0));

    /*
    res1.print();
    res2.print();
    */

    res1 = _prob_1d(test3, 13);
    res2 = _prob_1d_opt(test3, 13);

    correctness = correctness && ((res1 == res2)(0, 0));

    /*
    res1.print();
    res2.print();
    */

    return correctness;
}

/**
 * Tests the correctness of the optimized _connect_1d function
 */
bool _connect_1d_testing() {
    nc::NdArray<double> test1 = {
        {3.0, 2.0, 5.0, 0.0, 5.0, 0.0, 0.0, 1.0, 1.0, 3.0, 0.0}};
    nc::NdArray<double> test2 = {{1.0, 0.0, 0.0, 3.0, 5.0, 0.0, 1.0, 4.0}};
    nc::NdArray<double> test3 = {
        {0.0, 3.0, 4.0, 1.0, 0.0, 3.0, 5.0, 0.0, 1.0, 4.0, 9.0, 4.0, 7.0}};
    nc::NdArray<double> test4 = {{0.0, 3.0, 4.0, 1.0}};
    nc::NdArray<double> test5 = {{0.0, 3.0, 4.0, 1.0, 5.0}};

    nc::NdArray<double> res1 = _connect_1d(test1, 11);
    nc::NdArray<double> res2 = _connect_1d_opt(test1, 11);

    bool correctness = ((res1 == res2)(0, 0));

    /*std::cout << "Here begin the tests" << std::endl;

    res1.print();
    res2.print();*/

    res1 = _connect_1d(test2, 8);
    res2 = _connect_1d_opt(test2, 8);

    correctness = correctness && ((res1 == res2)(0, 0));

    /*
    res1.print();
    res2.print();
    */

    res1 = _connect_1d(test3, 13);
    res2 = _connect_1d_opt(test3, 13);

    correctness = correctness && ((res1 == res2)(0, 0));

    /*
    res1.print();
    res2.print();
    */

    res1 = _connect_1d(test4, 4);
    res2 = _connect_1d_opt(test4, 4);

    correctness = correctness && (res1.size() == res2.size());

    /*
    res1.print();
    res2.print();
    */

    res1 = _connect_1d(test5, 5);
    res2 = _connect_1d_opt(test5, 5);

    correctness = correctness && ((res1 == res2)(0, 0));

    /*
    res1.print();
    res2.print();
    */

    return correctness;
}

/**
 * Tests the correctness of the optimized max_thresh function
 * However, while optimizing I noticed some inconsistencies in
 * the original codebase regarding what indices to take.
 * If all values are below the threshold, we took the index of
 * the greatest value found. If we encountered a value that was
 * greater than the threshold we returned the index of the
 * element following that. From my understanding it should have
 * been the index from the value that was above the threshold.
 * Due to lack of documentation, I can only guess what was
 * supposed to happen.
 *
 * My optimized implementation returns the index of the first
 * value greater than the threshold.
 */
bool max_thresh_testing() {
    nc::NdArray<double> test1 = {
        {3.0, 2.0, 5.0, 0.0, 5.0, 0.0, 0.0, 1.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 9.0}};

    int res1 = max_thresh(test1, 35, true, 7);
    std::tuple<int, int> res2 = max_thresh_opt(test1, 35, 7);

    bool correctness = (res1 == std::get<0>(res2));

    // std::cout << "The max edge was found at: " << res1 << ", and: " <<
    // std::get<0>(res2) << std::endl;

    res1 = max_thresh(test1, 35, false, 7);

    correctness = correctness && (res1 == (std::get<1>(res2) + 1));

    // res2 + 1 is done due to mistake in original implementation only giving
    // the index of the next pixel std::cout << "The max edge was found at: " <<
    // res1 << ", and: " << (std::get<1>(res2) + 1) << std::endl;

    return correctness;
}

int main() {

    assert(_prob_1d_testing());
    assert(_connect_1d_testing());
    assert(max_thresh_testing());

    std::cout << "All tests ran without issues" << std::endl;
    return 0;
}
