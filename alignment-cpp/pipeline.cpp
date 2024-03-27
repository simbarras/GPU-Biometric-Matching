#include "NumCpp.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "png.h"
#include <iostream> 
#include <string>
#include <array>
#include <numeric>
#include <tuple>

#include "mask_extraction.hpp"
#include "prealignment.hpp"
#include "extraction.hpp"



/**
 * This function reads out a 8-bit grayscale png file and writes it into a NdArray.
 * 
 * @param[in] filename: The file path as a string that leads to the image that
 * will run through the pipeline. It needs to be constant.
 * @param[in] wid: The expected width of the image given as a constant integer.
 * @param[in] hei: The expected height of the image given as a constant integer.
 * @returns A NumCpp NdArray of type uint8_t that contains the 8-bit grayscale 
 * pixel values of the image. 
*/
nc::NdArray<uint8_t> readpng_file_to_array(const char* filename, const int wid, const int hei) {

    // Open the file and abort if there is an error
    FILE *fp = fopen(filename, "rb");
    if (!fp) abort();

    // Create structures needed for reading the PNG
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) abort();

    png_infop info = png_create_info_struct(png);
    if(!info) abort();

    // libpng "error code" handling
    if(setjmp(png_jmpbuf(png))) abort();

    // Take filestream pointer and store in png
    png_init_io(png, fp);
    png_read_info(png, info);

    int width; 
    int height; 
    png_byte color_type; 
    png_byte bit_depth; 
    uint8_t** row_pointers = NULL;

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth  = png_get_bit_depth(png, info);

    // if width or height do not equal the expected width and height, abort
    if (width != wid || height != hei) abort();

    // if row pointers already set, abort
    if (row_pointers) abort();

    // Allocate space for row_pointers
    row_pointers = (uint8_t**)malloc(sizeof(uint8_t*) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (uint8_t*)malloc(png_get_rowbytes(png,info));
    }
    // Read out the images information
    png_read_image(png, row_pointers);
    // Close the filestream and destroy the read structure
    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);

    //turn the row_pointers structure into an NdArray
    std::array<std::array<uint8_t, 376>, 240> img_arr;

    for(int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++) {
            img_arr[y][x] = row[x];
        }
        free(row);
    }

    free(row_pointers);

    nc::NdArray<uint8_t> img = nc::NdArray<uint8_t>(img_arr);

    return img;
}

/**
 * This function executes the entire pipeline. The steps are as follows:
 *      1. Masking:              Edge Mask
 *      2. Prealignment:         Translation Alignment
 *      3. Preprocessing:        omitted
 *      4. Extracting Image:     Maximum Curvature
 *      5. Postprocessing:       omitted
 *      6. Postalignment:        Miura Matching
 *      7. Distance Computation: Miura Distance
 *
 * @param[in] image_path: The file path as a string that leads to the image that
 * will run through the pipeline. It needs to be constant.
 * @param[in] width: A constant integer denoting the width of the image.
 * @param[in] height: A constant integer denoting the height of the image.
 * @param[in] camera_persp: An integer denoting which camera the image was
 * provided by (either 1 or 2).
 * @param[in] caching: A boolean indicating whether the result of each pipeline
 * step should be saved.
 * @param[in] cache_path: The file path as a string where the caching results
   should be stored.
 * @returns An extracted and aligned feature vector.
*/
void run_pipeline(const char* image_path, const int width, const int height, int camera_persp, bool caching = false, std::string cache_path = "") {

    // Open and load the image to use
    nc::NdArray<uint8_t> img;
    img = readpng_file_to_array(image_path, width, height);
    
    // Extract mask
    nc::NdArray<uint8_t> mask;
    
    mask = edge_mask_extraction(img, 1, width, height);

    // TODO: Add "caching" of images

    // Prealign image
    std::tuple<nc::NdArray<uint8_t>, nc::NdArray<double>> res;
    res = translation_alignment(img, mask, width, height);

    img = std::get<0>(res);

    nc::NdArray<double> imgD;
    nc::NdArray<double> maskD;

    maskD = std::get<1>(res);

    // TODO: Add "caching" of images

    // Extract features
    std::tuple<nc::NdArray<double>, nc::NdArray<double>> res2;
    res2 = maximum_curvature(img, maskD, width, height);

    // TODO: Add "caching" of images
    
    return;
}


int main() {
    // here should be the code to be added, this will most likely run all
    // experiments and measure time

    run_pipeline("../dataset/0_left_index_1_cam1.png", 376, 240, 1);

    std::cout << "Did this work, compare with python!" << std::endl;
    return 0;
}