#include "fingervein_extraction.h"
#include <stdio.h>
#include <stdlib.h>
#include<assert.h>
#include "png.h"

/**
 * This function reads out a 8-bit grayscale png file and writes it into a NdArray.
 * 
 * @param[in] filename: The file path as a string pointer that leads to the image that
 * will run through the pipeline. It needs to be constant.
 * @param[in] wid: The expected width of the image given as a constant integer.
 * @param[in] hei: The expected height of the image given as a constant integer.
 * @returns A NumCpp NdArray of type uint8_t that contains the 8-bit grayscale 
 * pixel values of the image. 
*/
uint8_t* readpng_file_to_array(const char* filename, const int wid, const int hei) {

    // Open the file and abort if there is an error
    FILE *fp = fopen(filename, "rb");
    if (!fp) abort();

    // Create structures needed for reading the PNG
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, (void*)NULL, (void*)NULL, (void*)NULL);
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
    uint8_t** row_pointers = (void*)NULL;

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);

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
    png_destroy_read_struct(&png, &info, (void*)NULL);

    //turn the row_pointers structure into an NdArray
    uint8_t* img_arr = (uint8_t*)malloc(sizeof(uint8_t) * height * width);

    for(int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++) {
            img_arr[y * width + x] = row[x];
        }
        free(row);
    }

    free(row_pointers);

    return img_arr;
}

int main() {
    int width = 376;
    int height = 240;

    uint8_t* modelOut;
    uint8_t* imageIn = readpng_file_to_array("../dataset/0_left_index_1_cam1.png", width, height);
    uint8_t* imageIn2 = readpng_file_to_array("../dataset/0_left_index_1_cam2.png", width, height);

    //size_t length = register_fingervein_single(width, height, 1, &modelOut, imageIn);
    size_t length = register_fingerveins(width, height, &modelOut, imageIn, imageIn2);

    struct probeCache* pC = new_probeCache();

    //double dist = compare_model_with_input_single(width, height, 1, imageIn2, modelOut, length, pC);
    bool dist = compare_model_with_input(width, height, 0.55, imageIn, imageIn2, modelOut, length, pC);
    bool dist1 = compare_model_with_input(width, height, 0.55, imageIn, imageIn2, modelOut, length, pC);

    if (dist) {
        printf("The images seem to be from the same finger\n");
    } else {
        printf("The images seem to be from different fingers\n");
    }

    freeProbeCache(pC);

    free_model(modelOut);
    free(imageIn);
    free(imageIn2);
    return 0;
}