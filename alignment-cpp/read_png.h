#include <stdlib.h>

#ifndef READPNG_H
#define READPNG_H

/**
 * This function reads out a 8-bit grayscale png file and writes it into a
 * NdArray.
 *
 * @param[in] filename: The file path as a string pointer that leads to the
 * image that will run through the pipeline. It needs to be constant.
 * @param[in] wid: The expected width of the image given as a constant integer.
 * @param[in] hei: The expected height of the image given as a constant integer.
 * @returns A NumCpp NdArray of type uint8_t that contains the 8-bit grayscale
 * pixel values of the image.
 */
uint8_t *readpng_file_to_array(const char *filename, const int wid,
                               const int hei);

#endif
