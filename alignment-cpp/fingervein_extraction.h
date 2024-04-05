#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef FINGERVEIN_EXTRACTION_H
#define FINGERVEIN_EXTRACTION_H

#ifdef __cplusplus
extern "C" {
#endif
    /**
     * A function taking as input an array of uint8_t values representing a
     * grayscale image of a finger and extracting the fingerveins from this
     * image.
     * 
     * @param[in] width: An integer denoting the width (#columns) of the 
     * image.
     * @param[in] height: An integer denoting the height (#rows) of the 
     * image. 
     * @param[in] camera_perspective: An integer denoting which camera the 
     * image was provided by. (either 1 (left camera) or 2 (right camera))
     * @param[out] modelOut: A boolean double-pointer which will contain 
     * the address pointing to the computed fingervein image.
     * @param[in] imageIn: A uint8_t array containing the finger image. 
     * Each byte represents a pixel of the image, thus, the length of 
     * imageIn should equal (height * width).
     * @returns The size of the model given as size_t.
    */
    size_t register_fingervein (const int width, const int height, 
                            const int camera_perspective,
                            bool** modelOut,
                            uint8_t* imageIn);

    /**
     * A function taking as input an array of uint8_t values representing a
     * grayscale image of a finger for which the fingerveins will be extracted,
     * and a model for which this was already done. It will compute the distance
     * between these two fingervein images and return a bool denoting whether
     * the images should be considered to be from the same finger.
     * 
     * @param[in] width: An integer denoting the width (#columns) of the 
     * image.
     * @param[in] height: An integer denoting the height (#rows) of the 
     * image.
     * @param[in] camera_perspective: An integer denoting which camera the 
     * image was provided by. (either 1 (left camera) or 2 (right camera))
     * @param[in] imageIn: A uint8_t array containing the finger image. 
     * Each byte represents a pixel of the image, thus, the length of 
     * imageIn should equal (height * width).
     * @param[in] modelIn: A boolean array conatining the fingervein image.
     * Each byte is either 0 or 1, a 1 indicates that in this pixel is a 
     * fingervein present.
     * @param[in] modelSize: A size_t denoting the size of the model.
     * @returns True, if the distance between the model and the extracted 
     * fingervein image is smaller than the threshold, and false, otherwise.
    */
    bool compare_model_with_input (const int width, const int height, 
                                const int camera_perspective,
                                uint8_t* imageIn,
                                bool* modelIn,
                                size_t modelSize);

    /**
     * A function deleting the model.
     * 
     * @param[in] model: A boolean pointer containing the address to a boolean 
     * array that will be deleted.
    */
    void free_model(bool* model);
#ifdef __cplusplus
}
#endif

#endif
