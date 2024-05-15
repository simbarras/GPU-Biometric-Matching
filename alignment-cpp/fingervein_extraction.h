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
     * @param[out] modelOut: A uint8_t double-pointer which will contain 
     * the address pointing to a memory region containing in the first four 
     * bytes the number of non-zeros values of the vein image and in the 
     * remaining memory the computed fingervein image.
     * @param[in] imageIn: A uint8_t array containing the finger image. 
     * Each byte represents a pixel of the image, thus, the length of 
     * imageIn should equal (height * width).
     * @returns The size of the modelOut memory region given as size_t.
    */
    size_t register_fingervein_single (const int width, const int height, 
                                       const int camera_perspective,
                                       uint8_t** modelOut,
                                       uint8_t* imageIn);

    /**
     * A function taking as input two arrays of uint8_t values representing
     * two grayscale images of a finger from two different perspectives and 
     * extracts the fingerveins from these images.
     * 
     * @param[in] width: An integer denoting the width (#columns) of the 
     * images.
     * @param[in] height: An integer denoting the height (#rows) of the 
     * images. 
     * @param[out] modelOut: A uint8_t double-pointer which will contain 
     * the address pointing to a memory region containing in the first four 
     * bytes the number of non-zeros values of the first vein image, in the next 
     * four bytes the number of non-zeros values of the second vein image, and in the 
     * remaining memory the computed fingervein images.
     * @param[in] imageIn1: A uint8_t array containing the finger image of 
     * camera 1. Each byte represents a pixel of the image, thus, the length of 
     * imageIn1 should equal (height * width).
     * @param[in] imageIn2: A uint8_t array containing the finger image of 
     * camera 2. Each byte represents a pixel of the image, thus, the length of 
     * imageIn2 should equal (height * width).
     * @returns The size of the modelOut memory region given as size_t.
    */
    size_t register_fingerveins (const int width, const int height,
                                 uint8_t** modelOut,
                                 uint8_t* imageIn1,
                                 uint8_t* imageIn2);

    /**
     * This function creates a new instance of an empty probeCache struct.
     * 
     * @returns A pointer to a newly initialised probeCache.
    */
    struct probeCache* new_probeCache();

    /**
     * A function that deleting a probeCache struct.
     * 
     * @param[in] cache: A pointer to the probeCache that 
     * needs to be deleted.
    */
    void freeProbeCache(struct probeCache* cache);

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
     * @param[in] modelIn: A uint8_t array containing in the first 4 bytes the 
     * number of non-zero values of the vein image, and in the remaining memory 
     * the Fourier transform of the fingervein image.
     * @param[in] modelSize: A size_t denoting the size of the memory region of 
     * modelIn.
     * @param[inout] probeC: A pointer to a probeCache struct, from which 
     * an already computed fingervein can be taken, instead of recomputed and 
     * in which a computed fingervein image can be cached. This 
     * parameter must always be provided and never be a nullptr.
     * @returns True, if the distance between the model and the extracted 
     * fingervein image is smaller than the threshold, and false, otherwise.
    */
    bool compare_model_with_input_single (const int width, const int height, 
                                          const int camera_perspective,
                                          uint8_t* imageIn,
                                          uint8_t* modelIn,
                                          size_t modelSize,
                                          struct probeCache* probeC);

    /**
     * A function taking as input two arrays of uint8_t values representing two
     * grayscale images of a finger from two different perspectives for which 
     * the fingerveins will be extracted, and a model for which this was already 
     * done. It will compute the distance between these four fingervein images 
     * and return a bool denoting whether the images should be considered to be 
     * from the same finger.
     * 
     * @param[in] width: An integer denoting the width (#columns) of the 
     * images.
     * @param[in] height: An integer denoting the height (#rows) of the 
     * images.
     * @param[in] tau: A double chosen in a way to minimize the equal 
     * error rate.
     * @param[in] imageIn1: A uint8_t array containing the finger image of 
     * camera 1. Each byte represents a pixel of the image, thus, the length of 
     * imageIn1 should equal (height * width).
     * @param[in] imageIn2: A uint8_t array containing the finger image of 
     * camera 2. Each byte represents a pixel of the image, thus, the length of 
     * imageIn2 should equal (height * width).
     * @param[in] modelIn: A uint8_t array containing in the first 4 bytes the 
     * number of non-zero values of the first vein image, in the next 
     * four bytes the number of non-zeros values of the second vein image, and in the 
     * remaining memory the computed fingervein images.
     * @param[in] modelSize: A size_t denoting the size of the memory region of 
     * modelIn.
     * @param[inout] probeC: A pointer to a probeCache struct, from which 
     * already computed fingerveins can be taken, instead of recomputed and in 
     * which a computed fingervein image can be cached. This 
     * parameter must always be provided and never be a nullptr.
     * @returns True, if the distance between the models and the extracted 
     * fingervein images is smaller than the threshold, and false, otherwise.
    */
    bool compare_model_with_input (const int width, const int height,
                                    const double tau,
                                    uint8_t* imageIn1,
                                    uint8_t* imageIn2,
                                    uint8_t* modelIn,
                                    size_t modelSize,
                                    struct probeCache* probeC);

    /**
     * A function deleting the model.
     * 
     * @param[in] model: A uint8_t pointer containing the address to a uint8_t 
     * array that will be deleted.
    */
    void free_model(uint8_t* model);
#ifdef __cplusplus
}
#endif

#endif
