#include "fingervein_extraction.h"
#include "read_png.h"
#include <assert.h>
#include <stdio.h>

int main() {
    int width = 376;
    int height = 240;

    uint8_t *modelOut;
    uint8_t *imageIn = readpng_file_to_array(
        "../../dataset_png/0_left_index_1_cam1.png", width, height);
    uint8_t *imageIn2 = readpng_file_to_array(
        "../../dataset_png/0_left_index_1_cam2.png", width, height);

    // size_t length = register_fingervein_single(width, height, 1, &modelOut,
    // imageIn);
    size_t length =
        register_fingerveins(width, height, &modelOut, imageIn, imageIn2);

    struct probeCache *pC = new_probeCache();

    // double dist = compare_model_with_input_single(width, height, 1, imageIn2,
    // modelOut, length, pC);
    bool dist = compare_model_with_input(width, height, 0.55, imageIn, imageIn2,
                                         modelOut, length, pC);
    bool dist1 = compare_model_with_input(width, height, 0.55, imageIn,
                                          imageIn2, modelOut, length, pC);

    if (dist && dist1) {
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
