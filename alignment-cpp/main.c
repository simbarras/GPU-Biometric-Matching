#include "fingervein_extraction.h"
#include <stdio.h>
#include <stdlib.h>
#include<assert.h>

int main() {
    int width = 376;
    int height = 240;

    bool* modelOut;
    uint8_t* imageIn = (uint8_t*)malloc(sizeof(uint8_t) * height * width);

    for (int i = 0; i < height * width; i++) {
        imageIn[i] = (uint8_t) (i % 256);
    }

    size_t length = register_fingervein(width, height, 1, &modelOut, imageIn);

    free_model(modelOut);
    free(imageIn);
    return 0;
}