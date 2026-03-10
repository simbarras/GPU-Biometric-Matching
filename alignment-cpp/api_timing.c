#include "fingervein_extraction.h"
#include "read_png.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

    int width = 376;
    int height = 240;

    clock_t t0, t1;
    double time_taken;

    uint8_t *modelOut;
    uint8_t **imageInList = (uint8_t **)malloc(sizeof(uint8_t *) * 40);
    uint8_t *imageIn0_13_1 = readpng_file_to_array(
        "../dataset/13_left_index_3_cam1.png", width, height);
    imageInList[0] = imageIn0_13_1;
    uint8_t *imageIn0_13_2 = readpng_file_to_array(
        "../dataset/13_left_index_3_cam2.png", width, height);
    imageInList[1] = imageIn0_13_2;
    uint8_t *imageIn1_12_1 = readpng_file_to_array(
        "../dataset/12_right_middle_1_cam1.png", width, height);
    imageInList[2] = imageIn1_12_1;
    uint8_t *imageIn1_12_2 = readpng_file_to_array(
        "../dataset/12_right_middle_1_cam2.png", width, height);
    imageInList[3] = imageIn1_12_2;
    uint8_t *imageIn2_9_1 = readpng_file_to_array(
        "../dataset/9_left_index_2_cam1.png", width, height);
    imageInList[4] = imageIn2_9_1;
    uint8_t *imageIn2_9_2 = readpng_file_to_array(
        "../dataset/9_left_index_2_cam2.png", width, height);
    imageInList[5] = imageIn2_9_2;
    uint8_t *imageIn3_6_1 = readpng_file_to_array(
        "../dataset/6_right_index_1_cam1.png", width, height);
    imageInList[6] = imageIn3_6_1;
    uint8_t *imageIn3_6_2 = readpng_file_to_array(
        "../dataset/6_right_index_1_cam2.png", width, height);
    imageInList[7] = imageIn3_6_2;
    uint8_t *imageIn4_4_1 = readpng_file_to_array(
        "../dataset/4_right_middle_1_cam1.png", width, height);
    imageInList[8] = imageIn4_4_1;
    uint8_t *imageIn4_4_2 = readpng_file_to_array(
        "../dataset/4_right_middle_1_cam2.png", width, height);
    imageInList[9] = imageIn4_4_2;

    uint8_t *imageIn5_10_1 = readpng_file_to_array(
        "../dataset/10_right_index_4_cam1.png", width, height);
    imageInList[10] = imageIn5_10_1;
    uint8_t *imageIn5_10_2 = readpng_file_to_array(
        "../dataset/10_right_index_4_cam2.png", width, height);
    imageInList[11] = imageIn5_10_2;
    uint8_t *imageIn6_17_1 = readpng_file_to_array(
        "../dataset/17_right_index_2_cam1.png", width, height);
    imageInList[12] = imageIn6_17_1;
    uint8_t *imageIn6_17_2 = readpng_file_to_array(
        "../dataset/17_right_index_2_cam2.png", width, height);
    imageInList[13] = imageIn6_17_2;
    uint8_t *imageIn7_13_1 = readpng_file_to_array(
        "../dataset/13_right_middle_1_cam1.png", width, height);
    imageInList[14] = imageIn7_13_1;
    uint8_t *imageIn7_13_2 = readpng_file_to_array(
        "../dataset/13_right_middle_1_cam2.png", width, height);
    imageInList[15] = imageIn7_13_2;
    uint8_t *imageIn8_1_1 = readpng_file_to_array(
        "../dataset/1_left_index_4_cam1.png", width, height);
    imageInList[16] = imageIn8_1_1;
    uint8_t *imageIn8_1_2 = readpng_file_to_array(
        "../dataset/1_left_index_4_cam2.png", width, height);
    imageInList[17] = imageIn8_1_2;
    uint8_t *imageIn9_9_1 = readpng_file_to_array(
        "../dataset/9_right_index_1_cam1.png", width, height);
    imageInList[18] = imageIn9_9_1;
    uint8_t *imageIn9_9_2 = readpng_file_to_array(
        "../dataset/9_right_index_1_cam2.png", width, height);
    imageInList[19] = imageIn9_9_2;

    uint8_t *imageIn10_13_1 = readpng_file_to_array(
        "../dataset/13_left_index_5_cam1.png", width, height);
    imageInList[20] = imageIn10_13_1;
    uint8_t *imageIn10_13_2 = readpng_file_to_array(
        "../dataset/13_left_index_5_cam2.png", width, height);
    imageInList[21] = imageIn10_13_2;
    uint8_t *imageIn11_12_1 = readpng_file_to_array(
        "../dataset/12_right_middle_2_cam1.png", width, height);
    imageInList[22] = imageIn11_12_1;
    uint8_t *imageIn11_12_2 = readpng_file_to_array(
        "../dataset/12_right_middle_2_cam2.png", width, height);
    imageInList[23] = imageIn11_12_2;
    uint8_t *imageIn12_9_1 = readpng_file_to_array(
        "../dataset/9_left_index_4_cam1.png", width, height);
    imageInList[24] = imageIn12_9_1;
    uint8_t *imageIn12_9_2 = readpng_file_to_array(
        "../dataset/9_left_index_4_cam2.png", width, height);
    imageInList[25] = imageIn12_9_2;
    uint8_t *imageIn13_6_1 = readpng_file_to_array(
        "../dataset/6_right_index_2_cam1.png", width, height);
    imageInList[26] = imageIn13_6_1;
    uint8_t *imageIn13_6_2 = readpng_file_to_array(
        "../dataset/6_right_index_2_cam2.png", width, height);
    imageInList[27] = imageIn13_6_2;
    uint8_t *imageIn14_4_1 = readpng_file_to_array(
        "../dataset/4_right_middle_1_cam1.png", width, height);
    imageInList[28] = imageIn14_4_1;
    uint8_t *imageIn14_4_2 = readpng_file_to_array(
        "../dataset/4_right_middle_1_cam2.png", width, height);
    imageInList[29] = imageIn14_4_2;

    uint8_t *imageIn15_10_1 = readpng_file_to_array(
        "../dataset/10_right_index_5_cam1.png", width, height);
    imageInList[30] = imageIn15_10_1;
    uint8_t *imageIn15_10_2 = readpng_file_to_array(
        "../dataset/10_right_index_5_cam2.png", width, height);
    imageInList[31] = imageIn15_10_2;
    uint8_t *imageIn16_17_1 = readpng_file_to_array(
        "../dataset/17_right_index_1_cam1.png", width, height);
    imageInList[32] = imageIn16_17_1;
    uint8_t *imageIn16_17_2 = readpng_file_to_array(
        "../dataset/17_right_index_1_cam2.png", width, height);
    imageInList[33] = imageIn16_17_2;
    uint8_t *imageIn17_13_1 = readpng_file_to_array(
        "../dataset/13_right_middle_4_cam1.png", width, height);
    imageInList[34] = imageIn17_13_1;
    uint8_t *imageIn17_13_2 = readpng_file_to_array(
        "../dataset/13_right_middle_4_cam2.png", width, height);
    imageInList[35] = imageIn17_13_2;
    uint8_t *imageIn18_1_1 = readpng_file_to_array(
        "../dataset/1_left_index_3_cam1.png", width, height);
    imageInList[36] = imageIn18_1_1;
    uint8_t *imageIn18_1_2 = readpng_file_to_array(
        "../dataset/1_left_index_3_cam2.png", width, height);
    imageInList[37] = imageIn18_1_2;
    uint8_t *imageIn19_9_1 = readpng_file_to_array(
        "../dataset/9_right_index_3_cam1.png", width, height);
    imageInList[38] = imageIn19_9_1;
    uint8_t *imageIn19_9_2 = readpng_file_to_array(
        "../dataset/9_right_index_3_cam2.png", width, height);
    imageInList[39] = imageIn19_9_2;

    // Benchmark register_fingervein_single
    printf("\nTesting for register_fingervein_single started.\n\n");
    for (int i = 0; i < 20; i++) {
        register_fingervein_single(width, height, 1, &modelOut, imageIn0_13_1);
        free_model(modelOut);
    }

    for (int i = 0; i < 40; i++) {
        t0 = clock();
        register_fingervein_single(width, height, (i % 2) + 1, &modelOut,
                                   imageInList[i]);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
    }
    // Benchmark register_fingerveins
    printf("\nTesting for register_fingerveins started.\n\n");
    for (int i = 0; i < 20; i += 2) {
        register_fingerveins(width, height, &modelOut, imageIn0_13_1,
                             imageIn0_13_2);
        free_model(modelOut);
    }

    for (int i = 0; i < 40; i += 2) {
        t0 = clock();
        register_fingerveins(width, height, &modelOut, imageInList[i],
                             imageInList[i + 1]);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
    }

    // Benchmark match_finger_single same finger without caching
    printf("\nTesting for compare_single_same_wo_caching started.\n\n");
    for (int i = 0; i < 20; i++) {
        size_t l = register_fingervein_single(width, height, 1, &modelOut,
                                              imageIn0_13_1);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input_single(width, height, 1, imageIn10_13_1,
                                        modelOut, l, pC);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    for (int i = 0; i < 20; i++) {
        size_t l = register_fingervein_single(width, height, (i % 2) + 1,
                                              &modelOut, imageInList[i]);
        struct probeCache *pC = new_probeCache();
        t0 = clock();
        compare_model_with_input_single(width, height, (i % 2) + 1,
                                        imageInList[i + 20], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    // Benchmark match_fingers same finger without caching
    printf("\nTesting for compare_same_wo_caching started.\n\n");
    for (int i = 0; i < 20; i += 2) {
        size_t l = register_fingerveins(width, height, &modelOut, imageIn0_13_1,
                                        imageIn0_13_2);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input(width, height, 0.55, imageIn10_13_1,
                                 imageIn10_13_2, modelOut, l, pC);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    for (int i = 0; i < 20; i += 2) {
        size_t l = register_fingerveins(width, height, &modelOut,
                                        imageInList[i], imageInList[i + 1]);
        struct probeCache *pC = new_probeCache();
        t0 = clock();
        compare_model_with_input(width, height, 0.55, imageInList[i + 20],
                                 imageInList[i + 21], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    // Benchmark match_finger_single same finger with caching
    printf("\nTesting for compare_single_same_w_caching started.\n\n");
    for (int i = 0; i < 20; i++) {
        size_t l = register_fingervein_single(width, height, 1, &modelOut,
                                              imageIn0_13_1);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input_single(width, height, 1, imageIn10_13_1,
                                        modelOut, l, pC);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    for (int i = 0; i < 20; i++) {
        size_t l = register_fingervein_single(width, height, (i % 2) + 1,
                                              &modelOut, imageInList[i]);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input_single(width, height, (i % 2) + 1,
                                        imageInList[i + 20], modelOut, l, pC);
        t0 = clock();
        compare_model_with_input_single(width, height, (i % 2) + 1,
                                        imageInList[i + 20], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    // Benchmark match_fingers same finger with caching
    printf("\nTesting for compare_same_w_caching started.\n\n");
    for (int i = 0; i < 20; i += 2) {
        size_t l = register_fingerveins(width, height, &modelOut, imageIn0_13_1,
                                        imageIn0_13_2);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input(width, height, 0.55, imageIn10_13_1,
                                 imageIn10_13_2, modelOut, l, pC);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    for (int i = 0; i < 20; i += 2) {
        size_t l = register_fingerveins(width, height, &modelOut,
                                        imageInList[i], imageInList[i + 1]);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input(width, height, 0.55, imageInList[i + 20],
                                 imageInList[i + 21], modelOut, l, pC);
        t0 = clock();
        compare_model_with_input(width, height, 0.55, imageInList[i + 20],
                                 imageInList[i + 21], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    // Benchmark match_finger_single different finger without caching
    printf("\nTesting for compare_single_diff_wo_caching started.\n\n");
    for (int i = 0; i < 20; i++) {
        size_t l = register_fingervein_single(width, height, 1, &modelOut,
                                              imageIn0_13_1);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input_single(width, height, 1, imageIn10_13_1,
                                        modelOut, l, pC);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    for (int i = 0; i < 40; i += 4) {
        size_t l = register_fingervein_single(width, height, 1, &modelOut,
                                              imageInList[i]);
        struct probeCache *pC = new_probeCache();
        t0 = clock();
        compare_model_with_input_single(width, height, (i % 2) + 1,
                                        imageInList[i + 2], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);

        l = register_fingervein_single(width, height, 2, &modelOut,
                                       imageInList[i + 1]);
        pC = new_probeCache();
        t0 = clock();
        compare_model_with_input_single(width, height, (i % 2) + 1,
                                        imageInList[i + 3], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    // Benchmark match_fingers different finger without caching
    printf("\nTesting for compare_diff_wo_caching started.\n\n");
    for (int i = 0; i < 20; i += 2) {
        size_t l = register_fingerveins(width, height, &modelOut, imageIn0_13_1,
                                        imageIn0_13_2);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input(width, height, 0.55, imageIn10_13_1,
                                 imageIn10_13_2, modelOut, l, pC);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    for (int i = 0; i < 40; i += 4) {
        size_t l = register_fingerveins(width, height, &modelOut,
                                        imageInList[i], imageInList[i + 1]);
        struct probeCache *pC = new_probeCache();
        t0 = clock();
        compare_model_with_input(width, height, 0.55, imageInList[i + 2],
                                 imageInList[i + 3], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    // Benchmark match_finger_single different finger with caching
    printf("\nTesting for compare_single_diff_w_caching started.\n\n");
    for (int i = 0; i < 20; i++) {
        size_t l = register_fingervein_single(width, height, 1, &modelOut,
                                              imageIn0_13_1);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input_single(width, height, 1, imageIn10_13_1,
                                        modelOut, l, pC);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    for (int i = 0; i < 40; i += 4) {
        size_t l = register_fingervein_single(width, height, 1, &modelOut,
                                              imageInList[i]);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input_single(width, height, (i % 2) + 1,
                                        imageInList[i + 2], modelOut, l, pC);
        t0 = clock();
        compare_model_with_input_single(width, height, (i % 2) + 1,
                                        imageInList[i + 2], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);

        l = register_fingervein_single(width, height, 2, &modelOut,
                                       imageInList[i + 1]);
        pC = new_probeCache();
        compare_model_with_input_single(width, height, (i % 2) + 1,
                                        imageInList[i + 3], modelOut, l, pC);
        t0 = clock();
        compare_model_with_input_single(width, height, (i % 2) + 1,
                                        imageInList[i + 3], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    // Benchmark match_fingers different finger without caching
    printf("\nTesting for compare_diff_w_caching started.\n\n");
    for (int i = 0; i < 20; i += 2) {
        size_t l = register_fingerveins(width, height, &modelOut, imageIn0_13_1,
                                        imageIn0_13_2);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input(width, height, 0.55, imageIn10_13_1,
                                 imageIn10_13_2, modelOut, l, pC);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    for (int i = 0; i < 40; i += 4) {
        size_t l = register_fingerveins(width, height, &modelOut,
                                        imageInList[i], imageInList[i + 1]);
        struct probeCache *pC = new_probeCache();
        compare_model_with_input(width, height, 0.55, imageInList[i + 2],
                                 imageInList[i + 3], modelOut, l, pC);
        t0 = clock();
        compare_model_with_input(width, height, 0.55, imageInList[i + 2],
                                 imageInList[i + 3], modelOut, l, pC);
        t1 = clock();
        time_taken = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
        printf("%f\n", time_taken);
        free_model(modelOut);
        freeProbeCache(pC);
    }

    return 0;
}
