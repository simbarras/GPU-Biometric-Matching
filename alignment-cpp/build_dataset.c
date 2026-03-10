#include "fingervein_extraction.h"
#include "read_png.h"
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PATH_LENGTH 256
#define COMMAND_LENGTH 512
#define IMG_WIDTH 376
#define IMG_HEIGHT 240

/**
 * Saves the model data to a binary file
 * @param output_path The path to the output file
 * @param model_data The model data to save
 * @param length The length of the model data
 * @return 0 if successful, 1 if an error occurred
 */
int save_model(const char *output_path, uint8_t *model_data, size_t length) {
    FILE *file = fopen(output_path, "wb");
    if (file == NULL) {
        perror("Unable to open file for writing");
        return 1;
    }
    fwrite(model_data, sizeof(uint8_t), length, file);
    fclose(file);
    return 0;
}

/**
 * This function takes the input folder, data name and output folder as
 * arguments and reads the corresponding png files for the two camera
 * perspectives, extracts the Fourier transform of the vein images using the
 * register_fingervein_single and register_fingerveins functions, and saves the
 * 3 output to the output folder in binary
 * @param input_folder The folder where the input png files are located
 * @param data_name The name of the data file (without camera perspective and
 * extension)
 * @param output_folder The folder where the output binary files will be saved
 * @return 0 if successful, 1 if an error occurred
 */
int convert_files(char *input_folder, char *data_name, char *output_folder) {
    // Create the corresponding filename for the other camera perspective
    char filepath_cam1[PATH_LENGTH];
    snprintf(filepath_cam1, sizeof(filepath_cam1), "%s/%s_cam1.png",
             input_folder, data_name);
    char filepath_cam2[PATH_LENGTH];
    snprintf(filepath_cam2, sizeof(filepath_cam2), "%s/%s_cam2.png",
             input_folder, data_name);

    // Read the png
    uint8_t *cam1_data =
        readpng_file_to_array(filepath_cam1, IMG_WIDTH, IMG_HEIGHT);
    uint8_t *cam2_data =
        readpng_file_to_array(filepath_cam2, IMG_WIDTH, IMG_HEIGHT);

    // Compute Fourier transform
    uint8_t *model_out_cam1;
    size_t length_cam1 = register_fingervein_single(IMG_WIDTH, IMG_HEIGHT, 1,
                                                    &model_out_cam1, cam1_data);
    uint8_t *model_out_cam2;
    size_t length_cam2 = register_fingervein_single(IMG_WIDTH, IMG_HEIGHT, 2,
                                                    &model_out_cam2, cam2_data);
    uint8_t *model_out_both;
    size_t length_both = register_fingerveins(
        IMG_WIDTH, IMG_HEIGHT, &model_out_both, cam1_data, cam2_data);

    // Save model_out_cam1, model_out_cam2 and model_out_both to output_folder
    char output_path_cam1[PATH_LENGTH];
    snprintf(output_path_cam1, sizeof(output_path_cam1), "%s/cam1/%s.bin",
             output_folder, data_name);
    char output_path_cam2[PATH_LENGTH];
    snprintf(output_path_cam2, sizeof(output_path_cam2), "%s/cam2/%s.bin",
             output_folder, data_name);
    char output_path_both[PATH_LENGTH];
    snprintf(output_path_both, sizeof(output_path_both), "%s/both/%s.bin",
             output_folder, data_name);

    // Save models
    int result = save_model(output_path_cam1, model_out_cam1, length_cam1);
    result |= save_model(output_path_cam2, model_out_cam2, length_cam2);
    result |= save_model(output_path_both, model_out_both, length_both);

    // Free memory and return
    free(cam1_data);
    free(cam2_data);
    free_model(model_out_cam1);
    free_model(model_out_cam2);
    free_model(model_out_both);

    return result;
}

/**
 * @file build_dataset.c
 * @author Simbarras
 * @brief This file is used to take the raw png images from the dataset and
 * extract the Fourier transform of the vein images to create a new dataset.
 * @version 0.1
 * @date 2026-03-06
 */
int main(int argc, char *argv[]) {
    // Get arguments
    if (argc != 3) {
        printf("Usage: %s <png_folder> <output_folder>\n", argv[0]);
        return 1;
    }
    char *png_folder = argv[1];
    char *output_folder = argv[2];

    // Read png_folder files
    DIR *dir = opendir(png_folder);
    if (dir == NULL) {
        perror("Unable to read directory");
        return 1;
    }

    // Loop through files in png_folder
    char previous_data_name[PATH_LENGTH] = "";
    struct dirent *entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        // Check if file is a png
        char *filename = entry->d_name;
        if (strstr(filename, ".png") == NULL)
            continue;

        // Get only the name of the file without the camera perspective and
        // extension
        char data_name[PATH_LENGTH];
        strncpy(data_name, filename, sizeof(data_name) - 1);
        data_name[sizeof(data_name) - 1] = '\0';
        char *underscore = strrchr(data_name, '_');
        if (underscore != NULL) {
            *underscore = '\0';
        }

        // Check if we have already processed this data_name
        if (strcmp(data_name, previous_data_name) == 0) {
            continue;
        }
        strncpy(previous_data_name, data_name, sizeof(previous_data_name) - 1);
        previous_data_name[sizeof(previous_data_name) - 1] = '\0';

        printf("Process file: %s\n", data_name);
        if (convert_files(png_folder, data_name, output_folder) != 0) {
            perror("Error occurred while converting files");
            break;
        }
    }

    closedir(dir);
    return 0;
}
