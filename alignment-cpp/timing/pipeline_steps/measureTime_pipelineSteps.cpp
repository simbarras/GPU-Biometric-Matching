#include "../../fingervein_extraction.h"
#include "../../pipeline.hpp"
#include "../../mask_extraction.hpp"
#include "../../prealignment.hpp"
#include "../../extraction.hpp"
#include "../../postalignment.hpp"
#include "../../distance.hpp"
#include <filesystem>
#include "png.h"
#include <ctime>
#include <ratio>
#include <chrono>
#include "NumCpp.hpp"

#include <iostream>
#include <fstream>
#include <ctime>

/**
 * This function reads out a 8-bit grayscale png file and writes it into a NdArray.
 * 
 * @param[in] filename: The file path as a string pointer that leads to the image that
 * will run through the pipeline. It needs to be constant.
 * @param[in] wid: The expected width of the image given as a constant integer.
 * @param[in] hei: The expected height of the image given as a constant integer.
 * @returns A pointer of type uint8_t that contains the 8-bit grayscale 
 * pixel values of the image. 
*/
uint8_t* readpng_file_to_array(const char* filename, const int wid, const int hei) {

    // Open the file and abort if there is an error
    FILE *fp = fopen(filename, "rb");
    if (!fp) abort();

    // Create structures needed for reading the PNG
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
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
    uint8_t** row_pointers = nullptr;

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
    png_destroy_read_struct(&png, &info, nullptr);

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

int main () {

    std::timespec ts;
    std::timespec_get(&ts, TIME_UTC);
    char buf[100];
    std::strftime(buf, sizeof buf, "%FT%TZ", std::gmtime(&ts.tv_sec));
    std::string time(buf);

    std::chrono::high_resolution_clock::time_point timeNow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> zeroDuration = std::chrono::duration_cast<std::chrono::duration<double>>(timeNow - timeNow);

    std::string timePath("./timing/pipeline_steps/");
    std::string edgeMaskPath("edge_mask/");
    std::string prealigmentPath("prealignment/");
    std::string maxCurvPath("maximum_curvature/");
    std::string csvEnding(".csv");

    std::ofstream maskFile(timePath + edgeMaskPath + time + csvEnding);
    std::ofstream prealignmentFile(timePath + prealigmentPath + time + csvEnding);
    std::ofstream maxCurvFile(timePath + maxCurvPath + time + csvEnding);

    int width = 376;
    int height = 240;
    // Traverse over all images in dataset and test/benchmark
    const std::filesystem::path dataset{"../dataset/"};
    std::vector<std::filesystem::path> files;

    for (std::filesystem::directory_entry const& dir_entry : std::filesystem::directory_iterator{dataset}) {
        if (dir_entry.is_regular_file() || dir_entry.is_symlink()) {
            files.push_back(std::filesystem::path(dir_entry.path()));
        }
    }

    std::sort(files.begin(), files.end());

    std::vector<std::vector<std::chrono::duration<double>>> timesMask;
    std::vector<std::vector<std::chrono::duration<double>>> timesPreal;
    std::vector<std::vector<std::chrono::duration<double>>> timesMCurv;

    std::cout << "Warm-up started." << std::endl;

    // Warm-up caches
    uint8_t* imageIn = readpng_file_to_array("../dataset/0_left_index_1_cam1.png", width, height);
    nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);

    for (int i = 0; i < 100; i++) {
        nc::NdArray<bool> model = run_pipeline(width, height, 1, &image);
    }

    std::cout << "Warm-up finished." << std::endl << "Timing for pipeline steps done for:" << std::endl;

    int j = 0;
    for (auto it = files.begin(); it != files.end(); it++, j++) {

        std::string filename = (*it).string();
        std::string fileNameShort = (*it).stem().string();
        char camPersp = fileNameShort.back();

        uint8_t* imageIn = readpng_file_to_array((&filename)->c_str(), width, height);
        nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);

        std::vector<std::chrono::duration<double>> timesMaskPerImage;
        std::vector<std::chrono::duration<double>> timesPrealPerImage;
        std::vector<std::chrono::duration<double>> timesMCurvPerImage;

        nc::NdArray<uint8_t> mask;
        nc::NdArray<double> res;

        for (int i = 0; i < 15; i++) {
                std::chrono::high_resolution_clock::time_point timeMEStart = std::chrono::high_resolution_clock::now();
                mask = edge_mask_extraction(image, (camPersp - '0'), width, height);
                std::chrono::high_resolution_clock::time_point timeMEEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_spanME = std::chrono::duration_cast<std::chrono::duration<double>>(timeMEEnd - timeMEStart);

                timesMaskPerImage.push_back(time_spanME);

        }

        for (int i = 0; i < 15; i++) {
                nc::NdArray<uint8_t> imageRep = image;
                std::chrono::high_resolution_clock::time_point timeTAStart = std::chrono::high_resolution_clock::now();
                res = translation_alignment(imageRep, mask, width, height);
                std::chrono::high_resolution_clock::time_point timeTAEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_spanTA = std::chrono::duration_cast<std::chrono::duration<double>>(timeTAEnd - timeTAStart);

                timesPrealPerImage.push_back(time_spanTA);

        }

        for (int i = 0; i < 15; i++) {
                std::chrono::high_resolution_clock::time_point timeMCStart = std::chrono::high_resolution_clock::now();
                nc::NdArray<bool> veins = maximum_curvature(image, res, width, height);
                std::chrono::high_resolution_clock::time_point timeMCEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_spanMC = std::chrono::duration_cast<std::chrono::duration<double>>(timeMCEnd - timeMCStart);

                timesMCurvPerImage.push_back(time_spanMC);

        }

        timesMask.push_back(timesMaskPerImage);
        timesPreal.push_back(timesPrealPerImage);
        timesMCurv.push_back(timesMCurvPerImage);

        std::cout << "(" << j << ") " << fileNameShort << std::endl;

    }

    for (int i = 0; i < timesMask.size(); i++) {
        std::string fileName = files.at(i).stem().string();
        maskFile << files.at(i).stem().string() << ", ";
        prealignmentFile << files.at(i).stem().string() << ", ";
        maxCurvFile << files.at(i).stem().string() << ", ";

        std::vector<std::chrono::duration<double>> timesPerImage = timesMask.at(i);

        for (auto it = timesPerImage.begin(); it != timesPerImage.end(); it++) {
            maskFile << (*it).count() << ", ";
        }

        maskFile << std::endl;

        timesPreal.at(i);
        for (auto it = timesPerImage.begin(); it != timesPerImage.end(); it++) {
            prealignmentFile << (*it).count() << ", ";
        }

        prealignmentFile << std::endl;

        timesPerImage = timesMCurv.at(i);
        for (auto it = timesPerImage.begin(); it != timesPerImage.end(); it++) {
            maxCurvFile << (*it).count() << ", ";
        }

        maxCurvFile << std::endl;

    }

    maskFile.close();
    prealignmentFile.close();
    maxCurvFile.close();



    return 0;
}