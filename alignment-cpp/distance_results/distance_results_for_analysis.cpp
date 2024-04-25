#include "../fingervein_extraction.h"
#include "../pipeline.hpp"
#include "../mask_extraction.hpp"
#include "../prealignment.hpp"
#include "../extraction.hpp"
#include "../postalignment.hpp"
#include "../distance.hpp"
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

    std::string distResPath("./distance_results/");
    #ifdef WITH_OPT
    std::string pathApped("after_optimizations/")
    distResPath += pathAppend;
    #endif
    std::string sameFingerDist("distances_same_finger.csv");
    std::string diffFingerDist("distances_different_finger.csv");

    std::ofstream sameDist(distResPath + time + sameFingerDist);
    std::ofstream diffDist(distResPath + time + diffFingerDist);

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

    std::cout << "Run pipeline for each image." << std::endl;

    // TODO: run pipeline for each image and save result in vector
    std::vector<nc::NdArray<bool>> pipelinedImages;
    std::vector<std::string> pipelinedImagesNames;

    int i = 0;
    for (auto it = files.begin(); it != files.end(); it++, i++) {

        std::string filename = (*it).string();
        std::string fileNameShort = (*it).stem().string();
        char camPersp = fileNameShort.back();

        std::cout << "\r(" << i << ") " << fileNameShort << "                 " << std::flush;

        uint8_t* imageIn = readpng_file_to_array((&filename)->c_str(), width, height);
        nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);

        nc::NdArray<bool> model = run_pipeline(width, height, (camPersp - '0'), &image);

        pipelinedImages.push_back(model);
        pipelinedImagesNames.push_back(fileNameShort);

    }

    std::cout << "\rRunning pipeline finished.                        " << std::endl;

    std::vector<std::tuple<std::string, std::vector<double>>> distSame;
    std::vector<std::tuple<std::string, std::vector<double>>> distDiff;

    std::cout << "Results have been computed for:" << std::endl;

    i = 0;
    for (auto it = pipelinedImages.begin(); it != pipelinedImages.end(); it++, i++) {
        std::string fileName1Short = pipelinedImagesNames.at(i);
        std::string fileIdentifier1 = fileName1Short.substr(0, 13);
        char camPersp1 = fileName1Short.back();

        std::cout << "\r(" << i << ") Processing " << fileName1Short << "...                        " << std::flush;

        nc::NdArray<bool> veins1 = (*it);

        std::vector<double> distSame1Image;
        std::vector<double> distDiff1Image;
        
        int j = i;
        for (auto it2 = it; it2 != pipelinedImages.end(); it2++, j++) {
            std::string fileName2Short = pipelinedImagesNames.at(j);
            std::string fileIdentifier2 = fileName2Short.substr(0, 13);
            char camPersp2 = fileName2Short.back();

            nc::NdArray<bool> veins2 = (*it2);

            if (camPersp1 == camPersp2) {

                veins2 = miura_matching(veins2, veins1, width, height);
                double dist = compute_miura_distance(veins1, veins2);
                
                if (fileIdentifier1.compare(fileIdentifier2) == 0) {
                    distSame1Image.push_back(dist);
                    continue;
                }

                distDiff1Image.push_back(dist);
            }
        }

        distSame.push_back({fileName1Short, distSame1Image});
        distDiff.push_back({fileName1Short, distDiff1Image});
    }
    std::cout << std::endl;

    for (auto k = distSame.begin(); k != distSame.end(); k++) {
        std::tuple<std::string, std::vector<double>> sameElem = (*k);
        std::string sameElemName = std::get<0>(sameElem);
        std::vector<double> sameElemDist = std::get<1>(sameElem);

        sameDist << sameElemName << ", ";

        for (auto l = sameElemDist.begin(); l != sameElemDist.end(); l++) {
            sameDist << (*l) << ", ";
        }

        sameDist << std::endl;
    }
    sameDist.close();

    for (auto k = distDiff.begin(); k != distDiff.end(); k++) {
        std::tuple<std::string, std::vector<double>> diffElem = (*k);
        std::string diffElemName = std::get<0>(diffElem);
        std::vector<double> diffElemDist = std::get<1>(diffElem);

        diffDist << diffElemName << ", ";

        for (auto l = diffElemDist.begin(); l != diffElemDist.end(); l++) {
            diffDist << (*l) << ", ";
        }

        diffDist << std::endl;
    }
    diffDist.close();
    return 0;
}