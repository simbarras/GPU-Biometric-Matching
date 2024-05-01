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

int main (int argc, char** argv) {

    if (argc != 3) {
        std::cout << "Invalid number of arguments. Please provide min and max." << std::endl;
        exit(1);
    }

    size_t min_image_no = atoi(argv[1]);
    size_t max_image_no = atoi(argv[2]);

    std::timespec ts;
    std::timespec_get(&ts, TIME_UTC);
    char buf[100];
    std::strftime(buf, sizeof buf, "%FT%TZ", std::gmtime(&ts.tv_sec));
    std::string time(buf);

    std::chrono::high_resolution_clock::time_point timeNow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> zeroDuration = std::chrono::duration_cast<std::chrono::duration<double>>(timeNow - timeNow);

    std::string distResPath("./timing/pipeline_steps/");
    std::string postalignmentPath("postalignment/");
    std::string distancePath("distance/");
    std::string csvEnding(".csv");

    std::ofstream postFile(distResPath + postalignmentPath + time + csvEnding);
    std::ofstream distFile(distResPath + distancePath + time + csvEnding);

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

        if (i < min_image_no || i >= max_image_no)
            continue;

        std::string filename = (*it).string();
        std::string fileNameShort = (*it).stem().string();
        char camPersp = fileNameShort.back();

        std::cout << "(" << i << ") Processing " << fileNameShort << "...                        " << std::endl;

        uint8_t* imageIn = readpng_file_to_array((&filename)->c_str(), width, height);
        nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);

        nc::NdArray<bool> model = run_pipeline(width, height, (camPersp - '0'), &image);

        pipelinedImages.push_back(model);
        pipelinedImagesNames.push_back(fileNameShort);

    }

    std::cout << "Running pipeline finished.                        " << std::endl;

    std::vector<std::tuple<std::string, std::vector<std::chrono::duration<double>>>> timesPost;
    std::vector<std::tuple<std::string, std::vector<std::chrono::duration<double>>>> timesDist;

    std::cout << "Results have been computed for:" << std::endl;

    i = 0;
    for (auto it = pipelinedImages.begin(); it != pipelinedImages.end(); it++, i++) {
        std::string fileName1Short = pipelinedImagesNames.at(i);
        std::string fileIdentifier1 = fileName1Short.substr(0, 13);
        char camPersp1 = fileName1Short.back();

        nc::NdArray<bool> veins1 = (*it);

        std::cout << "(" << i << ") Processing " << fileName1Short << "...                        " << std::endl;

        std::vector<std::chrono::duration<double>> timesPostPerImage;
        std::vector<std::chrono::duration<double>> timesDistPerImage;
        
        int j = i;
        for (auto it2 = it; it2 != pipelinedImages.end(); it2++, j++) {
            std::string fileName2Short = pipelinedImagesNames.at(j);
            std::string fileIdentifier2 = fileName2Short.substr(0, 13);
            char camPersp2 = fileName2Short.back();

            if (camPersp1 == camPersp2) {

                nc::NdArray<bool> veins2;
                std::chrono::duration<double> timeSpanMI = zeroDuration;
                for (int i = 0; i < 10; i++) {
                    veins2 = (*it2);
                    std::chrono::high_resolution_clock::time_point timeMMStart = std::chrono::high_resolution_clock::now();
                    veins2 = miura_matching(veins2, veins1, width, height);
                    std::chrono::high_resolution_clock::time_point timeMMEnd = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> time_spanMM = std::chrono::duration_cast<std::chrono::duration<double>>(timeMMEnd - timeMMStart);

                    // either save time spans in file or do some statistical computation here and output result
                    timeSpanMI += time_spanMM;
                }
                timesPostPerImage.push_back(timeSpanMI / 10.);

                std::chrono::duration<double> timeSpanDI = zeroDuration;
                for (int i = 0; i < 10; i++) {
                    std::chrono::high_resolution_clock::time_point timeDStart = std::chrono::high_resolution_clock::now();
                    double dist = compute_miura_distance(veins1, veins2);
                    std::chrono::high_resolution_clock::time_point timeDEnd = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> time_spanD = std::chrono::duration_cast<std::chrono::duration<double>>(timeDEnd - timeDStart);

                    // either save time spans in file or do some statistical computation here and output result
                    timeSpanDI += time_spanD;
                }
                timesDistPerImage.push_back(timeSpanDI / 10.);
            }
        }

        timesPost.push_back({fileName1Short, timesPostPerImage});
        timesDist.push_back({fileName1Short, timesDistPerImage});
    }

    std::cout << std::endl;

    for (auto k = timesPost.begin(); k != timesPost.end(); k++) {
        std::tuple<std::string, std::vector<std::chrono::duration<double>>> elem = (*k);
        std::string name = std::get<0>(elem);
        std::vector<std::chrono::duration<double>> times = std::get<1>(elem);

        postFile << name << ", ";

        for (auto l = times.begin(); l != times.end(); l++) {
            postFile << (*l).count() << ", ";
        }

        postFile << std::endl;
    }
    postFile.close();

    for (auto k = timesDist.begin(); k != timesDist.end(); k++) {
        std::tuple<std::string, std::vector<std::chrono::duration<double>>> elem = (*k);
        std::string name = std::get<0>(elem);
        std::vector<std::chrono::duration<double>> times = std::get<1>(elem);

        distFile << name << ", ";

        for (auto l = times.begin(); l != times.end(); l++) {
            distFile << (*l).count() << ", ";
        }

        distFile << std::endl;
    }
    distFile.close();
    return 0;
}