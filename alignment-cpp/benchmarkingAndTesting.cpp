#include "fingervein_extraction.h"
#include "pipeline.hpp"
#include "mask_extraction.hpp"
#include "prealignment.hpp"
#include "extraction.hpp"
#include "postalignment.hpp"
#include "distance.hpp"
#include <filesystem>
#include "png.h"
#include <ctime>
#include <ratio>
#include <chrono>
#include "NumCpp.hpp"

/**
 * This function reads out a 8-bit grayscale png file and writes it into a NdArray.
 * 
 * @param[in] filename: The file path as a string pointer that leads to the image that
 * will run through the pipeline. It needs to be constant.
 * @param[in] wid: The expected width of the image given as a constant integer.
 * @param[in] hei: The expected height of the image given as a constant integer.
 * @returns A NumCpp NdArray of type uint8_t that contains the 8-bit grayscale 
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

    uint8_t* imageIn = readpng_file_to_array("../dataset/0_left_index_1_cam1.png", width, height);
    nc::NdArray<uint8_t> image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);

    for (int i = 0; i < 5; i++) {
        nc::NdArray<bool> model = run_pipeline(width, height, 1, &image, nullptr);
    }

    for (auto it = files.begin(); it != files.end(); it++) {
        std::string fileName1 = (*it).stem().string();
        std::string fileIdentifier1 = fileName1.substr(0, 13);
        char camPersp1 = fileName1.back();
        fileName1 = dataset.string() + fileName1 + ".png";

        imageIn = readpng_file_to_array((&fileName1)->c_str(), width, height);
        image = nc::NdArray<uint8_t>(imageIn, height, width, nc::PointerPolicy::COPY);

        nc::NdArray<bool> model;

        // already benchmark pipeline for single finger
        #ifdef BENCHMARK_PIPELINE

        for (int i = 0; i < 15; i++) {
            std::chrono::high_resolution_clock::time_point timePipStart = std::chrono::high_resolution_clock::now();
            model = run_pipeline(width, height, (camPersp1 - '0'), &image, nullptr);
            std::chrono::high_resolution_clock::time_point timePipEnd = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> time_spanPip = std::chrono::duration_cast<std::chrono::duration<double>>(timePipEnd - timePipStart);

            // either save time spans in file or do some statistical computation here and output result

            //std::cout << "Pipeline for " << fileName1 << " took " << time_spanPip.count() << std::endl;

        }
        #endif

        model = run_pipeline(width, height, (camPersp1 - '0'), &image, nullptr);

        #ifdef BENCHMARK_PIPELINE_STEPS
        for (int i = 0; i < 15; i++) {
            std::chrono::high_resolution_clock::time_point timeMEStart = std::chrono::high_resolution_clock::now();
            nc::NdArray<uint8_t> mask = edge_mask_extraction(image, (camPersp1 - '0'), width, height);
            std::chrono::high_resolution_clock::time_point timeMEEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_spanME = std::chrono::duration_cast<std::chrono::duration<double>>(timeMEEnd - timeMEStart);

            std::chrono::high_resolution_clock::time_point timeTAStart = std::chrono::high_resolution_clock::now();
            std::tuple<nc::NdArray<uint8_t>, nc::NdArray<double>> res = translation_alignment(image, mask, width, height);
            std::chrono::high_resolution_clock::time_point timeTAEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_spanTA = std::chrono::duration_cast<std::chrono::duration<double>>(timeTAEnd - timeTAStart);

            std::chrono::high_resolution_clock::time_point timeMCStart = std::chrono::high_resolution_clock::now();
            nc::NdArray<bool> veins = maximum_curvature(std::get<0>(res), std::get<1>(res), width, height);
            std::chrono::high_resolution_clock::time_point timeMCEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_spanMC = std::chrono::duration_cast<std::chrono::duration<double>>(timeMCEnd - timeMCStart);

            // either save time spans in file or do some statistical computation here and output result
        }
        #endif


        // Benchmark finger comparisons
        for (auto it2 = it; it2 != files.end(); it2++) {
            std::string fileName2 = (*it2).stem().string();
            std::string fileIdentifier2 = fileName2.substr(0, 13);
            char camPersp2 = fileName2.back();
            fileName2 = dataset.string() + fileName2 + ".png";

            if (camPersp1 == camPersp2) {
                uint8_t* imageIn2 = readpng_file_to_array((&fileName2)->c_str(), width, height);
                nc::NdArray<uint8_t> image2 = nc::NdArray<uint8_t>(imageIn2, height, width, nc::PointerPolicy::COPY);

                nc::NdArray<bool> probe = run_pipeline(width, height, (camPersp2 - '0'), &image2, nullptr);

                #ifdef BENCHMARK_PIPELINE_STEPS
                for (int i = 0; i < 10; i++) {
                    std::chrono::high_resolution_clock::time_point timeMMStart = std::chrono::high_resolution_clock::now();
                    probe = miura_matching(probe, model, width, height);
                    std::chrono::high_resolution_clock::time_point timeMMEnd = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> time_spanMM = std::chrono::duration_cast<std::chrono::duration<double>>(timeMMEnd - timeMMStart);

                    // either save time spans in file or do some statistical computation here and output result
                }
                #endif

                probe = miura_matching(probe, model, width, height);

                #ifdef BENCHMARK_COMPARISON 
                for (int i = 0; i < 10; i++) {
                    std::chrono::high_resolution_clock::time_point timeDStart = std::chrono::high_resolution_clock::now();
                    double dist = compute_miura_distance(model, probe);
                    std::chrono::high_resolution_clock::time_point timeDEnd = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> time_spanD = std::chrono::duration_cast<std::chrono::duration<double>>(timeDEnd - timeDStart);

                    // either save time spans in file or do some statistical computation here and output result
                }
                #endif

                #ifndef BENCHMARK_COMPARISON 
                double dist = compute_miura_distance(model, probe);
                #endif
                
                if (fileIdentifier1.compare(fileIdentifier2) == 0 && camPersp1 == camPersp2) {
                    // Benchmarking and Testing for same finger
                    // write distance results in different file than for different fingers
                }

                // Benchmarking and Testing for different finger
                // write distance results in different file than for same fingers
            }
        }
    }
    return 0;
}