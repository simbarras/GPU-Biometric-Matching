#include "pipeline.hpp"

nc::NdArray<uint8_t> readpng_file_to_array(std::string* filename, const int wid, const int hei) {

    // Open the file and abort if there is an error
    FILE *fp = fopen(filename->c_str(), "rb");
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
    png_byte color_type; 
    png_byte bit_depth; 
    uint8_t** row_pointers = nullptr;

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth  = png_get_bit_depth(png, info);

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
    std::array<std::array<uint8_t, 376>, 240> img_arr;

    for(int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++) {
            img_arr[y][x] = row[x];
        }
        free(row);
    }

    free(row_pointers);

    nc::NdArray<uint8_t> img = nc::NdArray<uint8_t>(img_arr);

    return img;
}

nc::NdArray<bool> run_pipeline(const int width, const int height, 
                               int camera_persp, 
                               nc::NdArray<uint8_t>* image, 
                               std::string* image_path, 
                               const nc::NdArray<bool>* modelIn, 
                               std::string* model_path, 
                               bool caching, 
                               std::string* cache_path) {
    
    assert(image != nullptr || image_path != nullptr);

    // Open and load the image to use
    nc::NdArray<uint8_t> img;
    if (image != nullptr) {
        img = *image;
    } else if (image_path != nullptr) {
        img = readpng_file_to_array(image_path, width, height); 
    }

    // Extract mask
    nc::NdArray<uint8_t> mask;
    
    mask = edge_mask_extraction(img, 1, width, height);

    // TODO: Add "caching" of images

    // Prealign image
    std::tuple<nc::NdArray<uint8_t>, nc::NdArray<double>> res;
    res = translation_alignment(img, mask, width, height);

    img = std::get<0>(res);

    nc::NdArray<double> maskD;

    maskD = std::get<1>(res);

    // TODO: Add "caching" of images

    // Extract features
    nc::NdArray<bool> veins = maximum_curvature(img, maskD, width, height);

    // TODO: Add "caching" of images

    // Postalignment with the help of a model
    if (model_path != nullptr || modelIn != nullptr) {
        nc::NdArray<bool> model;
        if (modelIn != nullptr) {
            model = *modelIn;
        } else if (model_path != nullptr) {
            model = readpng_file_to_array(model_path, width, height).astype<bool>();
        }
        veins = miura_matching(veins, model, width, height);
    } 

    // TODO: Add "caching" of images
    
    return veins;
}
