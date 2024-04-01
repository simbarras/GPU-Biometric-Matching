#ifndef FINGERVEIN_EXTRACTION_H
#define FINGERVEIN_EXTRACTION_H
size_t register_fingervein (const int width, const int height, 
                           const int camera_perspective,
                           bool** modelOut,
                           uint8_t* imageIn);

bool compare_model_with_input (const int width, const int height, 
                               const int camera_perspective,
                               uint8_t* imageIn,
                               bool* modelIn,
                               size_t modelSize);

void free_model(bool* model);

#endif
