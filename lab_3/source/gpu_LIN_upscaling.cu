#include "gpu_LIN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>

std::uint32_t get_LIN_upscaled_width(std::uint32_t image_width){
    if (image_width == 0) return 0;
    return (image_width * 2) -1;
}

std::uint32_t get_LIN_upscaled_height(std::uint32_t image_height){
    if (image_height == 0) return 0;
    return (image_height * 2) -1;
}



void LIN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
}