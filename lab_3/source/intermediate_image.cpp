#include "intermediate_image.h"
#include <omp.h>
#include <limits>

void IntermediateImage::resize(std::uint32_t new_height, std::uint32_t new_width){
    height = new_height;
    width = new_width;
    pixels.resize(new_height * new_width);
}

void IntermediateImage::load_grayscale_image(GrayscaleImage& image){
    resize(image.height, image.width);
    #pragma omp parallel for shared(pixels, image) collapse(2)
    for(std::int32_t i = 0; i < static_cast<std::int32_t>(image.height); ++i){
        for(std::int32_t j = 0; j < static_cast<std::int32_t>(image.width); ++j){
            pixels[i * image.width + j] = static_cast<double>(image.pixels[i * image.width + j]);
        }
    }
}

void IntermediateImage::update_min_pixel_value(){
    if(height == 0 || width == 0){
        min_pixel_value = 0;
        return;
    }

    double global_min = std::numeric_limits<double>::infinity();

    #pragma omp parallel
    {
        double local_min = std::numeric_limits<double>::infinity();
        #pragma omp for nowait
        for(std::int32_t i = 0; i < static_cast<std::int32_t>(height * width); ++i){
            if(pixels[i] < local_min) {
                local_min = pixels[i];
            }
        }
        #pragma omp critical
        {
            if(local_min < global_min) {
                global_min = local_min;
            }
        }
    }
    min_pixel_value = global_min;
}

void IntermediateImage::update_max_pixel_value(){
    if(height == 0 || width == 0){
        max_pixel_value = 0;
        return;
    }

    double global_max = -std::numeric_limits<double>::infinity();

    #pragma omp parallel
    {
        double local_max = -std::numeric_limits<double>::infinity();
        #pragma omp for nowait
        for(std::int32_t i = 0; i < static_cast<std::int32_t>(height * width); ++i){
            if(pixels[i] > local_max) {
                local_max = pixels[i];
            }
        }
        #pragma omp critical
        {
            if(local_max > global_max) {
                global_max = local_max;
            }
        }
    }
    max_pixel_value = global_max;
}