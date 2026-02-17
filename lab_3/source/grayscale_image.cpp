#include "grayscale_image.h"
#include <iostream>
#include <omp.h>
#include "intermediate_image.h"


void GrayscaleImage::convert_bitmap(BitmapImage& bitmap){
    this->height = bitmap.get_height();
    this->width = bitmap.get_width();

    this->pixels.resize(this->height * this->width);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < static_cast<int>(height); ++y) {
        for (int x = 0; x < static_cast<int>(width); ++x) {
            const auto pixel = bitmap.get_pixel(static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(x));

            const double R = static_cast<double>(pixel.get_red_channel());
            const double G = static_cast<double>(pixel.get_green_channel());
            const double B = static_cast<double>(pixel.get_blue_channel());

            double L = 0.2126 * R + 0.7152 * G + 0.0722 * B + 0.5;

            this->pixels[y * width + x] = static_cast<std::uint8_t>(L);
        }
    }
}

void GrayscaleImage::convert_intermediate_image(IntermediateImage& image){
    image.update_min_pixel_value();
    image.update_max_pixel_value();

    this->height = image.height;
    this->width = image.width;
    this->pixels.resize(this->height * this->width);

    double min_val = image.min_pixel_value;
    double max_val = image.max_pixel_value;

    double effective_min = min_val;
    double effective_max = max_val;

    if (min_val >= 0.0 && max_val <= 255.0) {
        effective_min = 0.0;
        effective_max = 255.0;
    }

    const double range = effective_max - effective_min;

    const double scale_factor = (range > 1e-9) ? (255.0 / range) : 0.0;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < static_cast<int>(height); ++y) {
        for (int x = 0; x < static_cast<int>(width); ++x) {

            std::uint32_t idx = static_cast<std::uint32_t>(y) * width + static_cast<std::uint32_t>(x);

            const double v = image.pixels[idx];
            double g_val = (v - effective_min) * scale_factor;

            if (g_val < 0.0) g_val = 0.0;
            if (g_val > 255.0) g_val = 255.0;


            this->pixels[idx] = static_cast<std::uint8_t>(g_val);
        }
    }
}