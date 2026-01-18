#include "loader.h"
#include <bit>
#include <fstream>
#include <iostream>
#include <print>
#include <stdexcept>

std::vector<LabeledImage> loadImages(const std::string& imPath, const std::string& lbPath, uint32_t maxSamples) {
    std::ifstream is(imPath, std::ios::binary);
    std::ifstream ls(lbPath, std::ios::binary);
    if (!is.is_open()) throw std::runtime_error("Unable to open file to images");
    if (!ls.is_open()) throw std::runtime_error("Unable to open file to labels");

    uint32_t im_magic, lb_magic;
    uint32_t im_count, lb_count;
    uint32_t im_rows, im_cols;
    is.read(reinterpret_cast<char*>(&im_magic), 4);
    is.read(reinterpret_cast<char*>(&im_count), 4);
    is.read(reinterpret_cast<char*>(&im_rows), 4);
    is.read(reinterpret_cast<char*>(&im_cols), 4);

    ls.read(reinterpret_cast<char*>(&lb_magic), 4);
    ls.read(reinterpret_cast<char*>(&lb_count), 4);

    im_magic = std::byteswap(im_magic);
    lb_magic = std::byteswap(lb_magic);
    im_count = std::byteswap(im_count);
    lb_count = std::byteswap(lb_count);
    im_rows = std::byteswap(im_rows);
    im_cols = std::byteswap(im_cols);

    if (im_magic != 2051) throw std::runtime_error("Images magic doesnt match");
    if (lb_magic != 2049) throw std::runtime_error("Labels magic doesnt match");
    if (im_count != lb_count) throw std::runtime_error("counts do not match");

    if (maxSamples > 0 && maxSamples < im_count) {
        im_count = lb_count = maxSamples;
    }

    std::println("Loading {} images of size {}x{}", im_count, im_rows, im_cols);

    std::vector<LabeledImage> images(im_count);
    for (size_t i = 0; i < im_count; ++i) {
        uint8_t d; ls.read(reinterpret_cast<char*>(&d), 1);
        images[i].label = (uint8_t)d;
        for (size_t r = 0; r < im_rows; ++r) {
            for (size_t c = 0; c < im_cols; ++c) {
                uint8_t p; is.read(reinterpret_cast<char*>(&p), 1);
                images[i].image[r][c] = static_cast<float>(p) / 255.0f;
            }
        }
    }

    return images;
}
